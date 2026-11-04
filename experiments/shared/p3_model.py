#!/usr/bin/env python3
"""
p3_model.py - one loaded model, every channel of every Paper 3 experiment.

Every experiment in Paper 3 needs the same 7B model. Loading it once and handing
the same object to all of them is why the whole paper fits in ONE Colab notebook
per model: a 4-bit load costs several minutes and most of the session's disk
traffic, and there are six experiments.

Nothing here is specific to an experiment. It exposes three operations:

    generate(prompts)                  channel A, free text
    generate_multiturn(triples)        e5, a follow-up on a replayed exchange
    score_continuations(prompts, word) channel B, mean log prob of a word
                                       continuing each prompt
    letter_logprobs(prompts, letters)  channel C, log prob of each single-token
                                       letter at the next position, plus the
                                       probability mass those letters carry
    score_answers(prompts, answers)    channel A2, log prob of the correct
                                       answer as the reply to a chat prompt
    check_answer_scoring(...)          proves score_answers on the real GPU
                                       before a single row is written

THE MASS IS NOT OPTIONAL
------------------------
letter_logprobs returns P(A)+P(B) on every read. Paper 2 learned this the hard
way three times: reading logP("A") against logP("B") is only a judgement if the
model was about to emit a letter at all. On Mistral the mass was 0.40 with 61
percent of reads below half, and a method 6b finding rested on ten usable pairs
out of three hundred before the check withdrew it.

Published work agrees and goes further. "My Answer is C" (arXiv:2402.14499,
Findings of ACL 2024) reports mismatch above 60 percent between the first-token
ranking and the text an instruction-tuned model actually emits, and "Look at the
Text" (arXiv:2404.08382, COLM 2024) argues the text is the more robust read.

So the mass is recorded on every row, the gate is applied in the analysis rather
than silently at collection time, and channel C reports how much data the gate
removed.
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BNB_MIN = "0.46.1"


def check_environment(load_4bit=True):
    """Fail before the model loads, not four minutes into it.

    A 7B at 4-bit needs bitsandbytes. Colab does not ship it, the install can
    fail quietly behind `pip -q`, and transformers then raises deep inside
    from_pretrained after the download has already started. This says what is
    wrong and what to type, before anything is downloaded.
    """
    problems = []
    if not torch.cuda.is_available():
        problems.append(
            "No GPU. Runtime > Change runtime type > T4 GPU, then run again. "
            "On CPU a 7B model is unusable, not merely slow.")
    if load_4bit and torch.cuda.is_available():
        try:
            import bitsandbytes
            have = getattr(bitsandbytes, "__version__", "0")
            def parts(v):
                out = []
                for x in v.split("."):
                    digits = "".join(c for c in x if c.isdigit())
                    out.append(int(digits) if digits else 0)
                return out
            if parts(have) < parts(BNB_MIN):
                problems.append(
                    f"bitsandbytes {have} is older than {BNB_MIN}, which "
                    f"4-bit loading needs.")
        except ImportError as e:
            problems.append(f"bitsandbytes is not installed ({e}).")
    if problems:
        print()
        print("=" * 74)
        print("ENVIRONMENT NOT READY. Nothing was downloaded or run.")
        print("=" * 74)
        for x in problems:
            print(f"  {x}")
        print()
        print("  Fix, in a cell of its own:")
        print(f'      !pip install -U "bitsandbytes>={BNB_MIN}" accelerate '
              f'transformers')
        print("  then Runtime > Restart session, and run the cells again.")
        print("  Work already saved to Drive is not lost and is not redone.")
        print("=" * 74)
        sys.exit(1)

MODELS = {
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}


class P3Model:
    def __init__(self, model_key, load_4bit=True):
        if model_key not in MODELS:
            raise KeyError(f"unknown model key {model_key}")
        self.key = model_key
        self.name = MODELS[model_key]
        check_environment(load_4bit)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"loading {self.name} on {self.device}, 4-bit={load_4bit}")

        # left padding for generation, right padding for scoring. Two
        # tokenizers rather than flipping a flag mid-run, which is a bug that
        # only shows up as quietly wrong numbers.
        self.tok_gen = AutoTokenizer.from_pretrained(self.name,
                                                     padding_side="left")
        self.tok_score = AutoTokenizer.from_pretrained(self.name,
                                                       padding_side="right")
        for t in (self.tok_gen, self.tok_score):
            if t.pad_token is None:
                t.pad_token = t.eos_token

        kw = dict(device_map="auto" if self.device == "cuda" else None)
        if load_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["dtype"] = (torch.float16 if self.device == "cuda"
                           else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, **kw)
        self.model.eval()
        if self.device == "cpu":
            self.model.to(self.device)

    # ----------------------------------------------------------------- chat

    def _chat(self, tok, prompts):
        return [tok.apply_chat_template([{"role": "user", "content": p}],
                                        tokenize=False,
                                        add_generation_prompt=True)
                for p in prompts]

    # ------------------------------------------------------------ channel A

    def generate(self, prompts, max_new_tokens=160):
        chats = self._chat(self.tok_gen, prompts)
        enc = self.tok_gen(chats, return_tensors="pt", padding=True).to(
            self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok_gen.pad_token_id)
        return self.tok_gen.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True)

    def generate_multiturn(self, triples, max_new_tokens=40):
        """Replay a finished exchange, then ask a follow-up.

        triples: (user_turn_1, assistant_turn_1, user_turn_2).

        Experiment e5 needs the model to answer a question ABOUT an answer it
        already gave. Turn one is never regenerated: the assistant message is
        the text that experiment e1 actually collected, replayed verbatim. A
        regenerated first turn would be a different exchange, and the model
        would be reporting on something that never happened.
        """
        chats = [self.tok_gen.apply_chat_template(
            [{"role": "user", "content": u1},
             {"role": "assistant", "content": a1},
             {"role": "user", "content": u2}],
            tokenize=False, add_generation_prompt=True)
            for u1, a1, u2 in triples]
        enc = self.tok_gen(chats, return_tensors="pt", padding=True).to(
            self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok_gen.pad_token_id)
        return self.tok_gen.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True)

    # ----------------------------------------------------------- channel A2

    def _answer_sequence(self, prompt, answer):
        tok = self.tok_gen
        chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False,
                                       add_generation_prompt=True)
        p_ids = tok(chat, add_special_tokens=False).input_ids
        a_ids = tok(str(answer), add_special_tokens=False).input_ids
        return p_ids, a_ids

    def score_answers(self, prompts, answers):
        """log P(answer | chat prompt), summed over the answer's own tokens.

        Returns one (summed log prob, number of answer tokens) pair per prompt.

        Prompt and answer are tokenised SEPARATELY and joined as id lists, so a
        merge across the boundary cannot change which tokens are scored. The
        chat template already carries its special tokens, so nothing is added
        twice. Only the last few positions' logits are computed: the full
        vocabulary over a long prompt is most of the memory a forward pass
        needs.

        ONE SEQUENCE PER FORWARD PASS, NEVER A BATCH
        The first Colab run scored 16 at a time with left padding. The GPU
        self-check stopped it on all three models: batched scores differed from
        a plain forward pass by up to 0.56 nats (mistral), 0.15 (qwen) and 0.05
        (llama), while the same answers scored one at a time matched it to four
        decimals. The gap appeared on the longest prompt too, which carried no
        padding at all, so it is not padding or positions: a batched 4-bit fp16
        forward pass takes different kernels and lands on different numbers.
        Channel A2 reads paired differences of hundredths of a nat, and a score
        that depends on which other prompts shared its batch would bury them.
        Scored alone, a sequence gets the same number whatever else is run.
        """
        out = []
        for p, a in zip(prompts, answers):
            p_ids, a_ids = self._answer_sequence(p, a)
            n = len(a_ids)
            if n == 0:
                out.append((float("nan"), 0))
                continue
            ids = torch.tensor([p_ids + a_ids], dtype=torch.long,
                               device=self.model.device)
            keep = n + 1
            with torch.no_grad():
                try:
                    logits = self.model(input_ids=ids,
                                        logits_to_keep=keep).logits
                except TypeError:
                    logits = self.model(input_ids=ids).logits[:, -keep:, :]
            logprobs = torch.log_softmax(logits[0].float(), dim=-1)
            # kept logits cover the last n+1 positions, and position t predicts
            # token t+1, so the answer's n tokens are predicted by kept
            # positions 0 up to n-1
            vals = logprobs[:n].gather(
                1, ids[0, -n:].unsqueeze(-1)).squeeze(-1)
            out.append((float(vals.sum().item()), n))
        return out

    def check_answer_scoring(self, prompts, answers, tolerance=0.01):
        """Prove score_answers on this GPU, before any row is written.

        The indexing in score_answers cannot be exercised offline, because it
        needs a real model. So the first thing channel A2 does in Colab is score
        a few prompts two ways and demand they agree:

            scorer      score_answers, logits kept for the answer only
            reference   a plain forward pass over the whole sequence, full
                        logits, sliced by hand

        and then score the same prompts again in reverse order, which must give
        identical numbers: a score may not depend on what ran before it.

        The tolerance is 0.01, down from 0.05. One at a time the two agreed to
        four decimals on every model, and an indexing error moves a score by
        whole nats.
        """
        scored = self.score_answers(prompts, answers)
        again = list(reversed(self.score_answers(list(reversed(prompts)),
                                                 list(reversed(answers)))))
        worst = 0.0
        for (p, a), (s_sum, s_n), (r_sum, _r_n) in zip(zip(prompts, answers),
                                                       scored, again):
            p_ids, a_ids = self._answer_sequence(p, a)
            full = torch.tensor([p_ids + a_ids], dtype=torch.long,
                                device=self.model.device)
            with torch.no_grad():
                lg = self.model(input_ids=full).logits[0].float()
            lp = torch.log_softmax(lg, dim=-1)
            pl, n = len(p_ids), len(a_ids)
            tgt = full[0, pl:pl + n]
            ref = float(lp[pl - 1:pl + n - 1].gather(
                1, tgt.unsqueeze(-1)).squeeze(-1).sum().item())
            gap = max(abs(s_sum - ref), abs(r_sum - s_sum))
            worst = max(worst, gap)
            print(f"    answer {str(a)[:14]:14} tokens {n}  scorer {s_sum:9.4f}"
                  f"  rerun {r_sum:9.4f}  reference {ref:9.4f}")
            if s_n != n:
                raise RuntimeError("answer scoring counted a different number "
                                   "of answer tokens than the reference")
        if worst > tolerance:
            raise RuntimeError(
                f"answer scoring disagrees with the reference by {worst:.4f}, "
                f"more than {tolerance}. Nothing was written. Send this output.")
        print(f"    answer scoring agrees with the reference, largest gap "
              f"{worst:.4f}")

    # ------------------------------------------------------------ channel B

    def score_continuations(self, prompts, words):
        """Mean log probability of `word` continuing `prompt`, per pair.

        Averaged over the word's own tokens so a longer word is not penalised
        for being longer. This is Paper 2's method 1 scorer, unchanged, so the
        numbers stay comparable with it.

        These are raw completion prompts, NOT chat-templated: method 1 measured
        continuation of a sentence, and wrapping it in a chat turn measures
        something else.
        """
        fulls = [p + " " + w for p, w in zip(prompts, words)]
        p_lens = [len(self.tok_score(p).input_ids) for p in prompts]
        enc = self.tok_score(fulls, return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(self.model.device)
        mask = enc["attention_mask"].to(self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=mask).logits
        logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        picked = logprobs.gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        out = []
        for i, p_len in enumerate(p_lens):
            n_full = int(mask[i].sum().item())
            n_new = n_full - p_len
            if n_new <= 0:
                out.append(float("nan"))
                continue
            vals = picked[i, p_len - 1:n_full - 1]
            out.append(float(vals.mean().item()))
        return out

    # ------------------------------------------------------------ channel C

    def letter_logprobs(self, prompts, letters=("A", "B")):
        """log P(letter) at the next position, and the mass those letters hold.

        Returns one dict per prompt: {"A": logp, "B": logp, "mass": p_a + p_b}.

        Each letter must be a single token after the model's own leading-space
        convention, which is checked once at startup by check_letter_tokens().
        """
        chats = self._chat(self.tok_gen, prompts)
        enc = self.tok_gen(chats, return_tensors="pt", padding=True).to(
            self.model.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[:, -1, :].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        probs = logprobs.exp()

        ids = {L: self._letter_id(L) for L in letters}
        out = []
        for i in range(len(prompts)):
            row = {L: float(logprobs[i, ids[L]].item()) for L in letters}
            row["mass"] = float(sum(probs[i, ids[L]].item() for L in letters))
            out.append(row)
        return out

    def _letter_id(self, letter):
        for form in (letter, " " + letter):
            got = self.tok_gen(form, add_special_tokens=False).input_ids
            if len(got) == 1:
                return got[0]
        # fall back to the first token, and say so rather than pretend
        got = self.tok_gen(letter, add_special_tokens=False).input_ids
        print(f"  WARNING: '{letter}' is not a single token on {self.key}; "
              f"reading its first token only")
        return got[0]

    def check_letter_tokens(self, letters=("A", "B")):
        print("\n  answer-letter tokenisation")
        for L in letters:
            bare = self.tok_gen(L, add_special_tokens=False).input_ids
            sp = self.tok_gen(" " + L, add_special_tokens=False).input_ids
            print(f"    '{L}': bare {len(bare)} token(s), "
                  f"with leading space {len(sp)} token(s)")
        print()

    def check_trait_tokens(self, words):
        print("\n  trait word token counts, fewer is better")
        for w in words:
            n = len(self.tok_score(" " + w).input_ids)
            flag = "   <- multi-token, expect noise" if n > 2 else ""
            print(f"    {w:>14} {n}{flag}")
        print()
