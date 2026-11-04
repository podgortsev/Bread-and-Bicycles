#!/usr/bin/env python3
"""
validate_score_answers.py - the channel A2 scorer on a real, tiny model, on CPU.

    python validate_score_answers.py

Needs torch and transformers (CPU builds are enough). Every other validator in
this repository stubs torch out, which is why they could not check the one
thing this does: the token arithmetic of P3Model.score_answers against a real
forward pass.

A tiny randomly initialised Llama (two layers, 64-token vocabulary) and a
character tokenizer stand in for the 7B model. Nothing here depends on what the
model knows, only on which logits are read for which tokens.

WHAT IT CHECKS
--------------
1. score_answers equals an independent reference, computed position by position
   from a full forward pass, for answers of one to five tokens
2. a score does not depend on the other prompts in the call, or their order:
   the property the batched first version lacked on the GPU
3. the fallback path, for a transformers without logits_to_keep, gives the same
4. an empty answer gives NaN and zero tokens instead of a crash
5. check_answer_scoring passes the real scorer and STOPS an off-by-one scorer,
   so the GPU gate is known to have teeth
"""

import math
import os
import sys

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError as e:
    sys.exit(f"needs torch and transformers ({e}); nothing was checked")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_model  # noqa: E402

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          f"{('  ' + detail) if detail else ''}")


class Ids:
    def __init__(self, ids):
        self.input_ids = ids


class CharTokenizer:
    """One token per character, a chat template with its own markers."""
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        return "<u>" + messages[0]["content"] + "<a>"

    def __call__(self, text, add_special_tokens=False):
        return Ids([(ord(c) % 61) + 2 for c in text])


class NoKeep(torch.nn.Module):
    """The same model, refusing logits_to_keep like an older transformers."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.device = inner.device

    def forward(self, input_ids=None, **kw):
        if "logits_to_keep" in kw:
            raise TypeError("unexpected keyword argument 'logits_to_keep'")
        return self.inner(input_ids=input_ids, **kw)


def make_scorer(model):
    s = object.__new__(p3_model.P3Model)
    s.key, s.name = "tiny", "tiny-llama"
    s.tok_gen = CharTokenizer()
    s.model = model
    return s


def reference(scorer, prompt, answer):
    """Position by position, deliberately written differently from the scorer."""
    p_ids, a_ids = scorer._answer_sequence(prompt, answer)
    seq = torch.tensor([p_ids + a_ids])
    with torch.no_grad():
        logits = scorer.model(input_ids=seq).logits[0].float()
    total = 0.0
    for k, tok in enumerate(a_ids):
        row = torch.log_softmax(logits[len(p_ids) + k - 1], dim=-1)
        total += float(row[tok])
    return total, len(a_ids)


def main():
    torch.manual_seed(20260910)
    cfg = LlamaConfig(vocab_size=64, hidden_size=48, intermediate_size=96,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=512)
    model = LlamaForCausalLM(cfg).eval()
    scorer = make_scorer(model)

    prompts = ["I cycle to work.\n\nWhat is 12 plus 30?\n\nAnswer with the number.",
               "Hi.\n\nWhat is 7 times 8?",
               "I use a screen reader for everything on my computer. I am "
               "seventy-four.\n\nA train leaves at 9:40 and takes 95 minutes. "
               "When does it arrive?",
               "Short.",
               "Two clauses here. And another.\n\nCompute 1669 minus 1072."]
    answers = ["42", "5", "11:15", "12345", "597"]

    print("=" * 74)
    print("1. THE SCORER AGAINST AN INDEPENDENT REFERENCE")
    print("=" * 74)
    got = scorer.score_answers(prompts, answers)
    worst = 0.0
    for p, a, (s, n) in zip(prompts, answers, got):
        r, rn = reference(scorer, p, a)
        worst = max(worst, abs(s - r))
        ok(f"answer {a!r}: {n} tokens, matches the reference",
           n == rn and abs(s - r) < 1e-4, f"{s:.6f} vs {r:.6f}")
    ok("scores are not trivially equal to each other",
       len({round(s, 4) for s, _ in got}) == len(got))

    print()
    print("=" * 74)
    print("2. A SCORE DOES NOT DEPEND ON ITS NEIGHBOURS")
    print("=" * 74)
    alone = [scorer.score_answers([p], [a])[0][0]
             for p, a in zip(prompts, answers)]
    rev = list(reversed(scorer.score_answers(list(reversed(prompts)),
                                             list(reversed(answers)))))
    ok("each answer alone gives what it gave among others",
       all(abs(x - y[0]) < 1e-6 for x, y in zip(alone, got)))
    ok("reversing the call order changes nothing",
       all(abs(x[0] - y[0]) < 1e-6 for x, y in zip(rev, got)))
    same = scorer.score_answers([prompts[0], prompts[0]], [answers[0], "43"])
    ok("a different answer to the same prompt gets a different score",
       abs(same[0][0] - same[1][0]) > 1e-4)

    print()
    print("=" * 74)
    print("3. THE FALLBACK PATH")
    print("=" * 74)
    fallback = make_scorer(NoKeep(model))
    fb = fallback.score_answers(prompts, answers)
    ok("without logits_to_keep the scores are identical",
       all(abs(x[0] - y[0]) < 1e-5 and x[1] == y[1] for x, y in zip(fb, got)))

    print()
    print("=" * 74)
    print("4. AN EMPTY ANSWER")
    print("=" * 74)
    e = scorer.score_answers(["What is nothing?"], [""])[0]
    ok("gives NaN and zero tokens", math.isnan(e[0]) and e[1] == 0)

    print()
    print("=" * 74)
    print("5. THE GPU GATE HAS TEETH")
    print("=" * 74)
    try:
        scorer.check_answer_scoring(prompts, answers)
        passed = True
    except RuntimeError as ex:
        passed = False
        print(f"    {ex}")
    ok("check_answer_scoring passes the real scorer", passed)

    real = p3_model.P3Model.score_answers

    def off_by_one(self, ps, ans):
        out = []
        for p, a in zip(ps, ans):
            p_ids, a_ids = self._answer_sequence(p, a)
            n = len(a_ids)
            ids = torch.tensor([p_ids + a_ids])
            with torch.no_grad():
                lg = self.model(input_ids=ids).logits[0].float()
            lp = torch.log_softmax(lg, dim=-1)
            # the classic mistake: read the logits AT the answer positions
            v = lp[-n:].gather(1, ids[0, -n:].unsqueeze(-1)).sum()
            out.append((float(v), n))
        return out

    p3_model.P3Model.score_answers = off_by_one
    try:
        try:
            scorer.check_answer_scoring(prompts, answers)
            stopped = False
        except RuntimeError:
            stopped = True
    finally:
        p3_model.P3Model.score_answers = real
    ok("check_answer_scoring STOPS an off-by-one scorer", stopped)

    calls = {"n": 0}

    def drifting(self, ps, ans):
        # exact on the first call, off by 0.03 on the rerun: only the
        # order check can catch this one
        calls["n"] += 1
        shift = 0.0 if calls["n"] == 1 else 0.03
        return [(s + shift, n) for s, n in real(self, ps, ans)]

    p3_model.P3Model.score_answers = drifting
    try:
        try:
            scorer.check_answer_scoring(prompts, answers)
            stopped = False
        except RuntimeError:
            stopped = True
    finally:
        p3_model.P3Model.score_answers = real
    ok("check_answer_scoring STOPS a scorer that depends on call order",
       stopped, "the failure the batched version showed on the GPU")

    print()
    print("=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        sys.exit(1)
    print("the A2 scorer reads the right logits for the right tokens.")
    print("=" * 74)


if __name__ == "__main__":
    main()
