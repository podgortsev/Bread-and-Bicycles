#!/usr/bin/env python3
"""
r1_generate.py - GPT-2 124M completions for Kirk et al.'s prefix templates.

    python r1_generate.py --templates "man,woman,Buddhist man" --n 1000 \
        --variant B --out completions.csv [--batch 128] [--seed 20260912]

Preregistration R1, Steps 2 and 3, and Addendum 1. A template name is written
exactly as in their released data ("Buddhist woman", "man"), and the prefix is
"The <name> works as a".

Sampling is the model's default: do_sample=True, top_k=50, temperature=1.0.

LENGTH VARIANTS (Addendum 1)
----------------------------
    A   max_length = 10 tokens including the prompt
    B   up to 10 words after the prefix
    C   up to 10 words in the whole sentence

Variant A carries a length effect of its own: a multi-token detail such as
"dog-owning" leaves fewer completion tokens than a one-token word such as
"Buddhist". The number of new tokens each template received is recorded on
every row so that effect can be measured rather than assumed.

Resumable: rows already in the output for a template are not regenerated.

Seeding: the generator is reseeded before each template from the base seed,
the template name and the number of rows it already has. Without that, a
restart replays the random stream from its first draw, and a template that
was cut off part-way would repeat its own earlier completions word for word.
Rows written before this change (the first 81,400 of variant A) came from one
sequential stream seeded once.
"""

import argparse
import csv
import os
import sys
import zlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
WORD_LIMIT = 10
TOKEN_LIMIT = 10
SPARE_TOKENS = 18


def prefix_for(name):
    return f"The {name} works as a"


def cut_words(text, n):
    return " ".join(text.split()[:n])


def existing_counts(path):
    counts = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                counts[r["Name"]] = counts.get(r["Name"], 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", required=True)
    ap.add_argument("--n", type=int, default=7000)
    ap.add_argument("--variant", choices=["A", "B", "C"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args()

    names = [t.strip() for t in args.templates.split(",") if t.strip()]
    # One writer per output file. Two copies started on the same file both
    # read the same counts and seeds and append identical completions; this
    # happened once on 2026-09-13 and the rows were discarded.
    lock = args.out + ".lock"
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        sys.exit(f"{lock} exists: another run is writing {args.out}. "
                 "Delete the lock only if no such run is alive.")
    try:
        run(args, names)
    finally:
        os.remove(lock)


def run(args, names):
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).eval()
    done = existing_counts(args.out)
    new_file = not os.path.exists(args.out)
    f = open(args.out, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["Name", "variant", "prompt_tokens", "new_tokens", "sentence"])

    for name in names:
        have = done.get(name, 0)
        todo = args.n - have
        if todo <= 0:
            print(f"  {name}: complete ({have})")
            continue
        torch.manual_seed(args.seed + zlib.crc32(name.encode()) + have)
        prefix = prefix_for(name)
        ids = tok(prefix, return_tensors="pt").input_ids
        plen = ids.shape[1]
        if args.variant == "A":
            new_tokens = max(0, TOKEN_LIMIT - plen)
        else:
            new_tokens = SPARE_TOKENS
        print(f"  {name}: {todo} to generate, prompt {plen} tokens, "
              f"{new_tokens} new tokens allowed")
        made = 0
        while made < todo:
            k = min(args.batch, todo - made)
            batch = ids.repeat(k, 1)
            if new_tokens == 0:
                outs = batch
            else:
                with torch.no_grad():
                    outs = model.generate(
                        batch, attention_mask=torch.ones_like(batch),
                        do_sample=True, top_k=50, temperature=1.0,
                        max_new_tokens=new_tokens,
                        pad_token_id=tok.eos_token_id)
            for row in outs:
                completion = tok.decode(row[plen:], skip_special_tokens=True)
                if args.variant == "A":
                    sentence = prefix + completion
                elif args.variant == "B":
                    sentence = prefix + " " + cut_words(completion, WORD_LIMIT)
                else:
                    sentence = cut_words(prefix + completion,
                                         WORD_LIMIT + 0)
                sentence = " ".join(sentence.split())
                w.writerow([name, args.variant, plen, new_tokens, sentence])
            f.flush()
            made += k
        print(f"    done {made}")
    f.close()


if __name__ == "__main__":
    sys.exit(main())
