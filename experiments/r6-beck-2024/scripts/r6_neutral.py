#!/usr/bin/env python3
"""
r6_neutral.py - the R6 floor: their Diaz pipeline with five socially neutral
attributes in the slots of their five sociodemographic ones.

    python r6_neutral.py --repo <their clone> --data <sample jsonl> --out <dir>
                         --cache <hf cache> --model google/flan-t5-small [--format 0]

Their prompt template, their attribute-value logic, their likelihood scoring
(imported from their `sd_prompting.py`), their answer options. Only the five
attribute names and their values are replaced:

    gender (2 values)                -> drink (2)
    race (6)                         -> colour (6)
    age (7)                          -> day (7)
    education level (8)              -> favourite fruit (8)
    political affiliation (3)        -> preferred pet (3)

The word counts of the labels match theirs slot by slot, and the number of
values matches, so the two arms differ in what the words mean, not in how many
there are.

"All attributes" in their run uses each annotator's own profile. The neutral
"all" arm relabels those same profiles value by value through a fixed bijection
per attribute, so the joint structure of the profiles is kept exactly.

Writes one jsonl per condition, in the shape their own script writes.
"""

import argparse
import importlib.util
import json
import os
import sys

import pandas as pd
import torch

NEUTRAL = {
    "drink": ["coffee", "tea"],
    "colour": ["blue", "brown", "green", "grey", "red", "yellow"],
    "day": ["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"],
    "favourite fruit": ["apple", "cherry", "grape", "lemon", "mango", "peach", "pear", "plum"],
    "preferred pet": ["cat", "dog", "fish"],
}
SOCIAL_TO_NEUTRAL = {  # their attribute -> ours, in their slot order
    "gender": "drink",
    "race": "colour",
    "age_range": "day",
    "education": "favourite fruit",
    "political_affiliation": "preferred pet",
}
SLOT_ORDER = ["drink", "colour", "day", "favourite fruit", "preferred pet"]


def load_theirs(repo):
    sys.path.insert(0, repo)
    spec = importlib.util.spec_from_file_location("sd_prompting", os.path.join(repo, "sd_prompting.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import util
    return mod, util


def bijections(rows):
    """Fixed value-by-value maps from their attribute values to the neutral ones."""
    maps = {}
    for social, neutral in SOCIAL_TO_NEUTRAL.items():
        seen = sorted({str(r[social]) for row in rows for r in row["ratings"] if social in r})
        values = NEUTRAL[neutral]
        if len(seen) > len(values):
            sys.exit(f"{social}: {len(seen)} values in the data, {len(values)} neutral ones")
        maps[social] = {v: values[i] for i, v in enumerate(seen)}
    return maps


def sd_info(pairs):
    """Their wording: person of a 'x', b 'y' ... and e 'z'."""
    parts = [f"{name} '{value}'" for name, value in pairs]
    return "person of " + ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--format", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    theirs, util = load_theirs(args.repo)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    maps = bijections(rows)
    template = util.dataset2prompt["sentiment_diaz"]["with_sd"][args.format]
    choices = util.choices_per_dataset["sentiment_diaz"]

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, cache_dir=args.cache,
                                                  local_files_only=True).eval().to(device)

    conditions = {name: [(name, v) for v in NEUTRAL[name]] for name in SLOT_ORDER}
    for name, value_pairs in conditions.items():
        tag = name.replace(" ", "_")
        path = os.path.join(args.out, f"neutral_{tag}_{args.model.replace('/', '-')}_format{args.format}.jsonl")
        if os.path.exists(path):
            print(f"exists: {path}", flush=True)
            continue
        data, prompts = [], []
        for row in rows:
            for pair in value_pairs:
                prompt = template.format(sd_info=sd_info([pair])).format(text=row["unit_text"])
                data.append({"unit_id": row["unit_id"], "condition": tag, "value": pair[1]})
                prompts.append(prompt)
        responses = theirs.ppl_generate(prompts, model, tokenizer, choices, args.batch_size, device)
        pd.DataFrame([{**d, "prompt": p, "response": r} for d, p, r in zip(data, prompts, responses)]
                     ).to_json(path, lines=True, orient="records")
        print(f"{tag}: {len(prompts)} prompts -> {path}", flush=True)

    path = os.path.join(args.out, f"neutral_all_{args.model.replace('/', '-')}_format{args.format}.jsonl")
    if not os.path.exists(path):
        data, prompts = [], []
        for row in rows:
            for rating in row["ratings"]:
                pairs = [(SOCIAL_TO_NEUTRAL[s], maps[s][str(rating[s])]) for s in SOCIAL_TO_NEUTRAL]
                prompt = template.format(sd_info=sd_info(pairs)).format(text=row["unit_text"])
                data.append({"unit_id": row["unit_id"], "condition": "all",
                             "value": "|".join(v for _, v in pairs)})
                prompts.append(prompt)
        responses = theirs.ppl_generate(prompts, model, tokenizer, choices, args.batch_size, device)
        pd.DataFrame([{**d, "prompt": p, "response": r} for d, p, r in zip(data, prompts, responses)]
                     ).to_json(path, lines=True, orient="records")
        print(f"all: {len(prompts)} prompts -> {path}", flush=True)
    else:
        print(f"exists: {path}", flush=True)


if __name__ == "__main__":
    main()
