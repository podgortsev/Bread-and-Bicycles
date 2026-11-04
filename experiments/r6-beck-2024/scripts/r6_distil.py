#!/usr/bin/env python3
"""
r6_distil.py - reduce the raw collection to what the analysis needs, so that it
can live in this repository.

    python r6_distil.py --out <dir of raw jsonl> --to <dir in this repo>

The raw files are about 1.45 GB for the full grid, nearly all of it the prompt
text, which is reconstructible from the stimuli and their template. What
`r6_step0.py` reads is two columns: the instance id and the predicted label. So
each run is distilled to those two columns plus the run's identity, and the
whole grid becomes a few megabytes that travel with the paper.

Written after the first collection was lost on 2026-10-15 because its outputs
lived only in a temporary directory. The point of this script is that the result
stays checkable from the repository alone.

Output: one gzipped CSV, `r6_predictions.csv.gz`, with columns

    task, model, format, condition, unit_id, response

and a plain-text manifest giving the row count and a SHA-256 of every raw file
it was built from, so a later rerun can show it read the same collection.
"""

import argparse
import csv
import gzip
import hashlib
import os
import re
import sys

import pandas as pd

# The condition is matched against the known set rather than by a wildcard.
# A non-greedy wildcard stops at the first underscore, so "political_affiliation"
# was split into a condition "political" and a model "affiliation_allenai-...".
CONDITIONS = ["gender", "race", "education", "age", "political_affiliation", "ratings"]

NAME = re.compile(
    r"^(?P<task>.+?)_filtered_sample_n1000_seed24_"
    r"(?:with_sociodemographics_(?P<cond>" + "|".join(CONDITIONS) + r")"
    r"|without_sociodemographics)_"
    r"(?P<model>.+?)_format(?P<fmt>\d+)\.jsonl$"
)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory holding the raw jsonl")
    ap.add_argument("--to", required=True, help="directory in this repository")
    args = ap.parse_args()
    os.makedirs(args.to, exist_ok=True)

    files = sorted(f for f in os.listdir(args.out) if f.endswith(".jsonl"))
    if not files:
        sys.exit(f"no jsonl in {args.out}")

    rows = 0
    manifest = []
    dest = os.path.join(args.to, "r6_predictions.csv.gz")
    with gzip.open(dest, "wt", newline="", encoding="utf-8") as gz:
        w = csv.writer(gz)
        w.writerow(["task", "model", "format", "condition", "unit_id", "response"])
        for name in files:
            m = NAME.match(name)
            if not m:
                print(f"  skipped, name not understood: {name}", flush=True)
                continue
            path = os.path.join(args.out, name)
            d = pd.read_json(path, lines=True)
            cond = m.group("cond") or "none"
            for uid, resp in zip(d["unit_id"], d["response"]):
                w.writerow([m.group("task"), m.group("model"), m.group("fmt"),
                            cond, uid, resp])
            rows += len(d)
            manifest.append((name, len(d), sha256(path)))
            print(f"  {name}: {len(d)} rows", flush=True)

    with open(os.path.join(args.to, "r6_manifest.txt"), "w", encoding="utf-8") as f:
        f.write("raw file, rows, sha256 of the raw file\n")
        for name, n, h in manifest:
            f.write(f"{name}, {n}, {h}\n")

    size = os.path.getsize(dest)
    print("=" * 78)
    print(f"{len(manifest)} runs, {rows} predictions -> {dest} ({size/1e6:.1f} MB)")
    print(f"manifest: {os.path.join(args.to, 'r6_manifest.txt')}")
    print("=" * 78)


if __name__ == "__main__":
    main()
