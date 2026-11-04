#!/usr/bin/env python3
"""
r6_step0.py - Addendum 11: share of label changes per condition against the
no-profile prediction, per model, and the comparison with their Table 8 (Diaz).

    python r6_step0.py --out <dir of their sd_prompting outputs> [--formats 0]
"""

import argparse
import os

import pandas as pd

INPUT = "sentiment_diaz_filtered_sample_n1000_seed24"
TABLE8_DIAZ = {  # model: (most influential attribute, percent of label changes)
    "google/flan-t5-small": ("All", 2.30),
    "google/flan-t5-base": ("All", 28.60),
    "google/flan-t5-large": ("All", 38.20),
    "allenai/tk-instruct-small-def-pos": ("All", 0.00),
    "allenai/tk-instruct-base-def-pos": ("All", 23.30),
    "allenai/tk-instruct-large-def-pos": ("All", 7.00),
}
ATTRS = {"gender": "G", "race": "R", "education": "E", "age": "A",
         "political_affiliation": "PA", "ratings": "All"}


# Either source gives the same two columns. The raw collection is about 1.45 GB
# and lives outside this repository; the distilled file is under a megabyte and
# travels with the paper, so the verdict below can be recomputed from the
# repository alone. r6_distil.py builds the second from the first.
PREDICTIONS = None


def use_predictions(path):
    """Load the distilled predictions once and serve slices of them."""
    global PREDICTIONS
    PREDICTIONS = pd.read_csv(path, dtype={"format": str})


def load(out, model, fmt, att):
    m = model.replace("/", "-")
    if PREDICTIONS is not None:
        cond = "none" if att is None else att
        d = PREDICTIONS[(PREDICTIONS["model"] == m)
                        & (PREDICTIONS["format"] == str(fmt))
                        & (PREDICTIONS["condition"] == cond)]
        return d[["unit_id", "response"]].reset_index(drop=True) if len(d) else None
    name = (f"{INPUT}_without_sociodemographics_{m}_format{fmt}.jsonl" if att is None
            else f"{INPUT}_with_sociodemographics_{att}_{m}_format{fmt}.jsonl")
    path = os.path.join(out, name)
    return pd.read_json(path, lines=True) if os.path.exists(path) else None


def shares(out, model, fmt):
    base = load(out, model, fmt, None)
    if base is None:
        return None
    ref = dict(zip(base["unit_id"], base["response"]))
    res = {}
    for att, short in ATTRS.items():
        d = load(out, model, fmt, att)
        if d is None:
            res[short] = None
            continue
        changed = (d["response"] != d["unit_id"].map(ref)).mean() * 100
        res[short] = (float(changed), len(d))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="directory of raw sd_prompting jsonl")
    ap.add_argument("--predictions", help="distilled r6_predictions.csv.gz instead of --out")
    ap.add_argument("--formats", default="0")
    args = ap.parse_args()
    if not args.out and not args.predictions:
        ap.error("give either --out or --predictions")
    if args.predictions:
        use_predictions(args.predictions)
    fmts = args.formats.split(",")
    print("=" * 78)
    print(f"R6 BECK STEP 0 (Addendum 11): Diaz, label changes against no profile, formats {fmts}")
    print("=" * 78)
    passed = 0
    complete = 0
    for model, (their_att, their_pct) in TABLE8_DIAZ.items():
        per_fmt = [shares(args.out, model, f) for f in fmts]
        if any(p is None or any(v is None for v in p.values()) for p in per_fmt):
            print(f"  {model}: incomplete")
            continue
        complete += 1
        avg = {k: sum(p[k][0] for p in per_fmt) / len(per_fmt) for k in ATTRS.values()}
        top = max(avg, key=avg.get)
        ok = abs(avg["All"] - their_pct) <= 5 and top == "All"
        passed += ok
        cells = "  ".join(f"{k} {v:5.1f}" for k, v in avg.items())
        print(f"  {model:34} {cells}")
        print(f"  {'':34} ours: top {top}, All {avg['All']:.2f}   theirs: top {their_att}, "
              f"{their_pct:.2f}   {'match' if ok else 'no match'}")
    print(f"\n  models matching: {passed} of {complete} complete (pass: at least 4 of 6)")
    if complete == len(TABLE8_DIAZ):
        print(f"  VERDICT: {'PASS' if passed >= 4 else 'FAIL on this reading'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
