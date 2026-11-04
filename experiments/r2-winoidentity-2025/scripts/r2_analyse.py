#!/usr/bin/env python3
"""
r2_analyse.py - the preregistered R2 decisions, from the scored CSVs. No GPU.

    python r2_analyse.py --prompts r2_prompts.csv --dir <folder with the
        r2_scores_stage1_<fmt>.csv, r2_scores_stage3_<fmt>.csv and
        r2_format_choice.json>

Written before any score existed. Reads the chosen format from
r2_format_choice.json and never looks at the other format's neutral rows (there
are none: stage 3 scores only the chosen format).

1. FIDELITY. Our disparity per attribute and type beside the authors' Table 2.

2. THE FLOOR. Each demographic attribute's disparity (max minus min of mean CC
   over gender x marker, their Equation 2) against the neutral sets with the same
   number of markers:

       2 markers   N2a, N2b, N2c       3 markers   N3       4 markers   N4

   "Above the floor" if it exceeds every neutral set of that size, otherwise
   "not distinguishable from the floor". Counted over the 20 attribute-by-type
   cells. Bootstrap intervals over sentences are reported beside each value and
   do not change the verdict.

3. DOUBLY DISADVANTAGED. For each marker, D is the mean over sentence pairs of
   CC(feminine) minus CC(masculine), on sentences whose referent occupation is
   male-dominated, where a pair is the same base sentence with the pronoun
   swapped. More negative D means the feminine version loses more. For an
   attribute, its strongest feminine disadvantage is the most negative D over
   its markers; likewise for each neutral set. The pattern is attributed to the
   identity only if the attribute's is more negative than every neutral set of
   the same size. Otherwise it is reported as gender bias in the base sentences
   carried through any marker.
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_build_prompts as B    # noqa: E402
import r2_score as S            # noqa: E402

BOOT = 1000
SEED = 20260912
PRONOUN = re.compile(r"\b(he|she|him|her|his)\b", re.I)


def marker_sets():
    sets = {name: list(ms) for name, ms in B.NEUTRAL_SETS.items()}
    return sets


def cc(score):
    return math.exp(score[0]) - math.exp(score[1])


def disparity(rows, scores, markers, task_type):
    means = defaultdict(list)
    for r in rows:
        if r["task_type"] != task_type or r["marker"] not in markers:
            continue
        s = scores.get(r["row_id"])
        if s is not None:
            means[(r["gender"], r["marker"])].append(cc(s))
    vals = [sum(v) / len(v) for v in means.values() if v]
    return (max(vals) - min(vals)) if len(vals) >= 2 else float("nan")


def boot_disparity(rows, scores, markers, task_type, reps=BOOT, seed=SEED):
    by_sid = defaultdict(list)
    for r in rows:
        if r["task_type"] == task_type and r["marker"] in markers:
            by_sid[r["sentence_id"]].append(r)
    sids = sorted(by_sid)
    rng = random.Random(seed)
    stats = []
    for _ in range(reps):
        pick = [by_sid[rng.choice(sids)] for _ in sids]
        flat = [r for group in pick for r in group]
        stats.append(disparity(flat, scores, markers, task_type))
    stats = sorted(x for x in stats if x == x)
    if not stats:
        return float("nan"), float("nan")
    return stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1]


def pair_key(none_sentence):
    return PRONOUN.sub("PRON", none_sentence)


def feminine_disadvantage(rows, scores, marker, none_by_sid):
    """Mean over pairs of CC(fem) - CC(masc), male-dominated referents only."""
    pairs = defaultdict(dict)
    for r in rows:
        if r["marker"] != marker or r["referent_dominance"] != "male":
            continue
        s = scores.get(r["row_id"])
        if s is None:
            continue
        key = (pair_key(none_by_sid[r["sentence_id"]]), r["task_type"])
        pairs[key].setdefault(r["gender"], []).append(cc(s))
    diffs = [sum(p["fem"]) / len(p["fem"]) - sum(p["masc"]) / len(p["masc"])
             for p in pairs.values() if "fem" in p and "masc" in p]
    return (sum(diffs) / len(diffs)) if diffs else float("nan"), len(diffs)


def floor_verdicts(table_demo, table_neutral):
    """table_demo[attr] = (t1, t2); table_neutral[set] = (t1, t2)."""
    out = []
    for attr, markers in B.ATTRIBUTES.items():
        size = len(markers)
        matched = [s for s, ms in B.NEUTRAL_SETS.items() if len(ms) == size]
        for t_i in (0, 1):
            demo = table_demo[attr][t_i]
            neutral_vals = [table_neutral[s][t_i] for s in matched]
            top = max(neutral_vals)
            out.append({"attribute": attr, "type": ("type1", "type2")[t_i],
                        "size": size, "demographic": demo,
                        "neutral_sets": dict(zip(matched, neutral_vals)),
                        "neutral_max": top,
                        "verdict": "above floor" if demo > top
                        else "not distinguishable"})
    return out


def dd_verdicts(fd):
    """fd[marker] = D. Per attribute: most negative D against matched sets."""
    out = []
    for attr, markers in B.ATTRIBUTES.items():
        size = len(markers)
        matched = [s for s, ms in B.NEUTRAL_SETS.items() if len(ms) == size]
        worst_attr = min(fd[m] for m in markers)
        worst_sets = {s: min(fd[m] for m in B.NEUTRAL_SETS[s]) for s in matched}
        attributed = all(worst_attr < v for v in worst_sets.values())
        out.append({"attribute": attr, "size": size, "worst_D": worst_attr,
                    "neutral_worst_D": worst_sets,
                    "verdict": "attributed to identity" if attributed
                    else "gender bias carried through any marker"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--boot", type=int, default=BOOT)
    args = ap.parse_args()

    with open(os.path.join(args.dir, "r2_format_choice.json")) as f:
        choice = json.load(f)
    fmt = choice["chosen_format"]
    rows = S.load_prompts(args.prompts)
    s1 = S.load_scores(os.path.join(args.dir, f"r2_scores_stage1_{fmt}.csv"))
    s3 = S.load_scores(os.path.join(args.dir, f"r2_scores_stage3_{fmt}.csv"))
    scores = dict(s1)
    scores.update(s3)
    none_by_sid = {r["sentence_id"]: r["sentence"] for r in rows
                   if r["kind"] == "none"}

    print("=" * 78)
    print(f"R2 WINOIDENTITY: floor test on mistral-7B-instruct-v0.2, format {fmt}")
    print("=" * 78)
    fid = choice["fidelity"][fmt]
    print(f"\n1. FIDELITY  spearman {fid['spearman']:.3f}  mean abs diff "
          f"{fid['mad']:.3f}  {'PASS' if fid['pass'] else 'FAIL'}"
          + ("" if choice["passed"] else
             "  -> instrument not shown equivalent; conclusions withheld"))

    demo_rows = [r for r in rows if r["kind"] in ("demographic", "none")]
    table_demo = {}
    for attr, markers in B.ATTRIBUTES.items():
        table_demo[attr] = tuple(disparity(demo_rows, scores, set(markers), t)
                                 for t in ("type1", "type2"))
    table_none = tuple(disparity(demo_rows, scores, {"none"}, t)
                       for t in ("type1", "type2"))
    neutral_rows = [r for r in rows if r["kind"] in ("neutral", "none")]
    table_neutral = {}
    for name, markers in B.NEUTRAL_SETS.items():
        table_neutral[name] = tuple(disparity(neutral_rows, scores, set(markers), t)
                                    for t in ("type1", "type2"))
    base_rows = [r for r in rows if r["kind"] == "their_base"]
    their_markers = B.THEIR_BASELINE["their_nondemographic"]
    their_all = tuple(disparity(base_rows, scores, set(their_markers), t)
                      for t in ("type1", "type2"))
    their_avg = tuple(
        sum(disparity(base_rows, scores, {m}, t) for m in their_markers)
        / len(their_markers) for t in ("type1", "type2"))

    print(f"\n   {'attribute':20}{'ours t1':>9}{'ours t2':>9}"
          f"{'theirs t1':>11}{'theirs t2':>11}")
    print(f"   {'none':20}{table_none[0]:9.3f}{table_none[1]:9.3f}"
          f"{S.PUBLISHED['none'][0]:11.3f}{S.PUBLISHED['none'][1]:11.3f}")
    for attr in B.ATTRIBUTES:
        o, p = table_demo[attr], S.PUBLISHED[attr]
        print(f"   {attr:20}{o[0]:9.3f}{o[1]:9.3f}{p[0]:11.3f}{p[1]:11.3f}")
    print(f"   {'their non-demographic, averaged per marker (their method)':s}")
    print(f"   {'':20}{their_avg[0]:9.3f}{their_avg[1]:9.3f}"
          f"{S.PUBLISHED_NONDEMOGRAPHIC[0]:11.3f}"
          f"{S.PUBLISHED_NONDEMOGRAPHIC[1]:11.3f}")
    print(f"   {'  same, max-min over all 6 subgroups':s}")
    print(f"   {'':20}{their_all[0]:9.3f}{their_all[1]:9.3f}")

    print("\n2. THE FLOOR: each attribute against neutral sets of the same size")
    print(f"   {'attribute':20}{'type':7}{'size':>5}{'demographic':>13}"
          f"{'neutral max':>13}   verdict")
    verdicts = floor_verdicts(table_demo, table_neutral)
    above = 0
    for v in verdicts:
        above += v["verdict"] == "above floor"
        sets = ", ".join(f"{k} {x:.3f}" for k, x in v["neutral_sets"].items())
        print(f"   {v['attribute']:20}{v['type']:7}{v['size']:5}"
              f"{v['demographic']:13.3f}{v['neutral_max']:13.3f}   "
              f"{v['verdict']}   [{sets}]")
    print(f"\n   {above} of {len(verdicts)} attribute-by-type cells are above "
          f"the floor; {len(verdicts) - above} are not distinguishable from it")

    if args.boot:
        print("\n   bootstrap 95 percent intervals over sentences (not deciding)")
        for name, markers in list(B.ATTRIBUTES.items()) + \
                list(B.NEUTRAL_SETS.items()):
            src = demo_rows if name in B.ATTRIBUTES else neutral_rows
            for t in ("type1", "type2"):
                lo, hi = boot_disparity(src, scores, set(markers), t,
                                        reps=args.boot)
                print(f"   {name:20}{t:7}[{lo:.3f}, {hi:.3f}]")

    print("\n3. DOUBLY DISADVANTAGED: feminine minus masculine CC on "
          "male-dominated referents")
    all_markers = sorted({m for ms in B.ATTRIBUTES.values() for m in ms}
                         | {m for ms in B.NEUTRAL_SETS.values() for m in ms}
                         | {"none"})
    fd = {}
    for m in all_markers:
        d, n = feminine_disadvantage(rows, scores, m, none_by_sid)
        fd[m] = d
        print(f"   {m:22} D {d:+.3f}  pairs {n}")
    print()
    for v in dd_verdicts(fd):
        sets = ", ".join(f"{k} {x:+.3f}" for k, x in v["neutral_worst_D"].items())
        print(f"   {v['attribute']:20} worst D {v['worst_D']:+.3f}  "
              f"neutral [{sets}]  {v['verdict']}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
