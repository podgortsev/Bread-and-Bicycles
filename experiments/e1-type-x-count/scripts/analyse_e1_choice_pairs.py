#!/usr/bin/env python3
"""
analyse_e1_choice_pairs.py - the valid interaction on the choice channel.

    python analyse_e1_choice_pairs.py e1_choice_pairs_qwen.csv e1_choice_qwen.csv

Give it BOTH files per model. The interaction needs one term from each run:

    interaction = margin(both vs A)  -  margin(B vs NONE)
                  from the pairs run    from the first choice run

and the mirror image,

    interaction = margin(both vs B)  -  margin(A vs NONE)

Both are one-clause changes read the same way, so the saturation that invalidated
the old contrast applies to both terms and cancels. The two versions must agree;
where they do not, the number belongs to the slot and is reported as such.

WHY THE OLD CONTRAST WAS WRONG, IN ONE LINE
-------------------------------------------
It differenced margins against a common reference, and the both-cell differs
from that reference in two clauses while each single differs in one. Those are
not comparable quantities. On qwen the two neutral details gave a floor of
+3.56 that was pure arithmetic: -2.64 minus -4.39 minus -1.82.

THE FLOOR STILL RUNS FIRST
--------------------------
The same neutral pair goes through the same new contrast. If it moves like the
social contrasts do, the rebuilt instrument is no better than the old one and
the channel is reported as failed rather than fixed.
"""

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared")))

import p3_stats as S      # noqa: E402

MASS_GATE = 0.5


def _conditions_path(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (name, os.path.join(here, name),
                 os.path.join(here, "..", "..", "shared", "conditions", name)):
        if os.path.exists(cand):
            return cand
    raise SystemExit(f"{name} not found")


def load(paths):
    """pairs[model][(contrast, side)][item] and base[model][cond][item]."""
    pairs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    base = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    gate = defaultdict(lambda: [0, 0])
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            fields = rd.fieldnames or []
            for r in rd:
                mk = r["model_key"]
                item = f"{r['profile']}:{r['question']}"
                gate[mk][1] += 1
                if float(r["mass"]) < MASS_GATE:
                    gate[mk][0] += 1
                    continue
                if "contrast" in fields:
                    pairs[mk][(r["contrast"], r["side"])][item].append(
                        float(r["margin_left"]))
                elif "margin_signal" in fields:
                    base[mk][r["condition"]][item].append(
                        float(r["margin_signal"]))
    return pairs, base, gate


def averaged(d):
    """Both slot orders must survive, or the slot term does not cancel."""
    return {k: sum(v) / len(v) for k, v in d.items() if len(v) >= 2}


def main(paths):
    with open(_conditions_path("e1_conditions.json"), encoding="utf-8") as f:
        contrasts = json.load(f)["contrasts"]
    pairs, base, gate = load(paths)
    models = sorted(set(pairs) & set(base))
    missing = sorted((set(pairs) | set(base)) - set(models))

    print("=" * 78)
    print("e1 channel C, rebuilt: the interaction the first run could not give")
    print(f"models with both files: {', '.join(models) or 'none'}")
    if missing:
        print(f"models missing one of the two files: {', '.join(missing)}")
    print("=" * 78)
    if not models:
        sys.exit("\nPass both the pairs CSV and the first choice CSV per model.")

    print("\n" + "-" * 78)
    print("0. THE LETTER-MASS GATE")
    print("-" * 78)
    for m in models:
        low, tot = gate[m]
        print(f"  {m:9}{low} of {tot} reads below {MASS_GATE} "
              f"({low / tot:.1%})")

    rows = []
    for m in models:
        pa = {k: averaged(v) for k, v in pairs[m].items()}
        bs = {k: averaged(v) for k, v in base[m].items()}
        for ct in contrasts:
            for side, single_here, single_there in (
                    ("vs_a", ct["single_a"], ct["single_b"]),
                    ("vs_b", ct["single_b"], ct["single_a"])):
                left = pa.get((ct["name"], side), {})
                right = bs.get(single_there, {})
                ids = sorted(set(left) & set(right))
                if len(ids) < 10:
                    continue
                d = [left[i] - right[i] for i in ids]
                mean, lo, hi = S.boot_ci(d)
                row = {"model": m, "name": ct["name"], "type": ct["type"],
                       "side": side, "n": len(d), "mean": mean,
                       "lo": lo, "hi": hi, "mde": S.mde(d)}
                row.update(S.two_tests(d))
                rows.append(row)
    # floors family-wise per model on either test, never in the same family as
    # the contrasts they protect; contrasts BH on both tests
    S.correct(rows)

    print("\n" + "-" * 78)
    print("1. FLOOR FIRST. Two socially neutral details through the same")
    print("   rebuilt contrast. If this moves, the rebuild failed too.")
    print("-" * 78)
    floor = defaultdict(list)
    for r in rows:
        if r["type"] == "floor":
            floor[r["model"]].append(r["mean"])
            print(f"  {r['model']:9}{r['side']:6}n={r['n']:5}  "
                  f"{r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  "
                  f"MDE +/-{r['mde']:.4f}  Holm, either test={r['bh']:.4g}"
                  + (" MOVES" if r["claimed"] else ""))

    print("\n" + "-" * 78)
    print("2. INTERACTIONS. NEGATIVE = super-additive: the second signal costs")
    print("   MORE when the first is already there.")
    print("-" * 78)
    print(f"\n  {'model':9}{'contrast':22}{'type':17}{'side':6}{'n':>5}"
          f"{'interaction':>13}{'MDE':>10}{'BH':>10}")
    for r in rows:
        if r["type"] == "floor":
            continue
        star = " *" if r["claimed"] else ""
        print(f"  {r['model']:9}{r['name']:22}{r['type']:17}{r['side']:6}"
              f"{r['n']:5}{r['mean']:+13.4f}{r['mde']:10.4f}"
              f"{r['bh']:10.5f}{star}")

    print("\n" + "-" * 78)
    print("3. DO THE TWO DIRECTIONS AGREE. Same interaction, computed from")
    print("   either single. Disagreement means the number is slot-bound.")
    print("-" * 78)
    by = defaultdict(dict)
    for r in rows:
        by[(r["model"], r["name"])][r["side"]] = r
    for (m, name), d in sorted(by.items()):
        if len(d) < 2:
            continue
        a, b = d["vs_a"]["mean"], d["vs_b"]["mean"]
        agree = (a > 0) == (b > 0)
        print(f"  {m:9}{name:22}{'AGREE' if agree else 'DISAGREE':10}"
              f"vs_a {a:+.4f}   vs_b {b:+.4f}   mean {(a + b) / 2:+.4f}")

    print("\n" + "-" * 78)
    print("4. BY SIGNAL TYPE, averaging the two directions.")
    print("-" * 78)
    for m in models:
        print(f"\n  {m}")
        for t in ("form_x_form", "stated_x_stated", "cross", "floor"):
            sel = [r for r in rows if r["model"] == m and r["type"] == t]
            if not sel:
                continue
            mean = sum(r["mean"] for r in sel) / len(sel)
            sig = sum(1 for r in sel if r["claimed"])
            vis = sum(1 for r in sel if abs(r["mean"]) > r["mde"])
            print(f"    {t:18}{len(sel):2} estimate(s)  mean {mean:+.4f}  "
                  f"{sig} significant after BH, {vis} above MDE")
        fl = floor.get(m, [])
        if fl:
            biggest = max((abs(r["mean"]) for r in rows
                           if r["model"] == m and r["type"] != "floor"),
                          default=0.0)
            clean = max(abs(x) for x in fl) <= biggest / 2 if biggest else True
            print(f"    floor is {'CLEAN' if clean else 'CONTAMINATED'}: "
                  f"largest floor {max(abs(x) for x in fl):.4f} against "
                  f"largest social {biggest:.4f}")

    print("\n" + "=" * 78)
    print("Read the floor before anything else. The first version of this")
    print("channel failed exactly there, and being rebuilt is not the same as")
    print("being right.")
    print("=" * 78)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        sys.exit("usage: analyse_e1_choice_pairs.py <pairs csv> <choice csv> ...")
    missing = [a for a in args if not os.path.exists(a)]
    if missing:
        sys.exit(f"missing: {', '.join(missing)}")
    main(args)
