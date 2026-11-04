#!/usr/bin/env python3
"""
analyse_e1w_wrappers.py - the read to trust for e1w. No GPU.

    python analyse_e1w_wrappers.py e1w_wrappers_qwen.csv [...]

Needs e1_conditions.json in the working directory.

WHAT THIS PRODUCES, AND WHY IT IS THE POINT
-------------------------------------------
For each interaction contrast the analysis computes the estimate SIX TIMES, once
per neutral wrapper, then reports:

    the across-wrapper mean          the effect size that goes in the paper
    the across-wrapper spread        the error bar that goes with it
    sign agreement                   how many of six wrappers agree in direction

An effect whose sign flips across wrappers is not an effect, it is a property of
the wrapper. Paper 2's method 3b found exactly that at r=0.13 between two
wrappers, and method 3c then showed that aggregating over six recovers a
defensible estimate for the signals that really are there.

The paper reports an interaction as real only if it clears the across-wrapper
spread. That rule is stricter than significance, and it is the rule none of the
compounding literature applies.
"""

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared")))

import p3_stats as S      # noqa: E402


def _conditions_path(name):
    """Find a condition file whether the analyser is run from its own folder,
    from the repository root, or from a directory of downloaded CSVs."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (name,
                 os.path.join(here, name),
                 os.path.join(here, "..", "..", "shared", "conditions", name),
                 os.path.join(here, "..", "shared", "conditions", name)):
        if os.path.exists(cand):
            return cand
    raise SystemExit(
        f"{name} not found. Build it with the matching builder in "
        f"shared/conditions/, or run this from a directory that has it.")


def load(paths):
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    ref = defaultdict(lambda: defaultdict(list))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                acc[r["model_key"]][r["wrapper"]][r["condition"]][
                    r["task_id"]] = int(r["correct"])
                ref[r["model_key"]][r["condition"]].append(int(r["refused"]))
    return acc, ref


def interaction(cells, ct):
    need = [ct["both"], ct["single_a"], ct["single_b"], ct["none"]]
    if any(c not in cells for c in need):
        return []
    ids = sorted(set.intersection(*[set(cells[c]) for c in need]))
    return [(cells[ct["both"]][i] - cells[ct["single_a"]][i])
            - (cells[ct["single_b"]][i] - cells[ct["none"]][i]) for i in ids]


def main(paths):
    with open(_conditions_path("e1_conditions.json"), encoding="utf-8") as f:
        contrasts = json.load(f)["contrasts"]
    acc, ref = load(paths)
    models = sorted(acc)

    print("=" * 78)
    print("e1w: the same interactions under six neutral wrappers")
    print(f"models: {', '.join(models)}")
    print("=" * 78)

    # ---- baseline movement caused by the wrapper alone ---------------------
    print("\n" + "-" * 78)
    print("1. WHAT THE WRAPPER DOES ON ITS OWN. NONE under each wrapper.")
    print("   In Paper 2 the six wrappers moved baseline accuracy by 2 to 5")
    print("   points before any signal was added.")
    print("-" * 78)
    for m in models:
        cells = []
        for wid in sorted(acc[m]):
            v = list(acc[m][wid].get("NONE", {}).values())
            cells.append(f"{wid}={sum(v) / len(v):.1%}" if v else f"{wid}=-")
        base = [sum(acc[m][w]["NONE"].values()) / len(acc[m][w]["NONE"])
                for w in acc[m] if acc[m][w].get("NONE")]
        spread = (max(base) - min(base)) if base else 0.0
        print(f"  {m:9}{'  '.join(cells)}   spread {spread:.1%}")

    # ---- the interaction, six times ---------------------------------------
    print("\n" + "-" * 78)
    print("2. EACH INTERACTION, ONCE PER WRAPPER.")
    print("   NEGATIVE = super-additive. The across-wrapper spread is the")
    print("   error bar; an effect must clear it to be reported as real.")
    print("-" * 78)

    verdicts = []
    for m in models:
        wrappers = sorted(acc[m])
        print(f"\n  {m}")
        print(f"    {'contrast':22}" + "".join(f"{w:>9}" for w in wrappers)
              + f"{'mean':>10}{'spread':>9}{'signs':>8}")
        for ct in contrasts:
            per = {}
            for wid in wrappers:
                d = interaction(acc[m][wid], ct)
                if d:
                    per[wid] = sum(d) / len(d)
            if len(per) < 2:
                continue
            vals = list(per.values())
            mean = sum(vals) / len(vals)
            spread = max(vals) - min(vals)
            pos = sum(1 for v in vals if v > 0)
            agree = max(pos, len(vals) - pos)
            cells = "".join(f"{per.get(w, float('nan')):+9.3f}"
                            for w in wrappers)
            clears = abs(mean) > spread / 2
            mark = " *" if clears else ""
            print(f"    {ct['name']:22}{cells}{mean:+10.3f}{spread:9.3f}"
                  f"{agree:>4}/{len(vals)}{mark}")
            verdicts.append({"model": m, "name": ct["name"], "type": ct["type"],
                             "mean": mean, "spread": spread,
                             "agree": agree, "n_w": len(vals),
                             "clears": clears})

    # ---- pooled test across wrappers ---------------------------------------
    print("\n" + "-" * 78)
    print("3. POOLED ACROSS WRAPPERS, item-level, with a bootstrap interval.")
    print("   Every task appears once per wrapper, so the pooled estimate")
    print("   treats wrapper as a replicate rather than as extra items.")
    print("-" * 78)
    rows = []
    for m in models:
        for ct in contrasts:
            per_wrapper = []
            for wid in sorted(acc[m]):
                d = interaction(acc[m][wid], ct)
                if d:
                    per_wrapper.append(sum(d) / len(d))
            if len(per_wrapper) < 2:
                continue
            mean, lo, hi = S.boot_ci(per_wrapper)
            row = {"model": m, "name": ct["name"], "type": ct["type"],
                   "n": len(per_wrapper), "mean": mean, "lo": lo, "hi": hi}
            row.update(S.two_tests(per_wrapper))
            rows.append(row)
    # floors family-wise per model on either test; contrasts BH on both tests
    S.correct(rows)
    print(f"\n  {'model':9}{'contrast':22}{'type':17}{'wraps':>6}"
          f"{'mean':>10}{'95% CI':>22}{'BH':>9}")
    for r in rows:
        star = " *" if r["claimed"] else ""
        print(f"  {r['model']:9}{r['name']:22}{r['type']:17}{r['n']:6}"
              f"{r['mean']:+10.3f}  [{r['lo']:+8.3f},{r['hi']:+8.3f}]"
              f"{r['bh']:9.4f}{star}")
    print("\n  With six wrappers the pooled test has six observations, so it is")
    print("  weak by construction. It is reported for completeness; the sign")
    print("  agreement and the spread above carry the argument.")

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT: which interactions survive a change of neutral wrapper")
    print("=" * 78)
    for v in verdicts:
        if v["type"] == "floor":
            continue
        state = ("SURVIVES" if v["clears"] and v["agree"] == v["n_w"]
                 else "wrapper-dependent" if v["agree"] < v["n_w"]
                 else "within the wrapper spread")
        print(f"  {v['model']:9}{v['name']:22}{state:22}"
              f"mean {v['mean']:+.3f}, spread {v['spread']:.3f}, "
              f"{v['agree']}/{v['n_w']} wrappers agree in sign")
    print("\n  An interaction that is wrapper-dependent is a fact about the")
    print("  wrapper. Report it as one, do not average it away.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        sys.exit("usage: analyse_e1w_wrappers.py <csv> [<csv> ...]")
    missing = [a for a in args if not os.path.exists(a)]
    if missing:
        sys.exit(f"missing: {', '.join(missing)}")
    main(args)
