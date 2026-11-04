#!/usr/bin/env python3
"""
analyse_e1_accuracy.py - the read to trust for e1 channel A. No GPU.

    python analyse_e1_accuracy.py e1_accuracy_qwen.csv e1_accuracy_llama.csv ...

Needs e1_conditions.json in the working directory.

WHAT IT DOES, IN THE ORDER IT MATTERS
-------------------------------------
1. THE FLOOR CHECK RUNS FIRST. Two socially neutral details are stacked the
   same way two disclosures are. If that contrast moves as much as the social
   ones, the instrument is measuring how many distinctive facts a message
   contains, not what those facts say, and nothing below it can be read. This
   check does not exist anywhere in the compounding literature.

2. MAIN EFFECTS, paired within task, exact McNemar (a binomial test on
   discordant pairs, not the chi-squared approximation). These say whether each
   signal costs anything on its own, which is the premise the whole experiment
   stands on. A signal with no main effect cannot have its interaction
   interpreted.

3. INTERACTIONS. Per task i,

       d_i = (y_both - y_A) - (y_B - y_none)

   d_i is in {-2,-1,0,1,2}. Reported as the mean with a bootstrap interval, and
   tested with Wilcoxon signed-rank. NEGATIVE means the pair costs MORE than the
   sum of its parts, which is the super-additive claim of the compounding camp.
   POSITIVE means sub-additive, which is what Paper 2's method 3b and Fu et al.
   report.

4. THE MINIMUM DETECTABLE INTERACTION, next to every estimate, computed from
   the observed spread of d. This channel is binary and n=200. Without this
   number a null reads as "no interaction" when it may only mean "not visible
   from here", and that misreading is the single most likely way this
   experiment gets reported wrongly.

5. ORDER. Every stated-bearing pair was run in both slot orders. If the two
   orders disagree about the sign, the result belongs to the slot and not to
   the person, and it is reported as such rather than averaged away.

6. REFUSAL against the number of signals.

Benjamini-Hochberg runs across the whole family of interaction tests, all
contrasts and all models together.
"""

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p3_stats import (bh, boot_ci, correct, exact_mcnemar,    # noqa: E402
                      mde, two_tests, wilcoxon)


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


# --------------------------------------------------------------------------

def load(paths):
    """Returns acc[model][condition][task_id] and ref[model][condition]."""
    acc = defaultdict(lambda: defaultdict(dict))
    ref = defaultdict(lambda: defaultdict(list))
    meta = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                mk = r["model_key"]
                acc[mk][r["condition"]][r["task_id"]] = int(r["correct"])
                ref[mk][r["condition"]].append(int(r["refused"]))
                meta[r["condition"]] = int(r["n_social_signals"])
    return acc, ref, meta


def paired(acc_model, a, b):
    if a not in acc_model or b not in acc_model:
        return []
    ids = sorted(set(acc_model[a]) & set(acc_model[b]))
    return [(acc_model[a][i], acc_model[b][i]) for i in ids]


def interaction(acc_model, ct):
    cells = [ct["both"], ct["single_a"], ct["single_b"], ct["none"]]
    if any(c not in acc_model for c in cells):
        return [], []
    ids = sorted(set.intersection(*[set(acc_model[c]) for c in cells]))
    d = [(acc_model[ct["both"]][i] - acc_model[ct["single_a"]][i])
         - (acc_model[ct["single_b"]][i] - acc_model[ct["none"]][i])
         for i in ids]
    return d, ids


def main(paths):
    with open(_conditions_path("e1_conditions.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    conditions = cfg["conditions"]
    contrasts = cfg["contrasts"]
    acc, ref, _ = load(paths)
    models = sorted(acc)

    print("=" * 76)
    print("e1 CHANNEL A: task accuracy under stacked signals")
    print(f"models: {', '.join(models)}")
    print("=" * 76)

    # ---- 1. the floor check ------------------------------------------------
    print("\n" + "-" * 76)
    print("1. FLOOR CHECK. Two socially neutral details, stacked the same way.")
    print("   If this moves like the social contrasts do, the instrument reads")
    print("   detail count and nothing below can be believed.")
    print("-" * 76)
    floor = {}
    for m in models:
        ct = next(c for c in contrasts if c["type"] == "floor")
        d, _ = interaction(acc[m], ct)
        if not d:
            print(f"  {m:9}no data")
            continue
        mean, lo, hi = boot_ci(d)
        t = two_tests(d)
        floor[m] = abs(mean)
        print(f"  {m:9}n={len(d):4}  interaction {mean:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  p rank {t['p_rank']:.4f} "
              f"mean {t['p_mean']:.4f}  MDE +/-{mde(d):.4f}")

    # ---- 2. main effects ---------------------------------------------------
    print("\n" + "-" * 76)
    print("2. MAIN EFFECTS against NONE, paired within task, exact McNemar.")
    print("   negative net = the signal COSTS accuracy.")
    print("-" * 76)
    singles = [c["id"] for c in conditions
               if c["n_social_signals"] == 1 or c["id"].startswith("ALT")]
    for m in models:
        print(f"\n  {m}")
        base = acc[m].get("NONE", {})
        if not base:
            print("    no NONE cell")
            continue
        for cid in singles:
            pr = paired(acc[m], cid, "NONE")
            if not pr:
                continue
            b, c, p = exact_mcnemar(pr)
            net = (sum(x for x, _ in pr) - sum(y for _, y in pr)) / len(pr)
            print(f"    {cid:14}n={len(pr):4}  net {net:+.4f}  "
                  f"gained {b:3} lost {c:3}  p={p:.4f}")

    # ---- 3 and 4. interactions --------------------------------------------
    print("\n" + "-" * 76)
    print("3. INTERACTIONS. d = (both - A) - (B - none), per task.")
    print("   NEGATIVE = super-additive, the pair costs MORE than its parts.")
    print("   POSITIVE = sub-additive.")
    print("-" * 76)

    rows = []
    for m in models:
        for ct in contrasts:
            d, _ = interaction(acc[m], ct)
            if not d:
                continue
            mean, lo, hi = boot_ci(d)
            row = {"model": m, "name": ct["name"], "type": ct["type"],
                   "n": len(d), "mean": mean, "lo": lo, "hi": hi,
                   "mde": mde(d)}
            row.update(two_tests(d))
            rows.append(row)
    # floors family-wise per model on either test; contrasts BH on both tests
    correct(rows)

    print(f"\n  {'model':9}{'contrast':22}{'type':17}{'n':>4}  "
          f"{'interaction':>12}  {'95% CI':>20}  {'MDE':>8}  {'p rank':>8}"
          f"  {'p mean':>8}  {'adj':>8}")
    for r in rows:
        star = " *" if r["claimed"] else ""
        print(f"  {r['model']:9}{r['name']:22}{r['type']:17}{r['n']:4}  "
              f"{r['mean']:+12.4f}  [{r['lo']:+7.4f},{r['hi']:+7.4f}]  "
              f"{r['mde']:8.4f}  {r['p_rank']:8.4f}  {r['p_mean']:8.4f}  "
              f"{r['bh']:8.4f}{star}")
    print("\n  adj: BH over max(p rank, p mean) for contrasts, so * needs both")
    print("  tests and the mean leaning with the signed ranks; Holm on either")
    print("  test for the floor, where * means the floor MOVES.")

    print("\n  READ THE MDE COLUMN BEFORE THE p COLUMN. Where |interaction| is")
    print("  smaller than MDE, this channel cannot see an effect of that size,")
    print("  and the result is 'not visible from here', not 'no interaction'.")

    # ---- 5. by type, and the verdict --------------------------------------
    print("\n" + "-" * 76)
    print("4. BY SIGNAL TYPE. The thesis is that form signals compound and")
    print("   stated signals do not.")
    print("-" * 76)
    for m in models:
        print(f"\n  {m}")
        for t in ("form_x_form", "stated_x_stated", "cross", "floor"):
            sel = [r for r in rows if r["model"] == m and r["type"] == t]
            if not sel:
                continue
            mean = sum(r["mean"] for r in sel) / len(sel)
            sig = sum(1 for r in sel if r["claimed"])
            visible = sum(1 for r in sel if abs(r["mean"]) > r["mde"])
            print(f"    {t:18}{len(sel)} contrast(s)  mean {mean:+.4f}  "
                  f"{sig} significant after BH, {visible} above MDE")

    # ---- 6. order ----------------------------------------------------------
    print("\n" + "-" * 76)
    print("5. ORDER. Pairs carrying a stated signal ran in both slot orders.")
    print("   Disagreement in sign means the result belongs to the slot.")
    print("-" * 76)
    pairs_by_stem = defaultdict(list)
    for r in rows:
        stem = r["name"].rstrip("ab").rstrip("_")
        if r["name"][-1] in "ab" or r["name"][:2] in ("P2", "P3", "P4"):
            pairs_by_stem[(r["model"], r["name"][:2], r["type"])].append(r)
    for (m, stem, t), sel in sorted(pairs_by_stem.items()):
        if len(sel) < 2:
            continue
        signs = {1 if r["mean"] > 0 else -1 if r["mean"] < 0 else 0
                 for r in sel}
        verdict = "AGREE" if len(signs) == 1 else "DISAGREE, slot-dependent"
        vals = ", ".join(f"{r['name']}={r['mean']:+.4f}" for r in sel)
        print(f"  {m:9}{stem:5}{verdict:26}{vals}")

    # ---- 7. refusal --------------------------------------------------------
    print("\n" + "-" * 76)
    print("6. REFUSAL against the number of signals. Counted separately,")
    print("   never scored as a wrong answer.")
    print("-" * 76)
    nsig = {c["id"]: c["n_social_signals"] for c in conditions}
    for m in models:
        print(f"\n  {m}")
        by_n = defaultdict(list)
        for cid, vals in ref[m].items():
            by_n[nsig.get(cid, 0)].extend(vals)
        for n in sorted(by_n):
            v = by_n[n]
            print(f"    {n} signal(s)   {sum(v) / len(v):7.2%}  "
                  f"({sum(v)} of {len(v)})")

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 76)
    print("VERDICT, against the expectations registered before the run")
    print("=" * 76)
    for m in models:
        ff = [r for r in rows if r["model"] == m and r["type"] == "form_x_form"]
        ss = [r for r in rows if r["model"] == m and r["type"] == "stated_x_stated"]
        if not ff or not ss:
            continue
        ff_super = all(r["claimed"] and r["mean"] < 0
                       and abs(r["mean"]) > r["mde"] for r in ff)
        ss_null = not any(r["claimed"] for r in ss)
        # the floor verdict is the TEST, not a ratio of magnitudes: an
        # earlier version called mistral's floor contaminated because its
        # +0.0217 was more than half the biggest effect, while the floor test
        # and the three-floor table both say it is quiet
        fl_row = next((r for r in rows if r["model"] == m
                       and r["type"] == "floor"), None)
        floor_ok = not (fl_row and fl_row["claimed"])
        parts = []
        parts.append("form x form super-additive" if ff_super
                     else "form x form NOT super-additive")
        parts.append("stated x stated null" if ss_null
                     else "stated x stated NOT null")
        parts.append("floor clean" if floor_ok else "FLOOR CONTAMINATED")
        support = ff_super and ss_null and floor_ok
        print(f"  {m:9}{'SUPPORTS' if support else 'does not support'} "
              f"the thesis on this channel: {'; '.join(parts)}")
    print("\n  This channel alone does not settle the thesis. Channels B and C")
    print("  carry a continuous outcome and are powered for interactions;")
    print("  this one is the behavioural evidence, not the sensitive one.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        sys.exit("usage: analyse_e1_accuracy.py <csv> [<csv> ...]")
    missing = [a for a in args if not os.path.exists(a)]
    if missing:
        sys.exit(f"missing: {', '.join(missing)}")
    main(args)
