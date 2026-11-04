#!/usr/bin/env python3
"""
analyse_e1_answer_logprob.py - the read to trust for channel A2. No GPU.

    python analyse_e1_answer_logprob.py e1_answer_logprob_qwen.csv [...] \
        --accuracy e1_accuracy_qwen.csv [...]

The accuracy files are optional but the validity check needs them.

READ IN THIS ORDER

0. VALIDITY. Scoring the log probability of the right answer is a reading of
   behaviour only if it tracks whether the model actually solves the task. On
   the NONE condition, the analysis reports the AUC: how often a task the model
   solved has a higher answer probability than a task it failed. 0.5 is no
   relation. A weak AUC means this channel measures something other than
   solving the task, and the rest is reported with that label on it.

1. FLOORS FIRST, all three neutral pairs. A floor is CONTAMINATED if it is
   significant after family-wise (Holm) correction across that model's floors.
   One contaminated floor contaminates the channel for that model; the verdict
   does not get to pick the floor it likes.

2. MAIN EFFECTS against NONE, paired within task. This is the question channel
   A could not answer on qwen and mistral: does a single signal move the model
   at all.

3. INTERACTIONS, only interpreted where the floor is clean.

Floors are corrected family-wise (Holm) per model, because the verdict asks
whether ANY of them moves. Interactions and main effects are each corrected for
false discoveries (Benjamini-Hochberg) across models, as separate families.
Correcting floors and interactions together would let strong interactions make a
quiet floor easier to flag.

TWO TESTS, ONE CONSERVATIVE RULE
--------------------------------
Every estimate is tested twice: signed-rank (Wilcoxon), and a sign-flip
permutation test on the mean, which is the quantity the interval describes. On
this channel they can disagree, because a model that answers many tasks at the
ceiling produces a pile of near-zero differences and a heavy tail. The rule,
fixed before reading which way it cuts:

    floor     CONTAMINATED if EITHER test rejects under Holm
    effect    claimed only if BOTH reject under BH, and the mean and the
              signed-rank statistic lean the same way

Each half can only remove a claim. The share of tasks at the ceiling is printed
with the validity check, because tasks already answered with certainty cannot
move up and every contrast is effectively read on the rest.

TWO MORE GUARDS, ADDED AFTER THE FIRST REAL READ (2026-09-11)
-------------------------------------------------------------
Floors WITHIN DIFFICULTY STRATA. On qwen the three floors were quiet in
aggregate, yet on tasks far from the ceiling floor 1 moved +0.51 and near the
ceiling -0.03, both significant: two movements that cancel in the average. Tasks
are split by their mean log probability over ALL conditions (above log 0.8,
between, below log 0.2). The four cells of an interaction enter that mean with
weights +1, -1, -1, +1, so selecting on it biases the interaction by almost
nothing. A floor that rejects in any stratum, Holm over floors x strata per
model and test, makes the verdict CLEAN IN AGGREGATE ONLY, and no interaction is
claimed from it.

A NEGATIVE INTERACTION IS ONLY "COMPOUNDING" WHEN THE SINGLES ARE COSTS. On qwen
the single stated signals RAISE the answer probability, and together they raise
it less than the sum. That is a benefit that does not add, not a harm that
compounds, and the label follows the direction of the two single effects.
"""

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared")))

import p3_stats as S      # noqa: E402

SINGLES = ["F01", "F04", "S01_s1", "S01_s2", "S08_s1", "S08_s2",
           "ALT1", "ALT2_ONLY", "ALT3", "ALT4_ONLY", "ALT5", "ALT6_ONLY"]


def _conditions_path(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (name, os.path.join(here, name),
                 os.path.join(here, "..", "..", "shared", "conditions", name)):
        if os.path.exists(cand):
            return cand
    raise SystemExit(f"{name} not found")


def load_contrasts():
    out, names = [], set()
    for name in ("e1_conditions.json", "e1_floor_conditions.json"):
        with open(_conditions_path(name), encoding="utf-8") as f:
            for ct in json.load(f)["contrasts"]:
                if ct["name"] not in names:
                    names.add(ct["name"])
                    out.append(ct)
    return out


def load_logprob(paths):
    val = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                v = float(r["logprob_sum"])
                if v == v:
                    val[r["model_key"]][r["condition"]][r["task_id"]] = v
    return val


def load_accuracy(paths):
    acc = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                acc[r["model_key"]][r["condition"]][r["task_id"]] = \
                    int(r["correct"])
    return acc


def auc(pos, neg):
    """P(random positive > random negative), ties counted as one half."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    r_pos = sum(r for r, (_v, lab) in zip(ranks, allv) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (r_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def interaction(cells, ct):
    need = [ct["both"], ct["single_a"], ct["single_b"], ct["none"]]
    if any(c not in cells for c in need):
        return []
    ids = sorted(set.intersection(*[set(cells[c]) for c in need]))
    return [(cells[ct["both"]][i] - cells[ct["single_a"]][i])
            - (cells[ct["single_b"]][i] - cells[ct["none"]][i]) for i in ids]


CEILING = -0.0513   # log(0.95): the answer is already near certain


def estimate(d):
    """p is the claim p-value, max(rank, mean): both tests must reject.

    The maximum of two valid p-values is itself a valid p-value for the
    intersection of the two rejections, so BH over it is legitimate.
    """
    mean, lo, hi = S.boot_ci(d)
    out = {"n": len(d), "mean": mean, "lo": lo, "hi": hi, "mde": S.mde(d)}
    out.update(S.two_tests(d))
    return out


def claimed(r):
    """Both tests reject after BH, and the mean and the signed-rank statistic
    lean the same way. See p3_stats for why not Hodges-Lehmann."""
    return S.claimed(r["mean"], r["rank_dir"], r["bh"])


def bh_cols(rows):
    """BH per test, for display, and BH on the claim p-value."""
    for key in ("rank", "mean"):
        for r, a in zip(rows, S.bh([x[f"p_{key}"] for x in rows])):
            r[f"bh_{key}"] = a


def mark(r):
    if claimed(r):
        return " *"
    if r["bh"] < 0.05:
        return "  (direction unclear: mean and signed ranks disagree)"
    if min(r["bh_rank"], r["bh_mean"]) < 0.05:
        return "  (one test only)"
    return ""


LOGPROB_CUTS = (-0.2231, -1.6094)    # log 0.8 and log 0.2
ACCURACY_CUTS = (0.8, 0.2)           # share of conditions solving the task
MIN_STRATUM = 30


def strata(cells, cuts=LOGPROB_CUTS):
    """{stratum: set of tasks}, by each task's mean over every condition."""
    hi, lo = cuts
    tasks = set.intersection(*[set(v) for v in cells.values()])
    mean = {t: sum(cells[c][t] for c in cells) / len(cells) for t in tasks}
    return {"near ceiling": {t for t in tasks if mean[t] > hi},
            "middle": {t for t in tasks if lo < mean[t] <= hi},
            "far from ceiling": {t for t in tasks if mean[t] <= lo}}


holm_either = S.holm_either


def stratified_floors(cells, contrasts, model, cuts=LOGPROB_CUTS):
    """Every floor in every stratum, Holm across all of them per test.

    Returns (rows, flagged). flagged is empty when no floor moves in any
    stratum on either test.
    """
    out = stratified_rows(cells, contrasts, model, cuts)
    return out, holm_either(out)


def stratified_rows(cells, contrasts, model, cuts=LOGPROB_CUTS):
    """The stratified floor estimates, uncorrected, for joint correction."""
    out = []
    for name, ids in strata(cells, cuts).items():
        if len(ids) < MIN_STRATUM:
            continue
        sub = {c: {t: v for t, v in cells[c].items() if t in ids}
               for c in cells}
        for ct in contrasts:
            if ct["type"] != "floor":
                continue
            d = interaction(sub, ct)
            if len(d) < MIN_STRATUM:
                continue
            e = estimate(d)
            e.update({"model": model, "name": ct["name"], "stratum": name})
            out.append(e)
    return out


def single_effect(cells, cid, none="NONE"):
    ids = set(cells.get(cid, {})) & set(cells.get(none, {}))
    if not ids:
        return float("nan")
    return sum(cells[cid][t] - cells[none][t] for t in ids) / len(ids)


def direction_label(r, eff_a, eff_b):
    """What a claimed interaction means, given the two single effects."""
    if not claimed(r):
        return ""
    if eff_a < 0 and eff_b < 0:
        return "COMPOUNDS: costs more than the sum" if r["mean"] < 0 \
            else "cost does not add"
    if eff_a > 0 and eff_b > 0:
        return "benefit does not add" if r["mean"] < 0 \
            else "benefit compounds"
    return "singles point opposite ways: no direction label"


def floor_verdict(rows, model, strat_rows=()):
    """One family: the model's floors overall and within every stratum.

    CONTAMINATED if a floor moves overall, CLEAN IN AGGREGATE ONLY if one moves
    only within a stratum, CLEAN otherwise.
    """
    floors = [r for r in rows if r["model"] == model and r["type"] == "floor"]
    strat_rows = list(strat_rows)
    flagged = holm_either(floors + strat_rows)
    bad = [r for r in floors if r in flagged]
    if bad:
        verdict = "CONTAMINATED"
    elif any(r in flagged for r in strat_rows):
        verdict = "CLEAN IN AGGREGATE ONLY"
    else:
        verdict = "CLEAN"
    return verdict, floors, bad


def main(argv):
    lp_paths, acc_paths, mode = [], [], "lp"
    for a in argv:
        if a == "--accuracy":
            mode = "acc"
            continue
        (acc_paths if mode == "acc" else lp_paths).append(a)
    if not lp_paths:
        sys.exit("usage: analyse_e1_answer_logprob.py <csv> ... "
                 "[--accuracy <csv> ...]")
    missing = [p for p in lp_paths + acc_paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"missing: {', '.join(missing)}")

    contrasts = load_contrasts()
    val = load_logprob(lp_paths)
    acc = load_accuracy(acc_paths) if acc_paths else {}
    models = sorted(val)

    print("=" * 78)
    print("e1 CHANNEL A2: log probability of the correct answer")
    print(f"models: {', '.join(models)}")
    print("=" * 78)

    print()
    print("-" * 78)
    print("0. VALIDITY. Does a higher answer probability mean the task was")
    print("   actually solved? AUC on the NONE condition, 0.5 = no relation.")
    print("-" * 78)
    for m in models:
        if m not in acc or "NONE" not in acc[m] or "NONE" not in val[m]:
            print(f"  {m:9}no channel A file given for this model")
            continue
        ids = set(acc[m]["NONE"]) & set(val[m]["NONE"])
        pos = [val[m]["NONE"][i] for i in ids if acc[m]["NONE"][i] == 1]
        neg = [val[m]["NONE"][i] for i in ids if acc[m]["NONE"][i] == 0]
        a = auc(pos, neg)
        label = ("strong" if a >= 0.8 else "moderate" if a >= 0.65
                 else "WEAK: this channel does not track solving the task")
        print(f"  {m:9}AUC {a:.3f}  ({len(pos)} solved, {len(neg)} failed)  "
              f"{label}")
    print()
    print("  ceiling: share of NONE tasks answered with P(correct) above 0.95.")
    print("  Those cannot move up, so every contrast is read mostly on the rest.")
    for m in models:
        v = list(val[m].get("NONE", {}).values())
        if v:
            print(f"  {m:9}{sum(1 for x in v if x > CEILING) / len(v):.0%} "
                  f"of {len(v)} tasks at the ceiling")

    rows = []
    for m in models:
        for ct in contrasts:
            d = interaction(val[m], ct)
            if not d:
                continue
            e = estimate(d)
            e.update({"model": m, "name": ct["name"], "type": ct["type"]})
            rows.append(e)
    social_rows = [r for r in rows if r["type"] != "floor"]
    for r, a in zip(social_rows, S.bh([r["p"] for r in social_rows])):
        r["bh"] = a

    print()
    print("-" * 78)
    print("1. FLOORS FIRST. Three neutral pairs, overall and within three")
    print("   difficulty strata, as ONE family per model: Holm per test, then")
    print("   x2 across the two tests. One floor moving anywhere is enough.")
    print("-" * 78)
    print(f"  {'':9}{'':16}{'':6}{'mean':>9} {'95% CI':>19}  {'HL':>8}  "
          f"{'MDE':>6}  {'Holm rank':>9}  {'Holm mean':>9}")
    verdicts, srows_by = {}, {}
    for m in models:
        srows_by[m] = stratified_rows(val[m], contrasts, m)
        verdict, floors, bad = floor_verdict(rows, m, srows_by[m])
        verdicts[m] = verdict
        for r in floors:
            star = " *" if r["holm"] < 0.05 else ""
            print(f"  {m:9}{r['name']:16}n={r['n']:4}{r['mean']:+9.4f} "
                  f"[{r['lo']:+8.4f},{r['hi']:+8.4f}]  {r['hl']:+8.4f}  "
                  f"{r['mde']:6.4f}  {r['holm_rank']:9.3g}  "
                  f"{r['holm_mean']:9.3g}{star}")
        pos = sum(1 for r in floors if r["mean"] > 0)
        print(f"  {m:9}floor verdict: {verdict}"
              + (f", {len(bad)} of {len(floors)} floors significant"
                 if bad else f", all {len(floors)} floors quiet")
              + f"; {pos} of {len(floors)} floor means positive")

    print()
    print("  1b. the same floors WITHIN difficulty strata, corrected in the")
    print("      same family as above. Movements that cancel in the average")
    print("      show here.")
    for m in models:
        srows = srows_by[m]
        flagged = [r for r in srows if r["holm"] < 0.05]
        sizes = {k: len(v) for k, v in strata(val[m]).items()}
        print(f"  {m:9}tasks per stratum: "
              + ", ".join(f"{k} {n}" for k, n in sizes.items()))
        for r in srows:
            star = " *" if r["holm"] < 0.05 else ""
            print(f"    {r['name']:16}{r['stratum']:18}n={r['n']:4}"
                  f"{r['mean']:+9.4f}  Holm {r['holm']:.3g}{star}")
        if verdicts[m] == "CLEAN IN AGGREGATE ONLY":
            print(f"  {m:9}floors quiet in aggregate but {len(flagged)} move "
                  f"within a stratum: CLEAN IN AGGREGATE ONLY")

    print()
    print("-" * 78)
    print("2. MAIN EFFECTS against NONE, paired within task.")
    print("   negative = the condition makes the right answer LESS likely")
    print("-" * 78)
    me = []
    for m in models:
        base = val[m].get("NONE", {})
        for cid in SINGLES:
            cells = val[m].get(cid)
            if not cells or not base:
                continue
            ids = sorted(set(cells) & set(base))
            e = estimate([cells[i] - base[i] for i in ids])
            e.update({"model": m, "cid": cid})
            me.append(e)
    for r, a in zip(me, S.bh([r["p"] for r in me])):
        r["bh"] = a
    bh_cols(me)
    print("   * = both tests reject after BH, and the mean and the signed ranks")
    print("       lean the same way")
    for m in models:
        print()
        print(f"  {m}")
        for r in [x for x in me if x["model"] == m]:
            print(f"    {r['cid']:12}n={r['n']:4}  {r['mean']:+9.4f} "
                  f"[{r['lo']:+8.4f},{r['hi']:+8.4f}]  HL {r['hl']:+7.4f}  "
                  f"BH rank {r['bh_rank']:<9.3g} mean {r['bh_mean']:<9.3g}"
                  f"{mark(r)}")

    print()
    print("-" * 78)
    print("3. INTERACTIONS. NEGATIVE = super-additive. Only read where the")
    print("   floor verdict above is CLEAN.")
    print("-" * 78)
    bh_cols(social_rows)
    by_name = {ct["name"]: ct for ct in contrasts}
    print("   the single effects of each pair are printed beside it: a negative")
    print("   interaction is compounding harm only when both singles are costs")
    for m in models:
        print()
        print(f"  {m}   floor {verdicts[m]}")
        for r in [x for x in rows if x["model"] == m and x["type"] != "floor"]:
            ct = by_name[r["name"]]
            ea = single_effect(val[m], ct["single_a"], ct["none"])
            eb = single_effect(val[m], ct["single_b"], ct["none"])
            r["label"] = direction_label(r, ea, eb)
            print(f"    {r['name']:20}n={r['n']:4}  singles {ea:+.2f} {eb:+.2f}"
                  f"  interaction {r['mean']:+8.4f}  HL {r['hl']:+7.4f}  "
                  f"MDE {r['mde']:.4f}  BH rank {r['bh_rank']:<8.3g} "
                  f"mean {r['bh_mean']:<8.3g}{mark(r)}"
                  + (f"  {r['label']}" if r["label"] else ""))

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    for m in models:
        if verdicts[m] == "CONTAMINATED":
            print(f"  {m:9}floor CONTAMINATED: channel A2 cannot identify an "
                  f"interaction on this model")
            continue
        if verdicts[m] != "CLEAN":
            print(f"  {m:9}floor CLEAN IN AGGREGATE ONLY: floors move within "
                  f"difficulty strata, so no interaction is identified on this "
                  f"model")
            continue
        social = [r for r in rows if r["model"] == m and r["type"] != "floor"]
        counts = defaultdict(int)
        for r in social:
            counts[r["label"] or "not claimed"] += 1
        print(f"  {m:9}floor clean. "
              + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main(sys.argv[1:])
