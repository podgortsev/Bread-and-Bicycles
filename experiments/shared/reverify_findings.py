#!/usr/bin/env python3
"""
reverify_findings.py - recompute every headline number from the raw CSVs,
using arithmetic written separately from the analysers.

    python reverify_findings.py

WHY THIS EXISTS
---------------
The validators check the analysers against synthetic data with a known answer.
They do not check that the analysers were pointed at the right cells, or that
the reading of the real data is right. Those are different failures and the
second one is the expensive kind: a correct program computing the wrong
contrast.

So this file imports nothing from any analyser. It re-reads the CSVs, rebuilds
each headline contrast by hand, and asserts it matches what was reported. It is
the same idea as `validate_tasks.py`, which re-derives every answer key with
arithmetic written separately from the generator.

Every number checked here appears in the summary given to the author.
"""

import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          f"{('  ' + detail) if detail else ''}")


def close(a, b, tol=0.0006):
    return abs(a - b) <= tol


def rows(folder, model, stem):
    p = os.path.join(EXP, folder, "outputs", model, f"{stem}_{model}.csv")
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(v):
    return sum(v) / len(v) if v else float("nan")


# ==========================================================================
print("=" * 78)
print("1. e1 CHANNEL A: the interaction that carries the paper")
print("=" * 78)

acc = defaultdict(lambda: defaultdict(dict))
for m in ("qwen", "llama", "mistral"):
    for r in rows("e1-type-x-count", m, "e1_accuracy"):
        acc[m][r["condition"]][r["task_id"]] = int(r["correct"])

for m in ("qwen", "llama", "mistral"):
    ok(f"{m}: 17 conditions x 600 tasks",
       len(acc[m]) == 17 and all(len(v) == 600 for v in acc[m].values()))


def inter(m, both, a, b, none="NONE"):
    ids = sorted(set(acc[m][both]) & set(acc[m][a]) & set(acc[m][b])
                 & set(acc[m][none]))
    return [(acc[m][both][i] - acc[m][a][i])
            - (acc[m][b][i] - acc[m][none][i]) for i in ids]


d = inter("llama", "P2_S01S08", "S01_s1", "S08_s2")
ok("llama stated x stated, order a = +0.0383", close(mean(d), 0.0383),
   f"{mean(d):+.4f}")
d2 = inter("llama", "P2_S08S01", "S08_s1", "S01_s2")
ok("llama stated x stated, order b = +0.0717", close(mean(d2), 0.0717),
   f"{mean(d2):+.4f}")
ok("both orders POSITIVE, so the pair costs LESS than its parts",
   mean(d) > 0 and mean(d2) > 0)

# a sign test written from scratch, no scipy
from math import comb                                          # noqa: E402
nz = [x for x in d2 if x != 0]
pos = sum(1 for x in nz if x > 0)
k = min(pos, len(nz) - pos)
p_two = min(1.0, 2 * sum(comb(len(nz), i) for i in range(k + 1))
            / 2 ** len(nz))
ok("llama order b is significant on a hand-written sign test too",
   p_two < 0.01, f"{pos} up, {len(nz) - pos} down of {len(nz)}, p={p_two:.2g}")

d3 = inter("llama", "P1_F01F04", "F01", "F04")
ok("llama form x form is near zero, not super-additive",
   close(mean(d3), -0.0100), f"{mean(d3):+.4f}")

# main effects, recomputed
for m, cid, want in (("llama", "S01_s1", -0.0500), ("llama", "S08_s1", -0.0650),
                     ("qwen", "S01_s1", 0.0033), ("mistral", "S01_s1", 0.0083)):
    ids = sorted(set(acc[m][cid]) & set(acc[m]["NONE"]))
    got = mean([acc[m][cid][i] - acc[m]["NONE"][i] for i in ids])
    ok(f"{m} main effect {cid} = {want:+.4f}", close(got, want), f"{got:+.4f}")

flo = inter("qwen", "ALT_BOTH", "ALT1", "ALT2_ONLY")
ok("qwen floor on channel A is exactly zero", close(mean(flo), 0.0),
   f"{mean(flo):+.4f}")

# ==========================================================================
print()
print("=" * 78)
print("2. e1 CHANNEL C: why its interaction contrast is invalid")
print("=" * 78)

ch = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in rows("e1-type-x-count", "qwen", "e1_choice"):
    if float(r["mass"]) < 0.5:
        continue
    ch["qwen"][r["condition"]][f"{r['profile']}:{r['question']}"].append(
        float(r["margin_signal"]))

marg = {}
for cid, items in ch["qwen"].items():
    marg[cid] = mean([mean(v) for v in items.values() if len(v) >= 2])

ok("qwen ALT1 margin = -4.386", close(marg["ALT1"], -4.3856, 0.002),
   f"{marg['ALT1']:+.4f}")
ok("qwen ALT2_ONLY margin = -1.817", close(marg["ALT2_ONLY"], -1.8166, 0.002),
   f"{marg['ALT2_ONLY']:+.4f}")
ok("qwen ALT_BOTH margin = -2.638", close(marg["ALT_BOTH"], -2.6381, 0.002),
   f"{marg['ALT_BOTH']:+.4f}")
ok("the both-cell sits BETWEEN the two singles, not below their sum",
   marg["ALT1"] < marg["ALT_BOTH"] < marg["ALT2_ONLY"],
   f"{marg['ALT1']:.2f} < {marg['ALT_BOTH']:.2f} < {marg['ALT2_ONLY']:.2f}")
additive = marg["ALT1"] + marg["ALT2_ONLY"]
ok("an additive model predicts -6.20 and the data give -2.64",
   close(additive, -6.2022, 0.003) and marg["ALT_BOTH"] > additive + 3,
   f"predicted {additive:.2f}, observed {marg['ALT_BOTH']:.2f}")
ok("so the reported floor of +3.56 is arithmetic, not a finding",
   close(marg["ALT_BOTH"] - marg["ALT1"] - marg["ALT2_ONLY"], 3.5641, 0.004),
   f"{marg['ALT_BOTH'] - marg['ALT1'] - marg['ALT2_ONLY']:+.4f}")
ok("mistral screen reader HELPS, replicating Paper 2 method 6b (+7.14)",
   True, "checked below")

ch2 = defaultdict(lambda: defaultdict(list))
for r in rows("e1-type-x-count", "mistral", "e1_choice"):
    if float(r["mass"]) < 0.5:
        continue
    ch2[r["condition"]][f"{r['profile']}:{r['question']}"].append(
        float(r["margin_signal"]))
m_s01 = mean([mean(v) for v in ch2["S01_s2"].values() if len(v) >= 2])
ok("  mistral S01_s2 = +7.145 against Paper 2's +7.14",
   close(m_s01, 7.1454, 0.002), f"{m_s01:+.4f}")

# ==========================================================================
print()
print("=" * 78)
print("3. e1 CHANNEL B: the floor that failed")
print("=" * 78)

tr = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in rows("e1-type-x-count", "qwen", "e1_traits"):
    if r["valence"] != "positive":
        continue
    tr["qwen"][r["condition"]][r["profile"]].append(float(r["logprob"]))
val = {c: {p: mean(v) for p, v in d.items()} for c, d in tr["qwen"].items()}
ids = sorted(set(val["ALT_BOTH"]) & set(val["ALT1"]) & set(val["ALT2_ONLY"])
             & set(val["NONE"]))
fl = mean([(val["ALT_BOTH"][i] - val["ALT1"][i])
           - (val["ALT2_ONLY"][i] - val["NONE"][i]) for i in ids])
ok("qwen channel B floor = +0.5533", close(fl, 0.5533, 0.002), f"{fl:+.4f}")
soc = mean([(val["P2_S01S08"][i] - val["S01_s1"][i])
            - (val["S08_s2"][i] - val["NONE"][i]) for i in ids])
ok("the floor is comparable with a social contrast, so the channel is unusable",
   abs(fl) > abs(soc) * 0.5,
   f"floor {fl:+.4f} against stated x stated {soc:+.4f}")

# ==========================================================================
print()
print("=" * 78)
print("4. e2b: Liu et al. do not replicate")
print("=" * 78)

b = defaultdict(dict)
for r in rows("e2b-legibility", "llama", "e2b_legibility"):
    b[r["arm"]][r["task_id"]] = int(r["correct"])
a = defaultdict(dict)
for r in rows("e1-type-x-count", "llama", "e1_accuracy"):
    for arm, cid in (("std", "NONE"), ("esl", "F01"), ("typo", "F04"),
                     ("esl_typo", "P1_F01F04")):
        if r["condition"] == cid:
            a[arm][r["task_id"]] = int(r["correct"])

ids = sorted(set.intersection(*[set(b[k]) for k in b],
                              *[set(a[k]) for k in a]))
db = [(b["esl_typo"][i] - b["esl"][i]) - (b["typo"][i] - b["std"][i])
      for i in ids]
da = [(a["esl_typo"][i] - a["esl"][i]) - (a["typo"][i] - a["std"][i])
      for i in ids]
ok("llama arm B (task corrupted) interaction = +0.0533",
   close(mean(db), 0.0533), f"{mean(db):+.4f}")
ok("llama arm A (task clean) interaction = -0.0100",
   close(mean(da), -0.0100), f"{mean(da):+.4f}")
diff = [x - y for x, y in zip(db, da)]
ok("arm B minus arm A = +0.0633", close(mean(diff), 0.0633),
   f"{mean(diff):+.4f}")
ok("arm B is SUB-additive, the opposite of the published claim",
   mean(db) > 0)
ok("corrupting the task does cost accuracy on its own",
   mean(b["typo"].values()) < mean(b["std"].values()),
   f"typo {mean(list(b['typo'].values())):.1%} against "
   f"std {mean(list(b['std'].values())):.1%}")

# ==========================================================================
print()
print("=" * 78)
print("5. e3: a credential does not buy back age")
print("=" * 78)

num = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for m in ("qwen", "llama", "mistral"):
    for r in rows("e3-offsetting", m, "e3_numeric"):
        if r["refused"] == "1" or not r["value"]:
            continue
        num[m][r["measure"]][r["condition"]][r["profile"]] = float(r["value"])

for m, meas, want_alone, want_gap in (("llama", "RATING", -6.707, -6.560),
                                      ("mistral", "SCORE", -39.282, -37.157)):
    cells = num[m][meas]
    ids = sorted(set(cells["S08_s1"]) & set(cells["NONE"]))
    alone = mean([cells["S08_s1"][i] - cells["NONE"][i] for i in ids])
    ids2 = sorted(set(cells["CRED_S08"]) & set(cells["NONE"]))
    gap = mean([cells["CRED_S08"][i] - cells["NONE"][i] for i in ids2])
    rec = 1 - gap / alone
    ok(f"{m} {meas}: age alone {want_alone:.2f}, with credential {want_gap:.2f}",
       close(alone, want_alone, 0.01) and close(gap, want_gap, 0.01),
       f"{alone:.3f} and {gap:.3f}, recovered {rec:.2f}")
    ok(f"{m} {meas}: recovery is essentially zero", abs(rec) < 0.10,
       f"{rec:.3f}")

# ==========================================================================
print()
print("=" * 78)
print("6. e5: llama's self-report is a constant")
print("=" * 78)

r5 = rows("e5-attribution", "llama", "e5_attribution")
vals = [float(x[k]) for x in r5 if x["parsed"] == "1"
        for k in ("rating_a", "rating_b")]
ok("llama produced 840 ratings", len(vals) == 840, f"{len(vals)}")
ok("every one of them is exactly 0", all(v == 0.0 for v in vals),
   f"min {min(vals)}, max {max(vals)}")

r5q = rows("e5-attribution", "qwen", "e5_attribution")
form, stated = [], []
for x in r5q:
    if x["parsed"] != "1":
        continue
    for which, sig in (("a", x["signal_a"]), ("b", x["signal_b"])):
        (form if sig.startswith("F") else stated).append(
            float(x[f"rating_{which}"]))
ok("qwen claims 9.65 for cues it was never told about",
   close(mean(form), 9.65, 0.01), f"{mean(form):.3f}")
ok("qwen claims 3.15 for the disclosures it WAS told about",
   close(mean(stated), 3.15, 0.01), f"{mean(stated):.3f}")
ok("qwen rates the unstated cue HIGHER than the stated one",
   mean(form) > mean(stated))

# ==========================================================================
print()
print("=" * 78)
print("7. e1 CHANNEL C2: the rebuilt contrast, and why it cannot work")
print("=" * 78)


def pairs_margins(model):
    d = defaultdict(lambda: defaultdict(list))
    for r in rows("e1-type-x-count", model, "e1_choice_pairs"):
        if float(r["mass"]) >= 0.5:
            d[(r["contrast"], r["side"])][f"{r['profile']}:{r['question']}"].append(
                float(r["margin_left"]))
    return {k: {i: mean(v) for i, v in x.items() if len(v) == 2}
            for k, x in d.items()}


def base_margins(model):
    d = defaultdict(lambda: defaultdict(list))
    for r in rows("e1-type-x-count", model, "e1_choice"):
        if float(r["mass"]) >= 0.5:
            d[r["condition"]][f"{r['profile']}:{r['question']}"].append(
                float(r["margin_signal"]))
    return {k: {i: mean(v) for i, v in x.items() if len(v) == 2}
            for k, x in d.items()}


def inter(pm, bm, contrast, side, single_a, single_b):
    other = single_b if side == "vs_a" else single_a
    left, right = pm[(contrast, side)], bm[other]
    return {i: left[i] - right[i] for i in set(left) & set(right)}


for model, fa, fb, cyc_alone, cyc_bread in (
        ("qwen", 0.6469, 6.4400, -4.386, 2.054),
        ("llama", 0.6303, 0.7013, -0.326, 0.375)):
    pm, bm = pairs_margins(model), base_margins(model)
    fl_a = inter(pm, bm, "FLOOR_ALTxALT", "vs_a", "ALT1", "ALT2_ONLY")
    fl_b = inter(pm, bm, "FLOOR_ALTxALT", "vs_b", "ALT1", "ALT2_ONLY")
    ok(f"{model} rebuilt floor, both directions, as reported",
       close(mean(list(fl_a.values())), fa, 0.002)
       and close(mean(list(fl_b.values())), fb, 0.002),
       f"{mean(list(fl_a.values())):+.4f} and {mean(list(fl_b.values())):+.4f}")
    alone = mean(list(bm["ALT1"].values()))
    with_bread = mean(list(pm[("FLOOR_ALTxALT", "vs_b")].values()))
    ok(f"{model}: the same neutral clause FLIPS SIGN with its neighbour",
       alone < 0 < with_bread and close(alone, cyc_alone, 0.01)
       and close(with_bread, cyc_bread, 0.01),
       f"cycling alone {alone:+.3f}, cycling next to bread {with_bread:+.3f}")
    so_a = inter(pm, bm, "P1_FORMxFORM", "vs_a", "F01", "F04")
    so_b = inter(pm, bm, "P1_FORMxFORM", "vs_b", "F01", "F04")
    ca = mean([so_a[i] - fl_a[i] for i in set(so_a) & set(fl_a)])
    cb = mean([so_b[i] - fl_b[i] for i in set(so_b) & set(fl_b)])
    ok(f"{model}: floor-corrected form pair disagrees in sign between directions",
       (ca > 0) != (cb > 0),
       f"{ca:+.3f} and {cb:+.3f}, so the correction is not a stable reference")

print()
print("=" * 78)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("every headline number re-derived independently of the analysers.")
print("=" * 78)
