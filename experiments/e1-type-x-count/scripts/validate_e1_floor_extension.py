#!/usr/bin/env python3
"""
validate_e1_floor_extension.py - offline check of the floor extension and
channel A2. No GPU, no torch.

    python validate_e1_floor_extension.py

WHAT IT CHECKS
--------------
A. the new neutral pairs: neutrality, length, NONE byte-identical to e1, and
   contrasts built the same way as e1's floor
B. channel A2 job construction: the union of both condition sets, and prompts
   identical to channel A's, character for character
C. the parameterised runners write to their OWN files and never to the main
   run's, and the channel A summary accepts the model key it is called with
D. the driver: its steps, their files, and their output names
E. the A2 analysis: the validity AUC, a known interaction recovered, a null not
   detected, and a contaminated floor turning the verdict CONTAMINATED even when
   only one of three floors moves
F. the side-by-side floor table: its interaction arithmetic and its two
   repeatability checks

What it cannot check: score_answers' token indexing needs a real model. That is
proven on the GPU instead, by P3Model.check_answer_scoring, before any row is
written.
"""

import inspect
import json
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.abspath(os.path.join(HERE, "..", "..", "shared"))
CONDS = os.path.join(SHARED, "conditions")
for p in (HERE, SHARED, CONDS):
    sys.path.insert(0, p)

import analyse_e1_answer_logprob as A2A      # noqa: E402
import analyse_e1_floors as FL               # noqa: E402
import build_e1_conditions as B1             # noqa: E402
import build_e1_floor_conditions as BF       # noqa: E402
import run_e1_accuracy as ACC                # noqa: E402
import run_e1_answer_logprob as ANS          # noqa: E402
import run_e1_choice_pairs as PAIRS          # noqa: E402
import run_e1_traits as TRAITS               # noqa: E402

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          f"{('  ' + detail) if detail else ''}")


def close(a, b, tol):
    return abs(a - b) <= tol


# ==========================================================================
print("=" * 74)
print("A. THE NEW NEUTRAL PAIRS")
print("=" * 74)
rows = BF.build()
problems = BF.check(rows)
ok("floor conditions pass their own checks", not problems,
   "; ".join(problems[:2]))
ok("seven conditions: NONE and two pairs of three cells",
   len(rows) == 7, f"{len(rows)}")
e1_none = next(r for r in B1.build() if r["id"] == "NONE")
ok("NONE is byte-identical to e1's",
   next(r for r in rows if r["id"] == "NONE")["lead_in"] == e1_none["lead_in"])
ok("two floor contrasts, both of type floor",
   len(BF.CONTRASTS) == 2 and all(c[1] == "floor" for c in BF.CONTRASTS))
ok("no social signal anywhere",
   all(r["n_social_signals"] == 0 for r in rows))

saved = dict(BF.NEW_CLAUSES)
BF.NEW_CLAUSES["alt6"] = "I have been reading a lot of crime novels."
ok("the neutrality check REJECTS a cue word",
   any("social cue" in p for p in BF.check(BF.build())),
   "'crime' was the reason the first draft was rewritten")
BF.NEW_CLAUSES["alt6"] = "I am seventy and read a lot of history."
ok("the neutrality check REJECTS an age cue",
   any("social cue" in p for p in BF.check(BF.build())))
BF.NEW_CLAUSES.clear()
BF.NEW_CLAUSES.update(saved)
ok("clauses restored after the negative tests", not BF.check(BF.build()))

# ==========================================================================
print()
print("=" * 74)
print("B. CHANNEL A2 JOBS")
print("=" * 74)
tmp = tempfile.mkdtemp(prefix="floorval_")
cwd = os.getcwd()
try:
    shutil.copy(os.path.join(CONDS, "e1_conditions.json"), tmp)
    BF_json = os.path.join(tmp, "e1_floor_conditions.json")
    with open(BF_json, "w", encoding="utf-8") as f:
        json.dump({"conditions": rows,
                   "contrasts": [{"name": n, "type": t, "both": c[0],
                                  "single_a": c[1], "single_b": c[2],
                                  "none": c[3]} for n, t, c in BF.CONTRASTS]},
                  f)
    os.chdir(tmp)
    conds, contrasts = ANS.load_conditions()
    ids = [c["id"] for c in conds]
    ok("A2 uses 23 conditions, NONE counted once",
       len(conds) == 23 and ids.count("NONE") == 1, f"{len(conds)}")
    ok("A2 carries three floors and seven social contrasts",
       sum(1 for c in contrasts if c["type"] == "floor") == 3
       and len(contrasts) == 10, f"{len(contrasts)} contrasts")

    tasks = [{"id": f"T{i:03d}", "family": "x",
              "question": f"What is {i} plus {i}?", "answer": str(2 * i),
              "answer_kind": "number"} for i in range(1, 6)]
    jobs = ANS.build_jobs(conds, tasks)
    ok("A2 job count is conditions x tasks", len(jobs) == 23 * 5)
    c0, t0 = conds[3], tasks[2]
    channel_a = f"{c0['lead_in']}\n\n{t0['question']}\n\n{ACC.INSTRUCTION}"
    j = next(x for x in jobs if x[0] == c0["id"] and x[1] == t0["id"])
    ok("A2 prompt is identical to channel A's, character for character",
       j[4] == channel_a)
    ok("A2 scores the answer key, not anything the model generated",
       j[3] == t0["answer"])

    # ======================================================================
    print()
    print("=" * 74)
    print("C. THE RUNNERS WRITE TO THEIR OWN FILES")
    print("=" * 74)
    ok("channel A default file is unchanged",
       ACC.raw_path("qwen").endswith("e1_accuracy_qwen.csv"))
    ok("channel A floor file is separate",
       ACC.raw_path("qwen", "e1_accuracy_floor").endswith(
           "e1_accuracy_floor_qwen.csv"))
    ok("channel B floor file is separate",
       TRAITS.raw_path("qwen", "e1_traits_floor").endswith(
           "e1_traits_floor_qwen.csv"))
    ok("channel C2 floor file is separate",
       PAIRS.raw_path("qwen", "e1_choice_pairs_floor").endswith(
           "e1_choice_pairs_floor_qwen.csv"))
    ok("A2 file name", ANS.raw_path("qwen").endswith(
        "e1_answer_logprob_qwen.csv"))

    fconds, fcontrasts = ACC.load_conditions("e1_floor_conditions.json")
    ok("channel A loads the floor condition set when asked", len(fconds) == 7)
    profiles = ["a profile", "another profile"]
    tj = TRAITS.build_jobs(fconds, profiles, coherence=False)
    ok("channel B with coherence off scores only positive words",
       all(x[4] == "positive" for x in tj)
       and len(tj) == 7 * 2 * len(TRAITS.PHRASINGS) * len(TRAITS.POSITIVE),
       f"{len(tj)} jobs")
    comps = PAIRS.build_comparisons(fcontrasts)
    ok("channel C2 on the floor set gives two floors x two sides",
       len(comps) == 4 and {c[0] for c in comps}
       == {"FLOOR2_ALTxALT", "FLOOR3_ALTxALT"})

    sig = inspect.signature(ACC.summary)
    ok("channel A summary now takes the model key it is called with",
       "model_key" in sig.parameters,
       "it took no argument before and failed on every run")
finally:
    os.chdir(cwd)
    shutil.rmtree(tmp, ignore_errors=True)

# ==========================================================================
print()
print("=" * 74)
print("D. THE DRIVER")
print("=" * 74)
import run_p3_floor as DRV                     # noqa: E402
st = DRV.steps()
ok("five steps in the agreed order",
   [s[0] for s in st] == ["e1c_floor", "e1p_floor", "e1b_floor", "e1a2",
                          "e1a_floor"])
ok("every step writes to a name the main run never used",
   all(s[5].endswith("_floor") or s[5] == "e1_answer_logprob" for s in st))
ok("the answer step needs both condition files and the tasks",
   set(st[3][4]) == {"e1_conditions.json", "e1_floor_conditions.json",
                     "tasks.json"})

# ==========================================================================
print()
print("=" * 74)
print("E. THE A2 ANALYSIS")
print("=" * 74)
ok("AUC of perfectly separated values is 1", A2A.auc([3, 4, 5], [0, 1, 2]) == 1)
ok("AUC of reversed values is 0", A2A.auc([0, 1], [5, 6]) == 0)
ok("AUC of identical values is one half", A2A.auc([1, 1], [1, 1]) == 0.5)

all_contrasts = contrasts


def world(floor_shift, social_shift, seed, n=600, noise=0.4):
    rng = random.Random(seed)
    base = {f"T{i:03d}": rng.gauss(-2.0, 1.0) for i in range(n)}
    shift = {c["id"]: 0.0 for c in conds}
    for c in conds:
        if c["n_social_signals"] == 1:
            shift[c["id"]] = -0.3
        if c["id"].startswith("ALT") and "BOTH" not in c["id"]:
            shift[c["id"]] = -0.2
        if c["n_social_signals"] == 2:
            shift[c["id"]] = -0.6
        if c["id"] == "ALT_BOTH" or c["id"].startswith("ALT_BOTH_"):
            shift[c["id"]] = -0.4
    for ct in all_contrasts:
        if ct["type"] == "floor" and ct["name"] in floor_shift:
            shift[ct["both"]] += floor_shift[ct["name"]]
        if ct["type"] != "floor" and ct["name"] in social_shift:
            shift[ct["both"]] += social_shift[ct["name"]]
    return {c: {t: v + shift[c] + rng.gauss(0, noise)
                for t, v in base.items()} for c in shift}


def rows_for(cells, model="qwen"):
    out = []
    for ct in all_contrasts:
        d = A2A.interaction(cells, ct)
        e = A2A.estimate(d)
        e.update({"model": model, "name": ct["name"], "type": ct["type"]})
        out.append(e)
    import p3_stats as S
    social = [r for r in out if r["type"] != "floor"]
    for r, a in zip(social, S.bh([r["p"] for r in social])):
        r["bh"] = a
    return out


r = rows_for(world({}, {"P1_FORMxFORM": -0.5}, seed=3))
verdict, floors, bad = A2A.floor_verdict(r, "qwen")
p1 = next(x for x in r if x["name"] == "P1_FORMxFORM")
ok("clean floors give a CLEAN verdict", verdict == "CLEAN" and len(floors) == 3)
ok("an injected -0.5 interaction is recovered",
   close(p1["mean"], -0.5, 0.1) and p1["bh"] < 0.05, f"{p1['mean']:+.3f}")
others = [x for x in r if x["type"] != "floor" and x["name"] != "P1_FORMxFORM"]
ok("the other social contrasts stay null",
   all(x["bh"] >= 0.05 for x in others))

r = rows_for(world({"FLOOR3_ALTxALT": 0.5}, {}, seed=4))
verdict, floors, bad = A2A.floor_verdict(r, "qwen")
ok("ONE moving floor out of three makes the verdict CONTAMINATED",
   verdict == "CONTAMINATED" and len(bad) == 1,
   "the verdict does not get to pick the floor it likes")

fp = tested = alarms = 0
seeds = 30
for s in range(seeds):
    r_ = rows_for(world({}, {}, seed=100 + s))
    for x in r_:
        if x["type"] != "floor":
            tested += 1
            fp += x["bh"] < 0.05
    alarms += A2A.floor_verdict(r_, "qwen")[0] == "CONTAMINATED"
ok("social false-positive rate near nominal with nothing injected",
   fp / tested <= 0.06, f"{fp}/{tested}")
ok("three null floors raise a false alarm at about 5 percent or less",
   alarms / seeds <= 0.10, f"{alarms} of {seeds}")

quiet = rows_for(world({}, {}, seed=7))
loud = rows_for(world({}, {"P1_FORMxFORM": -1.5, "P2a_STATEDxSTATED": -1.5},
                      seed=7))
A2A.floor_verdict(quiet, "qwen")
A2A.floor_verdict(loud, "qwen")
hq = [x["holm"] for x in quiet if x["type"] == "floor"]
hl = [x["holm"] for x in loud if x["type"] == "floor"]
ok("floor correction does not depend on how strong the interactions are",
   hq == hl, "a joint BH would have loosened it as the interactions grew")

# ==========================================================================
print()
print("=" * 74)
print("F. THE SIDE-BY-SIDE FLOOR TABLE")
print("=" * 74)
cells = {"NONE": {"a": 0.0}, "ALT3": {"a": -1.0}, "ALT4_ONLY": {"a": -2.0},
         "ALT_BOTH_34": {"a": -2.5}}
ct2 = {"name": "FLOOR2_ALTxALT", "type": "floor", "both": "ALT_BOTH_34",
       "single_a": "ALT3", "single_b": "ALT4_ONLY", "none": "NONE"}
ok("2x2 floor arithmetic: (both - a) - (b - none)",
   FL.interaction(cells, ct2) == [(-2.5 - -1.0) - (-2.0 - 0.0)])
pairs = {("FLOOR2_ALTxALT", "vs_a"): {"i": -1.2},
         ("FLOOR2_ALTxALT", "vs_b"): {"i": -0.7}}
base = {"ALT3": {"i": -1.0}, "ALT4_ONLY": {"i": -2.0}}
ok("pairs floor, side a: margin(both vs a) - margin(b vs none)",
   FL.pairs_interaction(pairs, base, ct2, "vs_a") == [-1.2 - -2.0])
ok("pairs floor, side b: margin(both vs b) - margin(a vs none)",
   FL.pairs_interaction(pairs, base, ct2, "vs_b") == [-0.7 - -1.0])
ok("repeatability: identical runs agree on 100 percent",
   FL.agreement({"t1": 1, "t2": 0}, {"t1": 1, "t2": 0}) == 1.0)
ok("repeatability: one flipped task of two is 50 percent",
   FL.agreement({"t1": 1, "t2": 0}, {"t1": 1, "t2": 1}) == 0.5)
ok("repeatability: largest difference",
   close(FL.max_abs_diff({"p": -3.0, "q": -2.0}, {"p": -3.25, "q": -2.0}),
         0.25, 1e-12))

# ==========================================================================
print()
print("=" * 74)
print("G. A CEILING AND A HEAVY TAIL: WHEN THE TWO TESTS DISAGREE")
print("=" * 74)
# Sections E and F use Gaussian noise, which is why they never saw this. Real
# qwen data: 46 percent of tasks at the ceiling, a third of differences near
# zero, and a floor with mean +0.18 (interval excluding zero) and signed-rank
# p 0.87.
import numpy as np                              # noqa: E402
import p3_stats as S                            # noqa: E402

ok("sign_flip: twenty identical positive differences reject hard",
   S.sign_flip([1.0] * 20, reps=4000) <= 2 / 4001)
ok("sign_flip: a mean of exactly zero gives p = 1",
   S.sign_flip([1.0, -1.0] * 10) == 1.0)
ok("hodges_lehmann of 1, 2, 3 is 2", S.hodges_lehmann([1.0, 2.0, 3.0]) == 2.0)
ok("hodges_lehmann ignores one wild value where the mean does not",
   abs(S.hodges_lehmann([0.1] * 30 + [50.0])) < 0.2)


def ceiling_d(seed, big=0.0, pbig=0.0, small=0.0, shift=0.0, n=600):
    rng = np.random.default_rng(seed)
    at = rng.random(n) < 0.45
    noise = rng.standard_t(3, n) * 0.35
    eff = np.where(rng.random(n) < pbig, big, small) + shift
    return list(np.where(at, 0.0, noise + eff))


def floor_rows(floor_ds):
    out = []
    for i, d in enumerate(floor_ds):
        e = A2A.estimate(d)
        e.update({"model": "qwen", "name": f"F{i}", "type": "floor"})
        out.append(e)
    return out


S.PERMUTATIONS = 4000
skew = dict(big=3.0, pbig=0.12, small=-0.15)
seed = next(s for s in range(200)
            if S.wilcoxon(ceiling_d(s, **skew))[0] > 0.3
            and S.sign_flip(ceiling_d(s, **skew)) < 0.005)
rows_g = floor_rows([ceiling_d(seed, **skew), ceiling_d(seed + 1000),
                     ceiling_d(seed + 2000)])
verdict, floors, bad = A2A.floor_verdict(rows_g, "qwen")
moved = rows_g[0]
ok("a skewed floor shift the signed-rank test MISSES",
   moved["holm_rank"] >= 0.05, f"Holm rank {moved['holm_rank']:.3g}")
ok("is caught by the test on the mean",
   moved["holm_mean"] < 0.05, f"Holm mean {moved['holm_mean']:.3g}")
ok("and the verdict is CONTAMINATED, as it must be",
   verdict == "CONTAMINATED" and len(bad) == 1)

alarms = 0
null_seeds = 30
for s in range(null_seeds):
    alarms += A2A.floor_verdict(floor_rows(
        [ceiling_d(5000 + 10 * s + k) for k in range(3)]), "qwen")[0] \
        == "CONTAMINATED"
ok("three null ceiling floors: false alarm stays small with two tests OR-ed",
   alarms / null_seeds <= 0.15, f"{alarms} of {null_seeds}")


def social_row(d):
    e = A2A.estimate(d)
    e.update({"model": "qwen", "name": "P", "type": "stated_x_stated"})
    e["bh"] = e["p"]
    return e


cons = social_row(ceiling_d(11, shift=-0.5))
ok("a consistent -0.5 effect on the non-ceiling tasks is claimed",
   A2A.claimed(cons) and cons["mean"] < 0,
   f"mean {cons['mean']:+.3f}, HL {cons['hl']:+.3f}")
sk = social_row(ceiling_d(seed, **skew))
ok("a skewed effect that only the mean test sees is NOT claimed",
   not A2A.claimed(sk),
   f"p rank {sk['p_rank']:.3g}, p mean {sk['p_mean']:.3g}")
opp = {"bh": 0.001, "mean": +0.2, "hl": -0.05, "rank_dir": -1}
ok("mean and signed ranks leaning opposite ways is NOT claimed, however small p",
   not A2A.claimed(opp))
binary = social_row([0.0] * 500 + [1.0] * 70 + [-1.0] * 30)
ok("a real effect on mostly-zero binary differences IS claimed",
   A2A.claimed(binary) and binary["hl"] == 0,
   f"HL {binary['hl']:+.1f}: the first direction rule would have dropped it")
ok("rank_direction: more and larger positives lean positive",
   S.rank_direction([0, 0, 1, 2, -1]) == 1
   and S.rank_direction([0, 0, 0]) == 0
   and S.rank_direction([1, -1]) == 0)
fp = sum(A2A.claimed(social_row(ceiling_d(7000 + s))) for s in range(30))
ok("a null ceiling effect is claimed at most rarely", fp <= 2, f"{fp} of 30")

# ==========================================================================
print()
print("=" * 74)
print("H. FLOORS THAT CANCEL ACROSS STRATA, AND WHAT A SIGN MEANS")
print("=" * 74)
# Real qwen: floor 1 was +0.51 on hard tasks and -0.03 on easy ones, quiet in
# aggregate. Here: +0.6 on 300 hard tasks, -0.6 on 300 easy ones, net zero.
rng_h = random.Random(21)
easy = {f"E{i:03d}": -0.05 for i in range(300)}
hard = {f"H{i:03d}": -3.0 for i in range(300)}
cells_h = {}
for c in conds:
    cells_h[c["id"]] = {}
    for t, base_v in list(easy.items()) + list(hard.items()):
        v = base_v + rng_h.gauss(0, 0.1)
        if c["id"] == "ALT_BOTH":
            v += 0.6 if t.startswith("H") else -0.6
        cells_h[c["id"]][t] = v
agg = []
for ct in all_contrasts:
    e = A2A.estimate(A2A.interaction(cells_h, ct))
    e.update({"model": "qwen", "name": ct["name"], "type": ct["type"]})
    agg.append(e)
verdict_h, _f, _b = A2A.floor_verdict(agg, "qwen")
f1 = next(x for x in agg if x["name"] == "FLOOR_ALTxALT")
ok("a floor of +0.6 and -0.6 in two halves is quiet in aggregate",
   verdict_h == "CLEAN", f"mean {f1['mean']:+.3f}, Holm {f1['holm']:.3g}")
sizes = {k: len(v) for k, v in A2A.strata(cells_h).items()}
ok("the strata split the easy and the hard tasks",
   sizes["near ceiling"] == 300 and sizes["far from ceiling"] == 300,
   str(sizes))
srows, flagged = A2A.stratified_floors(cells_h, all_contrasts, "qwen")
ok("within strata the floor is caught, in both halves",
   len(flagged) == 2 and all(r["name"] == "FLOOR_ALTxALT" for r in flagged),
   f"{len(flagged)} flagged")
quiet_cells = {c: {t: v - (0.6 if (c == "ALT_BOTH" and t.startswith("H"))
                           else -0.6 if c == "ALT_BOTH" else 0.0)
                   for t, v in d.items()} for c, d in cells_h.items()}
ok("with the floor removed nothing is flagged in any stratum",
   not A2A.stratified_floors(quiet_cells, all_contrasts, "qwen")[1])

r_c = {"bh": 0.001, "hl": -0.3, "mean": -0.4, "rank_dir": -1}
r_p = {"bh": 0.001, "hl": +0.3, "mean": +0.4, "rank_dir": +1}
ok("two costs and a negative interaction: COMPOUNDS",
   A2A.direction_label(r_c, -0.4, -0.3).startswith("COMPOUNDS"))
ok("two benefits and a negative interaction: benefit does not add",
   A2A.direction_label(r_c, +0.4, +0.3) == "benefit does not add",
   "the qwen stated pairs")
ok("two costs and a positive interaction: cost does not add",
   A2A.direction_label(r_p, -0.4, -0.3) == "cost does not add",
   "the floors")
ok("opposite singles get no direction label",
   A2A.direction_label(r_c, -0.4, +0.3).startswith("singles point"))
ok("an unclaimed interaction gets no label at all",
   A2A.direction_label({"bh": 0.4, "hl": -0.3, "mean": -0.4,
                        "rank_dir": -1}, -1, -1) == "")
S.PERMUTATIONS = 10000

# ==========================================================================
print()
print("=" * 74)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("the floor extension and channel A2 are safe to run.")
print("=" * 74)
