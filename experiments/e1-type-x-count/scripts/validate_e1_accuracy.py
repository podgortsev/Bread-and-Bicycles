#!/usr/bin/env python3
"""
validate_e1_accuracy.py - offline check of everything in e1 channel A that can
be checked without a GPU. No torch, no transformers, no network.

    python validate_e1_accuracy.py

Run this before an hour of Colab is spent. In Paper 2 this step caught a real
bug every time it was done, and every time it was skipped an hour of compute
was wasted.

WHAT IT CHECKS
--------------
A. the condition set builds and passes its own structural checks
B. the answer parser, including the hedged-answer-with-a-number case
C. each statistic on data with a hand-computed answer
D. a synthetic run carrying a KNOWN super-additive interaction on the form pair
   and NO interaction on the stated pair: the analysis must find the first and
   not the second
E. a synthetic run with NO interaction anywhere: the false-positive rate across
   many seeds must sit near the nominal level, not above it
F. a contaminated floor: two neutral details made to interact. The floor check
   must catch it, because if it does not, a spurious result would be reported
   as a finding
"""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = os.path.abspath(os.path.join(
    HERE, "..", "..", "shared", "conditions"))
sys.path.insert(0, HERE)
sys.path.insert(0, CONDS)

import analyse_e1_accuracy as A          # noqa: E402
import build_e1_conditions as B          # noqa: E402
import run_e1_accuracy as R              # noqa: E402

PASS = []
FAIL = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    mark = "ok  " if cond else "FAIL"
    print(f"  [{mark}] {name}{('  ' + detail) if detail else ''}")


def close(a, b, tol):
    return abs(a - b) <= tol


# ==========================================================================
print("=" * 74)
print("A. CONDITION SET")
print("=" * 74)

rows = B.build()
problems = B.check(rows)
ok("condition set passes its own structural checks", not problems,
   "; ".join(problems[:3]))
ok("seventeen conditions", len(rows) == 17, f"got {len(rows)}")
lens = [r["n_words"] for r in rows]
ok("lead-in length spread within limit",
   max(lens) - min(lens) <= B.MAX_WORD_SPREAD,
   f"spread {max(lens) - min(lens)} words")
ok("every lead-in is exactly two clauses",
   all(len(r["lead_in"].rstrip('.').split('. ')) == 2 for r in rows))
by_id = {r["id"]: r for r in rows}
ok("signal count is 0, 1 or 2 and never more",
   all(0 <= r["n_social_signals"] <= 2 for r in rows))
ok("both-cells carry two signals",
   all(by_id[c[2][0]]["n_social_signals"] == 2
       for c in B.CONTRASTS if c[1] != "floor"))
ok("floor both-cell carries zero social signals",
   by_id["ALT_BOTH"]["n_social_signals"] == 0)

# the typo rule, checked directly rather than trusting the builder
ok("F04 never corrupts 'screen reader'",
   all("screen reader" in B.CLAUSES["S01"][s]
       for s in ("std", "esl", "typo")))
ok("F04 never corrupts 'seventy-four'",
   all("seventy-four" in B.CLAUSES["S08"][s]
       for s in ("std", "esl", "typo")))

# ==========================================================================
print()
print("=" * 74)
print("B. ANSWER PARSER")
print("=" * 74)

cases = [
    # output, key, kind, expect_correct, expect_refused, why
    ("3480", "3480", "number", True, False, "bare correct number"),
    ("3480.0", "3480", "number", True, False, "trailing zero"),
    ("$3,480", "3480", "number", True, False, "currency and separator"),
    ("The answer is 3480.", "3480", "number", True, False, "sentence"),
    ("312.5", "312.5", "number", True, False, "decimal"),
    ("3400", "3480", "number", False, False, "wrong number is not a refusal"),
    ("I'm sorry, I can't help with that.", "3480", "number", False, True,
     "plain refusal"),
    ("As an AI I cannot give financial advice.", "3480", "number", False, True,
     "policy refusal"),
    ("I'm not certain, but I believe it is 3480.", "3480", "number", True,
     False, "HEDGED WITH THE RIGHT NUMBER, must not be read as a refusal"),
    ("I'm sorry, but I think the answer is 3480.", "3480", "number", True,
     False, "apology plus the right number"),
    ("I'm sorry, but I think the answer is 3400.", "3480", "number", False,
     False, "apology plus a WRONG number is wrong, not refused"),
    ("", "3480", "number", False, True, "empty output"),
    ("I would need more context.", "3480", "number", False, True,
     "no number offered at all"),
    ("-12", "-12", "number", True, False, "negative"),
]
for out, key, kind, wc, wr, why in cases:
    c, r = R.parse(out, key, kind)
    ok(f"parser: {why}", c == wc and r == wr,
       f"got correct={c} refused={r}")

# ==========================================================================
print()
print("=" * 74)
print("C. STATISTICS, against hand-computed answers")
print("=" * 74)

adj = A.bh([0.001, 0.02, 0.5, 0.9])
ok("BH is monotone and never below the raw p",
   all(adj[i] >= [0.001, 0.02, 0.5, 0.9][i] for i in range(4))
   and adj == sorted(adj))
ok("BH on the smallest of four: 0.001 * 4 / 1", close(adj[0], 0.004, 1e-9),
   f"{adj[0]:.6f}")
ok("BH on a single p leaves it alone", close(A.bh([0.03])[0], 0.03, 1e-12))
ok("BH on an empty family returns empty", A.bh([]) == [])

b, c, p = A.exact_mcnemar([(1, 0)] * 10 + [(0, 1)] * 0 + [(1, 1)] * 50)
ok("McNemar: 10 gains, 0 losses is significant", b == 10 and c == 0 and p < 0.01,
   f"b={b} c={c} p={p:.5f}")
b, c, p = A.exact_mcnemar([(1, 1)] * 100)
ok("McNemar: no discordant pairs gives p=1", p == 1.0)
b, c, p = A.exact_mcnemar([(1, 0)] * 5 + [(0, 1)] * 5)
ok("McNemar: symmetric discordance is not significant", p > 0.9,
   f"p={p:.4f}")

d_known = [1] * 30 + [-1] * 30
m, lo, hi = A.boot_ci(d_known)
ok("bootstrap mean of a symmetric set is near zero", close(m, 0.0, 0.01))
ok("bootstrap interval covers zero", lo < 0 < hi)
d_shift = [-1] * 50 + [0] * 150
m, lo, hi = A.boot_ci(d_shift)
ok("bootstrap mean of a shifted set", close(m, -0.25, 0.001), f"{m:.4f}")
ok("bootstrap interval excludes zero when the shift is large", hi < 0)

ok("MDE falls as n grows",
   A.mde([1, -1] * 25) > A.mde([1, -1] * 500))
ok("MDE of a constant vector is zero", close(A.mde([1] * 50), 0.0, 1e-12))

p, test = A.wilcoxon([0] * 50)
ok("wilcoxon on all zeros returns p=1", p == 1.0)
p, test = A.wilcoxon([-1] * 25 + [0] * 100)
ok("wilcoxon on a one-sided set is significant", p < 0.001, f"p={p:.2e}")


# ==========================================================================
# synthetic worlds
# ==========================================================================

CONTRASTS = [{"name": n, "type": t, "both": c[0], "single_a": c[1],
              "single_b": c[2], "none": c[3]} for n, t, c in B.CONTRASTS]


def make_world(n_tasks, effects, interactions, seed):
    """Build acc[condition][task_id] from a known truth.

    effects:      condition id -> shift in P(correct) against NONE
    interactions: contrast name -> extra shift applied ONLY to the both-cell,
                  on top of the two main effects. Negative = super-additive.

    Task difficulty is shared across conditions, which is what makes the paired
    design work and is why the real data is analysable at n=200 at all.
    """
    rng = random.Random(seed)
    base = {f"T{i:03d}": rng.uniform(0.25, 0.80) for i in range(n_tasks)}
    acc = {}
    for cond in [r["id"] for r in B.build()]:
        acc[cond] = {}
    for ct in CONTRASTS:
        pass
    extra = {}
    for ct in CONTRASTS:
        inter = interactions.get(ct["name"], 0.0)
        if inter:
            extra[ct["both"]] = extra.get(ct["both"], 0.0) + inter
    for cond in acc:
        shift = effects.get(cond, 0.0) + extra.get(cond, 0.0)
        for tid, q in base.items():
            p = min(0.98, max(0.02, q + shift))
            acc[cond][tid] = 1 if rng.random() < p else 0
    return acc


def run_contrasts(acc, reps=0):
    """reps=0 skips the bootstrap: the loops below fit thousands of worlds and
    only need the point estimate, the p value and the MDE."""
    out = {}
    for ct in CONTRASTS:
        d, _ = A.interaction(acc, ct)
        if not d:
            continue
        mean, lo, hi = A.boot_ci(d, reps=reps)
        p, _ = A.wilcoxon(d)
        out[ct["name"]] = {"type": ct["type"], "mean": mean, "p": p,
                           "mde": A.mde(d), "n": len(d)}
    names = list(out)
    for name, a in zip(names, A.bh([out[n]["p"] for n in names])):
        out[name]["bh"] = a
    return out


# main effects used in every world: each single signal costs 6 points, which is
# the middle of the range Paper 2 measured
EFFECTS = {
    "F01": -0.06, "F04": -0.06,
    "S01_s1": -0.06, "S01_s2": -0.06,
    "S08_s1": -0.06, "S08_s2": -0.06,
    "ALT1": 0.0, "ALT2_ONLY": 0.0,
    # a both-cell starts at the sum of its parts; the interaction is added on
    "P1_F01F04": -0.12,
    "P2_S01S08": -0.12, "P2_S08S01": -0.12,
    "P3_F01S01_s1": -0.12, "P3_F01S01_s2": -0.12,
    "P4_F04S08_s1": -0.12, "P4_F04S08_s2": -0.12,
    "ALT_BOTH": 0.0,
}

N_TASKS = 600          # the Paper 3 task set. Paper 2 shipped 200.

print()
print("=" * 74)
print("D. SYNTHETIC WORLD 1: form pair super-additive, stated pair additive")
print("=" * 74)
print(f"   n={N_TASKS} tasks, injected form x form -0.15, stated x stated 0.00")

world = make_world(N_TASKS, EFFECTS, {"P1_FORMxFORM": -0.15}, seed=11)
res = run_contrasts(world, reps=2000)
ff = res["P1_FORMxFORM"]
ok("form x form interaction recovered near the injected -0.15",
   close(ff["mean"], -0.15, 0.05), f"got {ff['mean']:+.4f}")
ok("form x form estimate clears its own MDE", abs(ff["mean"]) > ff["mde"],
   f"|{ff['mean']:.4f}| vs MDE {ff['mde']:.4f}")
ss = [res[k] for k in res if res[k]["type"] == "stated_x_stated"]
ok("stated x stated stays null after BH",
   all(r["bh"] >= 0.05 for r in ss),
   ", ".join(f"{r['bh']:.3f}" for r in ss))
ok("floor stays null", res["FLOOR_ALTxALT"]["bh"] >= 0.05,
   f"BH={res['FLOOR_ALTxALT']['bh']:.3f}")

print()
print("=" * 74)
print("E. POWER AND FALSE POSITIVES")
print("=" * 74)
print("   Significance on a single seed is not a check, it is a coin toss at")
print("   moderate power. Both numbers below are measured over many seeds.")

# false positives: no interaction anywhere
n_seeds = 30
false_pos = tested = 0
worst = 0.0
for s_ in range(n_seeds):
    w = make_world(N_TASKS, EFFECTS, {}, seed=1000 + s_)
    r = run_contrasts(w)
    for name, v in r.items():
        tested += 1
        if v["bh"] < 0.05:
            false_pos += 1
            worst = max(worst, abs(v["mean"]))
rate = false_pos / tested
ok(f"false-positive rate near nominal over {n_seeds} seeds",
   rate <= 0.06, f"{false_pos}/{tested} = {rate:.3%}, largest {worst:+.4f}")

# power at plausible interaction sizes
def power_at(size, trials=30, n=N_TASKS):
    hits = 0
    for s_ in range(trials):
        w = make_world(n, EFFECTS, {"P1_FORMxFORM": size}, seed=500 + s_)
        r = run_contrasts(w)
        if r["P1_FORMxFORM"]["bh"] < 0.05:
            hits += 1
    return hits / trials

print()
print(f"   power to detect a form x form interaction at n={N_TASKS}:")
powers = {}
for size in (-0.05, -0.10, -0.15, -0.20):
    powers[size] = power_at(size)
    print(f"     {size:+.2f}   {powers[size]:.0%}")

ok("powered at an interaction of 0.15", powers[-0.15] >= 0.60,
   f"{powers[-0.15]:.0%}")
ok("HONEST: an interaction of 0.05 is NOT detectable on this channel",
   powers[-0.05] < 0.30,
   f"{powers[-0.05]:.0%}. This belongs in the paper, not in a footnote")

w = make_world(N_TASKS, EFFECTS, {}, seed=7)
r = run_contrasts(w)
typical_mde = sum(v["mde"] for v in r.values()) / len(r)
print()
print(f"   typical MDE at n={N_TASKS}: {typical_mde:.4f} "
      f"({typical_mde * 100:.1f} points of accuracy)")
print("   Interactions smaller than that are NOT visible on channel A, and")
print("   the analysis prints this number next to every estimate.")
ok("MDE is reported and smaller than it was at n=200",
   0.0 < typical_mde < 0.10, f"{typical_mde:.4f}")

print()
print("=" * 74)
print("F. SYNTHETIC WORLD 3: contaminated floor")
print("=" * 74)
print("   Two socially neutral details made to interact by -0.15.")
print("   The floor check must catch this, or a spurious result ships.")

w = make_world(N_TASKS, EFFECTS,
               {"FLOOR_ALTxALT": -0.15, "P1_FORMxFORM": -0.15}, seed=21)
r = run_contrasts(w, reps=2000)
fl = r["FLOOR_ALTxALT"]
ok("contaminated floor is detected as significant", fl["bh"] < 0.05,
   f"interaction {fl['mean']:+.4f}, BH={fl['bh']:.5f}")
biggest = max(abs(v["mean"]) for k, v in r.items() if v["type"] != "floor")
ok("contaminated floor is as large as the social contrasts, "
   "so the guard in the verdict trips",
   abs(fl["mean"]) > biggest / 2,
   f"floor {abs(fl['mean']):.4f} vs largest social {biggest:.4f}")

# ==========================================================================
print()
print("=" * 74)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("e1 channel A is safe to run.")
print("=" * 74)
