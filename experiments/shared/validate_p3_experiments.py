#!/usr/bin/env python3
"""
validate_p3_experiments.py - offline check of e1w, e2, e2b, e3 and e5.
No GPU, no torch, no network.

    python validate_p3_experiments.py

Experiment e1 has two dedicated validators of its own, `validate_e1_accuracy.py`
and `validate_e1_channels.py`, because it carries the paper's claim and deserves
the space. The other five are checked here together, in one file, for one
reason: a validator that is tedious to run does not get run, and five separate
scripts is five things to remember.

WHAT IS CHECKED, AND WHY THESE THINGS
-------------------------------------
Only what can actually go wrong without a model.

  parsers          Every one of them, on cases with a known answer. Paper 2 lost
                   real data twice to a parser: once by checking for refusal
                   before looking for a number, once by dropping "7." because of
                   the full stop after the digit.
  condition sets   Every builder's own structural checks, plus the invariants
                   that matter across experiments: clause count constant, signal
                   words surviving, contrasts slot-matched.
  job construction The number of prompts, and that the text actually reaches
                   them. An experiment that silently runs 5,400 identical
                   prompts costs a session and produces nothing.
  analysis maths   On synthetic data with a known answer, including a null that
                   must NOT be detected. This is the part that has caught a real
                   bug every time it was done in Paper 2.
"""

import csv
import io as _io
import json
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _add(*parts):
    sys.path.insert(0, os.path.abspath(os.path.join(ROOT, *parts)))


_add("shared")
_add("shared", "conditions")
_add("e1-type-x-count", "scripts")
_add("e1w-wrappers", "scripts")
_add("e2-depth-curve", "scripts")
_add("e2b-legibility", "scripts")
_add("e3-offsetting", "scripts")
_add("e5-attribution", "scripts")

import p3_stats as S                    # noqa: E402
import build_e1_conditions as B1        # noqa: E402
import build_e2_conditions as B2        # noqa: E402
import build_e3_conditions as B3        # noqa: E402
import run_e1w_wrappers as W            # noqa: E402
import analyse_e1w_wrappers as WA       # noqa: E402
import analyse_e2_depth as D2A          # noqa: E402
import run_e3_numeric as E3N            # noqa: E402
import run_e5_attribution as E5         # noqa: E402

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          f"{('  ' + detail) if detail else ''}")


def close(a, b, tol):
    return abs(a - b) <= tol


# ==========================================================================
print("=" * 74)
print("e1w  WRAPPERS")
print("=" * 74)

ok("six wrappers, all distinct", len(set(W.WRAPPERS.values())) == 6)
ok("nine conditions cover the cells the headline needs",
   set(W.CONDITIONS) >= {"NONE", "F01", "F04", "P1_F01F04"},
   f"{len(W.CONDITIONS)} conditions")

e1rows = B1.build()
by_id = {r["id"]: r for r in e1rows}
ok("every e1w condition exists in the e1 set",
   all(c in by_id for c in W.CONDITIONS))

fake_tasks = [{"id": f"T{i:03d}", "question": f"What is {i} plus 1?",
               "answer": str(i + 1), "answer_kind": "number",
               "family": "x"} for i in range(1, 21)]
jobs = W.build_jobs([by_id[c] for c in W.CONDITIONS], fake_tasks)
ok("e1w job count is wrappers x conditions x tasks",
   len(jobs) == 6 * len(W.CONDITIONS) * len(fake_tasks),
   f"{len(jobs)}")
ok("every prompt carries its wrapper text",
   all(W.WRAPPERS[j[0]] in j[6] for j in jobs[:200]))
ok("every prompt carries its lead-in",
   all(by_id[j[1]]["lead_in"] in j[6] for j in jobs[:200]))
ok("prompts are not all identical",
   len({j[6] for j in jobs}) == len(jobs))


def wrapper_world(per_wrapper_interaction, seed, n=100):
    """acc[wrapper][condition][task] with a known interaction per wrapper."""
    rng = random.Random(seed)
    base = {f"T{i:03d}": rng.uniform(0.3, 0.75) for i in range(n)}
    eff = {"F01": -0.06, "F04": -0.06, "P1_F01F04": -0.12,
           "S01_s1": -0.06, "S08_s2": -0.06, "P2_S01S08": -0.12,
           "P3_F01S01_s1": -0.12, "P4_F04S08_s1": -0.12}
    out = {}
    for wid, inter in per_wrapper_interaction.items():
        cells = {}
        for cid in W.CONDITIONS:
            shift = eff.get(cid, 0.0) + (inter if cid == "P1_F01F04" else 0.0)
            cells[cid] = {t: (1 if rng.random() < min(.97, max(.03, q + shift))
                              else 0) for t, q in base.items()}
        out[wid] = cells
    return out


ct_p1 = {"name": "P1_FORMxFORM", "type": "form_x_form", "both": "P1_F01F04",
         "single_a": "F01", "single_b": "F04", "none": "NONE"}

w = wrapper_world({f"W{i}": -0.15 for i in range(1, 7)}, seed=4)
per = [sum(WA.interaction(w[k], ct_p1)) / len(WA.interaction(w[k], ct_p1))
       for k in w]
mean = sum(per) / len(per)
spread = max(per) - min(per)
ok("a consistent interaction is recovered across all six wrappers",
   close(mean, -0.15, 0.06) and abs(mean) > spread / 2,
   f"mean {mean:+.4f}, spread {spread:.4f}")
ok("all six wrappers agree in sign", all(v < 0 for v in per))

w = wrapper_world({"W1": -0.30, "W2": -0.25, "W3": +0.25, "W4": +0.30,
                   "W5": -0.05, "W6": +0.05}, seed=5)
per = [sum(WA.interaction(w[k], ct_p1)) / len(WA.interaction(w[k], ct_p1))
       for k in w]
mean = sum(per) / len(per)
spread = max(per) - min(per)
pos = sum(1 for v in per if v > 0)
ok("a wrapper-DEPENDENT effect is caught: signs disagree",
   min(pos, 6 - pos) >= 2, f"{pos} of 6 positive")
ok("a wrapper-dependent effect does not clear its own spread",
   abs(mean) <= spread / 2,
   f"mean {mean:+.4f} against spread {spread:.4f}. Paper 2's method 3b saw "
   f"exactly this, r=0.13 between two wrappers")

# ==========================================================================
print()
print("=" * 74)
print("e2  DEPTH CURVE")
print("=" * 74)

rows2 = B2.build()
ok("e2 conditions pass their own structural checks", not B2.check(rows2),
   "; ".join(B2.check(rows2)[:2]))
ok("seventeen conditions", len(rows2) == 17, f"{len(rows2)}")
ok("every lead-in is exactly three clauses",
   all(r["n_clauses"] == 3 for r in rows2))
ok("depth always equals the number of signals",
   all(r["depth"] == r["n_social_signals"] for r in rows2))
ok("both ladders reach depth 3",
   {r["ladder"] for r in rows2 if r["depth"] == 3} == {"form", "stated"})
ok("the form ladder has ONE route at depth 3, by construction",
   sum(1 for r in rows2 if r["ladder"] == "form" and r["depth"] == 3) == 1,
   "a style has no order, so there is nothing to rotate")
ok("the stated ladder has three routes at depth 3",
   sum(1 for r in rows2 if r["ladder"] == "stated" and r["depth"] == 3) == 3)
ok("no two form conditions produced the same text",
   len({r["lead_in"] for r in rows2 if r["ladder"] == "form"})
   == sum(1 for r in rows2 if r["ladder"] == "form"))
ok("typos never removed a stated signal's words",
   all(all(w in r["lead_in"] for s in r["stated_signals"]
           for w in B2.STATED[s]["signal_words"]) for r in rows2))

# the shape maths, on curves with a known answer
xs = [0, 1, 2, 3]
lin = [0.0, -0.05, -0.10, -0.15]
acc_curve = [0.0, -0.03, -0.09, -0.20]
sat = [0.0, -0.08, -0.12, -0.13]
sse_l, sse_q, curve = D2A.fit_quadratic(xs, lin)
ok("a straight line has a quadratic term of zero", close(curve, 0.0, 1e-9),
   f"{curve:+.6f}")
ok("a straight line is fitted exactly by a line", close(sse_l, 0.0, 1e-12))
_, _, curve = D2A.fit_quadratic(xs, acc_curve)
ok("a downward-bending curve gives a negative quadratic term", curve < 0,
   f"{curve:+.5f}")
_, _, curve = D2A.fit_quadratic(xs, sat)
ok("a saturating curve gives a positive quadratic term", curve > 0,
   f"{curve:+.5f}")

second = lambda y: (y[3] - y[2]) - (y[2] - y[1])
ok("second difference is zero on a straight line",
   close(second(lin), 0.0, 1e-12))
ok("second difference is negative on a compounding curve",
   second(acc_curve) < 0, f"{second(acc_curve):+.4f}")
ok("second difference is positive on a saturating curve",
   second(sat) > 0, f"{second(sat):+.4f}")

# ==========================================================================
print()
print("=" * 74)
print("e2b  LEGIBILITY")
print("=" * 74)

sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "e2b-legibility",
                                                "scripts")))
import build_e2b_tasks as B2B          # noqa: E402

samples = [
    ("T001", "A price of 2400 dollars increases by 45 percent. "
             "What is the new price?"),
    ("T002", "What is 25 percent of 1250?"),
    ("T003", "How many metres are there in 12 kilometres?"),
    ("T004", "Quarterly units sold. North: 300; South: 120. "
             "What is the difference between the highest and the lowest region?"),
]
for tid, q in samples:
    for arm in ("esl", "typo", "esl_typo"):
        rw = B2B.rewrite(q, tid, arm)
        ok(f"e2b {arm}: numbers preserved in {tid}",
           B2B.numbers(rw) == B2B.numbers(q),
           f"{sorted(B2B.numbers(q).elements())} -> "
           f"{sorted(B2B.numbers(rw).elements())}")
ok("e2b never mangles a meaning-carrying word",
   all(w in B2B.rewrite(samples[3][1], "T004", "esl_typo")
       for w in ("highest", "lowest", "difference")),
   "a mangled superlative could change what the question asks")
ok("e2b rewriting is deterministic",
   B2B.rewrite(samples[0][1], "T001", "typo")
   == B2B.rewrite(samples[0][1], "T001", "typo"))
ok("e2b esl actually changes the text",
   all(B2B.rewrite(q, t, "esl") != q for t, q in samples))

# the arm A against arm B comparison
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "e2b-legibility",
                                                "scripts")))
import analyse_e2b_legibility as E2BA  # noqa: E402


def arm_world(inter, seed, n=300):
    rng = random.Random(seed)
    base = {f"T{i:03d}": rng.uniform(0.3, 0.75) for i in range(n)}
    eff = {"std": 0.0, "esl": -0.06, "typo": -0.06, "esl_typo": -0.12}
    cells = {}
    for cell, shift in eff.items():
        s = shift + (inter if cell == "esl_typo" else 0.0)
        cells[cell] = {t: (1 if rng.random() < min(.97, max(.03, q + s)) else 0)
                       for t, q in base.items()}
    return cells


a = arm_world(0.0, seed=11)
b = arm_world(-0.18, seed=11)
da, _ = E2BA.interaction(a)
db, _ = E2BA.interaction(b)
ok("arm A recovers no interaction when none was injected",
   abs(sum(da) / len(da)) < S.mde(da), f"{sum(da) / len(da):+.4f}")
ok("arm B recovers the injected -0.18",
   close(sum(db) / len(db), -0.18, 0.07), f"{sum(db) / len(db):+.4f}")
diff = [y - x for x, y in zip(da, db)]
ok("the legibility component is detected as the difference of the arms",
   abs(sum(diff) / len(diff)) > S.mde(diff),
   f"{sum(diff) / len(diff):+.4f} against MDE {S.mde(diff):.4f}")

# ==========================================================================
print()
print("=" * 74)
print("e3  OFFSETTING")
print("=" * 74)

rows3 = B3.build()
ok("e3 conditions pass their own structural checks", not B3.check(rows3),
   "; ".join(B3.check(rows3)[:2]))
ok("eleven conditions", len(rows3) == 11, f"{len(rows3)}")
ok("both credential-plus-disclosure pairs run in both orders",
   sum(1 for r in rows3 if r["has_credential"] and r["disclosures"]) == 4)
ok("the credential clause never carries a disclosure",
   all(not r["disclosures"] for r in rows3
       if r["has_credential"] and r["n_social_signals"] == 1))

num_cases = [
    ("85000", 85000.0, False, "bare number"),
    ("$85,000", 85000.0, False, "currency and separator"),
    ("A fair salary would be 85000 dollars.", 85000.0, False, "sentence"),
    ("7.5", 7.5, False, "decimal"),
    ("I'm sorry, I can't estimate that.", None, True, "refusal"),
    ("As an AI I cannot assign a salary.", None, True, "policy refusal"),
    ("I would say around 72, though it depends.", 72.0, False,
     "HEDGED WITH A NUMBER, must not be read as a refusal"),
    ("", None, True, "empty"),
]
for out, want_v, want_r, why in num_cases:
    v, r = E3N.parse_number(out)
    ok(f"e3 numeric parser: {why}",
       (v == want_v) and (r == want_r), f"got {v}, refused={r}")

ok("e3 excludes the EXPERIENCE measure",
   "EXPERIENCE" not in {m[0] for m in E3N.MEASURES},
   "the credential prints 'fifteen years', so asking for years measures copying")
ok("e3 keeps a matched opposite-valence pair for the coherence check",
   {"SUCCESS", "CHURN"} <= {m[0] for m in E3N.MEASURES})
ok("e3 numbers_in finds every number in a prompt",
   E3N.numbers_in("led for fifteen years, 5 years in engineering, 1,200 staff")
   == {"5", "1200"},
   "spelled-out numbers are not caught, which is why the credential says "
   "'fifteen' in words")

# restoration maths
none_v = {f"P{i:03d}": 70.0 for i in range(50)}
alone_v = {k: v - 10.0 for k, v in none_v.items()}
full_v = {k: v for k, v in none_v.items()}
half_v = {k: v - 5.0 for k, v in none_v.items()}
gap_full = [full_v[k] - none_v[k] for k in none_v]
gap_half = [half_v[k] - none_v[k] for k in none_v]
harm = [alone_v[k] - none_v[k] for k in none_v]
h = sum(harm) / len(harm)
ok("full restoration reads as recovered = 1.00",
   close(1 - (sum(gap_full) / len(gap_full)) / h, 1.0, 1e-9))
ok("half restoration reads as recovered = 0.50",
   close(1 - (sum(gap_half) / len(gap_half)) / h, 0.5, 1e-9))
flat = [0.0] * 50
ok("with no harm to recover from, the denominator is refused",
   not (abs(sum(flat) / len(flat)) > S.mde(flat)),
   "dividing by noise produces spectacular ratios that mean nothing")

# ==========================================================================
print()
print("=" * 74)
print("e5  ATTRIBUTION")
print("=" * 74)

two_cases = [
    ("7, 3", (7.0, 3.0), "comma"),
    ("7,3", (7.0, 3.0), "no space"),
    ("(a) 8 (b) 2", (8.0, 2.0), "labelled"),
    ("8/10, 2/10", (8.0, 2.0), "out of ten, must not read 10 as a rating"),
    ("I would say 6 and 4.", (6.0, 4.0), "sentence"),
    ("0, 0", (0.0, 0.0), "both zero"),
    ("10, 0", (10.0, 0.0), "extremes"),
    ("It did not change my answer.", (None, None), "no numbers"),
    ("5", (None, None), "only one number is not an answer"),
]
for out, want, why in two_cases:
    got = E5.parse_two(out)
    ok(f"e5 parser: {why}", got == want, f"got {got}")

ok("e5 has a descriptor for every signal it can be asked about",
   {"S01", "S08", "F01", "F04"} <= set(E5.DESCRIPTOR))
ok("e5 never claims the person 'mentioned' a form signal",
   "mention" not in E5.QUESTION.lower(),
   "nobody mentions their own typos; the question asks about the message")

sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "e5-attribution",
                                                "scripts")))
import analyse_e5_attribution as E5A   # noqa: E402

ok("e5 knows which e1 cell measures each signal in each pair",
   E5A.SINGLE_FOR[("P2_S01S08", "S08")] == "S08_s2"
   and E5A.SINGLE_FOR[("P2_S08S01", "S08")] == "S08_s1",
   "slot-matched, or the position term would not cancel")

# degeneracy detection
const = [6.0] * 120
_vr = random.Random(3)
varied = [_vr.choice([0, 2, 5, 7, 9]) for _ in range(120)]


def sd(v):
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


ok("a constant self-report is detected as degenerate",
   sd(const) < E5A.SD_FLOOR,
   "Llama answered '6' to 57 of 60 in method 7, sd 0.45")
ok("a varied self-report is not flagged", sd(varied) >= E5A.SD_FLOOR,
   f"sd {sd(varied):.2f}")

# ==========================================================================
print()
print("=" * 74)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("e1w, e2, e2b, e3 and e5 are safe to run.")
print("=" * 74)
