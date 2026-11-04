#!/usr/bin/env python3
"""
r2_validate.py - offline checks for R2, no GPU, no torch.

    python r2_validate.py

A. their disparity metric, and why the floor must match the number of markers
B. the fidelity statistics and the preregistered format choice
C. the floor verdict: a real excess is called above the floor, a matched one is
   not
D. the doubly-disadvantaged rule: attributed only when neutral markers do not
   show it as strongly
E. the scoring loop with a fake model: every row written once, a rerun writes
   nothing, referent and non-referent kept apart
"""

import math
import os
import random
import shutil
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_analyse as A          # noqa: E402
import r2_build_prompts as B    # noqa: E402
import r2_score as S            # noqa: E402

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          f"{('  ' + detail) if detail else ''}")


def lp(p):
    return math.log(max(p, 1e-12))


def make_rows(markers, n_sent=40, seed=1):
    rows = []
    rid = 0
    for sid in range(n_sent):
        for gender, pron in (("masc", "he"), ("fem", "she")):
            base = f"The worker{sid} helped the clerk{sid} because {pron} was kind."
            for t in ("type1", "type2"):
                for m in markers:
                    rows.append({"row_id": str(rid), "kind": "x",
                                 "attribute": "a", "marker": m,
                                 "sentence_id": f"{sid}-{pron}-{t}",
                                 "gender": gender, "pronoun": pron,
                                 "task_type": t, "stereotype": "pro-stereotypical",
                                 "referent": "worker", "nonreferent": "clerk",
                                 "referent_dominance": "male" if sid % 2 else "female",
                                 "sentence": base})
                    rid += 1
    return rows


print("=" * 74)
print("A. THEIR DISPARITY METRIC")
print("=" * 74)
rows = make_rows(["m1", "m2"])
scores = {}
target = {("masc", "m1"): 0.50, ("fem", "m1"): 0.30,
          ("masc", "m2"): 0.45, ("fem", "m2"): 0.20}
for r in rows:
    p_ref = target[(r["gender"], r["marker"])] + 0.3
    scores[r["row_id"]] = (lp(p_ref), lp(0.3))
d = A.disparity(rows, scores, {"m1", "m2"}, "type1")
ok("max minus min of mean CC over gender x marker", abs(d - 0.30) < 1e-9,
   f"{d:.3f}")

rng = random.Random(7)
ranges = {}
for k in (2, 4, 8):
    vals = []
    for _ in range(2000):
        means = [rng.gauss(0, 0.05) for _ in range(k)]
        vals.append(max(means) - min(means))
    ranges[k] = statistics.mean(vals)
ok("with pure noise the range grows with the number of subgroups",
   ranges[2] < ranges[4] < ranges[8],
   ", ".join(f"{k} subgroups {v:.3f}" for k, v in ranges.items()))
ok("so an unmatched floor would be unfair to the attribute or to the floor",
   ranges[8] > 1.5 * ranges[2])

print()
print("=" * 74)
print("B. FIDELITY AND THE FORMAT CHOICE")
print("=" * 74)
ok("spearman of a perfect order is 1", abs(S.spearman([1, 2, 3], [4, 5, 9]) - 1) < 1e-12)
ok("spearman of a reversed order is -1",
   abs(S.spearman([1, 2, 3], [9, 5, 4]) + 1) < 1e-12)
exact = {a: v for a, v in S.PUBLISHED.items()}
f_exact = S.fidelity(exact)
ok("our table equal to theirs passes", f_exact["pass"] and f_exact["mad"] == 0)
shuffled = {a: (v[1] + 0.3, v[0] + 0.3) for a, v in S.PUBLISHED.items()}
f_bad = S.fidelity(shuffled)
ok("a table off by 0.3 everywhere fails on mean absolute difference",
   not f_bad["pass"], f"mad {f_bad['mad']:.3f}")
ok("a table with a missing attribute fails", not S.fidelity({"none": (0.1, 0.1)})["pass"])
fid = {"raw": {"spearman": 0.8, "mad": 0.05, "pass": True},
       "chat": {"spearman": 0.9, "mad": 0.04, "pass": True}}
ok("both pass: the higher spearman is chosen", S.choose_format(fid) == ("chat", True))
fid = {"raw": {"spearman": 0.7, "mad": 0.05, "pass": True},
       "chat": {"spearman": 0.9, "mad": 0.20, "pass": False}}
ok("only one passes: that one is chosen, however good the other's spearman",
   S.choose_format(fid) == ("raw", True))
fid = {"raw": {"spearman": 0.3, "mad": 0.2, "pass": False},
       "chat": {"spearman": 0.5, "mad": 0.2, "pass": False}}
ok("none passes: chosen, and marked as not passing",
   S.choose_format(fid) == ("chat", False))

print()
print("=" * 74)
print("C. THE FLOOR VERDICT")
print("=" * 74)
table_demo = {a: (0.10, 0.10) for a in B.ATTRIBUTES}
table_demo["socioeconomic"] = (0.40, 0.05)
table_neutral = {s: (0.12, 0.12) for s in B.NEUTRAL_SETS}
v = {(x["attribute"], x["type"]): x for x in A.floor_verdicts(table_demo, table_neutral)}
ok("an attribute well above every matched neutral set is above the floor",
   v[("socioeconomic", "type1")]["verdict"] == "above floor")
ok("the same attribute below the neutral max on the other type is not",
   v[("socioeconomic", "type2")]["verdict"] == "not distinguishable")
ok("an attribute equal to the neutral level is not distinguishable",
   v[("age", "type1")]["verdict"] == "not distinguishable")
ok("race is compared with the four-marker set, religion with the three",
   set(v[("race", "type1")]["neutral_sets"]) == {"N4"}
   and set(v[("religion", "type1")]["neutral_sets"]) == {"N3"})
ok("two-marker attributes are compared with all three two-marker sets",
   set(v[("age", "type1")]["neutral_sets"]) == {"N2a", "N2b", "N2c"})
ok("twenty cells", len(v) == 20)

print()
print("=" * 74)
print("D. THE DOUBLY-DISADVANTAGED RULE")
print("=" * 74)
rows = make_rows(["x", "n"], n_sent=60)
none_by_sid = {r["sentence_id"]: r["sentence"] for r in rows}
scores = {}
for r in rows:
    p = 0.6
    if r["marker"] == "x" and r["gender"] == "fem" and r["referent_dominance"] == "male":
        p -= 0.2
    scores[r["row_id"]] = (lp(p), lp(0.2))
dx, nx = A.feminine_disadvantage(rows, scores, "x", none_by_sid)
dn, _ = A.feminine_disadvantage(rows, scores, "n", none_by_sid)
ok("an injected feminine loss of 0.2 on male-dominated referents is recovered",
   abs(dx + 0.2) < 1e-6, f"D {dx:+.3f} over {nx} pairs")
ok("a marker without it reads 0", abs(dn) < 1e-9)
fd = {m: 0.0 for ms in B.ATTRIBUTES.values() for m in ms}
fd.update({m: 0.0 for ms in B.NEUTRAL_SETS.values() for m in ms})
fd["poor"] = -0.2
res = {x["attribute"]: x for x in A.dd_verdicts(fd)}
ok("a loss only on the demographic marker is attributed to the identity",
   res["socioeconomic"]["verdict"] == "attributed to identity")
fd["dog-owning"] = -0.25
res = {x["attribute"]: x for x in A.dd_verdicts(fd)}
ok("the same loss with a neutral marker losing as much is not",
   res["socioeconomic"]["verdict"] != "attributed to identity")

print()
print("=" * 74)
print("E. THE SCORING LOOP, WITH A FAKE MODEL")
print("=" * 74)


class FakeScorer:
    def prompt_text(self, r, fmt):
        return f"{r['sentence']} The pronoun '{r['pronoun']}' refers to the", True

    def sequence(self, prompt, add_special, occ):
        ids = [ord(c) % 97 for c in prompt] + [ord(c) % 97 for c in " " + occ]
        return ids, len(prompt)

    def score_group(self, seqs):
        return [-(len(s[0]) - s[1]) / 10.0 for s in seqs]

    def self_check(self, seqs):
        pass


rows = make_rows(["m1"], n_sent=5)
tmp = tempfile.mkdtemp()
try:
    path = os.path.join(tmp, "scores.csv")
    S.score_rows(FakeScorer(), rows, "raw", path)
    first = S.load_scores(path)
    ok("every row written once", len(first) == len(rows))
    S.score_rows(FakeScorer(), rows, "raw", path)
    with open(path, encoding="utf-8") as f:
        n_lines = sum(1 for _ in f) - 1
    ok("a rerun writes nothing new", n_lines == len(rows))
    r0 = rows[0]
    exp_ref = -(len(" " + r0["referent"])) / 10.0
    exp_non = -(len(" " + r0["nonreferent"])) / 10.0
    got = first[r0["row_id"]]
    ok("referent and non-referent scores are not swapped",
       abs(got[0] - exp_ref) < 1e-6 and abs(got[1] - exp_non) < 1e-6,
       f"{got}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
print("=" * 74)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("R2 analysis and scoring loop are safe to run.")
print("=" * 74)
