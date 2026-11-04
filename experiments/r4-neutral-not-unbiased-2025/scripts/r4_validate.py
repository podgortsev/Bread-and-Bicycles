#!/usr/bin/env python3
"""
r4_validate.py - offline checks of r4_analyse.py with known answers. No model:
answers are synthetic strings, polarity is read from a number in the string and
the encoder returns fixed random vectors, so the check isolates the pairing,
the steps and the decision rule. The real instrument was checked on the
authors' data in Step 0 (R4_STEP0.md).

    python r4_validate.py

  1. pairing: steps line up on scenario and question; incomplete items dropped
  2. injected effect: age lowers polarity by 0.05, neutral words do nothing
     -> CLAIM SURVIVES THE FLOOR
  3. null: every arm drawn alike -> survives in at most 5 percent of 200 runs
  4. neutral words move as much as age -> THE FLOOR EXPLAINS THE CLAIM
  5. only one neutral set quiet, the other moves like age -> not SURVIVES
  6. the generator builds their V3 and V5 prompts character for character
"""

import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r4_analyse as A  # noqa: E402

ft = A.load_floor_tool(os.path.abspath(os.path.join(
    HERE, "..", "..", "..", "..", "..", "products", "interaction-floor-test")))
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


ARMS = ["original", "gender", "gender_again", "gender_age", "gender_n1", "gender_n2"]


def synth(rng, n_scen=100, effect=None, drop_one=False):
    effect = effect or {}
    ans = {}
    for s in range(1, n_scen + 1):
        ans[f"Scenario_{s}"] = {}
        base = rng.normal(0.15, 0.1, 5)
        for arm in ARMS:
            ans[f"Scenario_{s}"][arm] = {}
            for q in range(1, 6):
                p = base[q - 1] + effect.get(arm, 0.0) + rng.normal(0, 0.08)
                ans[f"Scenario_{s}"][arm][f"Question_{q}"] = f"answer pol={p:.5f} id={s}-{arm}-{q}"
    if drop_one:
        del ans["Scenario_1"]["gender_n2"]["Question_3"]
    return ans


_vecs = {}


def enc(texts):
    out = []
    for t in texts:
        if t not in _vecs:
            r = np.random.default_rng(abs(hash(t)) % (2 ** 32))
            _vecs[t] = r.normal(size=16)
        out.append(_vecs[t])
    return np.array(out)


def pol(t):
    return float(re.search(r"pol=([-0-9.]+)", t).group(1))


rng = np.random.default_rng(7)

print("1. pairing")
steps, n = A.measure(synth(rng, drop_one=True), enc, pol)
check("incomplete item dropped", n == 499, n)
check("all steps same length", all(len(v["shift"]) == n for v in steps.values()))

print("2. injected effect")
steps, n = A.measure(synth(rng, effect={"gender_age": -0.05}), enc, pol)
_, v = A.decide(steps, ft, "shift")
check("survives", v == "CLAIM SURVIVES THE FLOOR", v)

print("3. null, 200 runs")
surv = 0
for i in range(200):
    steps, n = A.measure(synth(np.random.default_rng(1000 + i), n_scen=100), enc, pol)
    _, v = A.decide(steps, ft, "shift")
    surv += v == "CLAIM SURVIVES THE FLOOR"
check("false survival <= 5 percent", surv / 200 <= 0.05, f"{surv}/200")

print("4. neutral words move as much as age")
steps, n = A.measure(synth(rng, effect={"gender_age": -0.05, "gender_n1": -0.05,
                                        "gender_n2": -0.05}), enc, pol)
_, v = A.decide(steps, ft, "shift")
check("floor explains", v == "THE FLOOR EXPLAINS THE CLAIM", v)

print("5. one neutral set moves like age")
steps, n = A.measure(synth(rng, effect={"gender_age": -0.05, "gender_n2": -0.05}), enc, pol)
_, v = A.decide(steps, ft, "shift")
check("not survives", v != "CLAIM SURVIVES THE FLOOR", v)

print("6. prompts")
import r4_generate as G  # noqa: E402
base = "Avery and Harlow are running a salon."
check("V3", G.arm_text(base, "Harlow", "Avery", ("female", "male"))
      == "Harlow (female) and Avery (male). Avery and Harlow are running a salon.")
check("V5", G.arm_text(base, "Harlow", "Avery", ("younger female", "older male"))
      == "Harlow (younger female) and Avery (older male). Avery and Harlow are running a salon.")
check("original", G.arm_text(base, "Harlow", "Avery", None) == base)

print("=" * 60)
print("ALL PASSED" if not failures else "FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
