#!/usr/bin/env python3
"""
r1_validate_addendum4.py - offline checks for the Addendum 4 additions to
r1_analyse.py, with known answers. No model, no NER, no Kirk regressions: the
regression statistic is replaced by a synthetic one.

    python r1_validate_addendum4.py

Checks
  1. pool sizes are 252, 70 and 15, as registered
  2. relabelling maps descriptors onto labels, including "lesbian woman"
  3. centred interval: on a statistic whose resamples sit above the point
     estimate (the bias found on their data), the unshifted interval misses the
     point and the centred one contains it; with unbiased resamples the shift
     is near zero
  4. randomisation p with an injected effect: the real group is the most
     extreme of 252, so p = 1/252
  5. randomisation p under a null: over many simulated pools, p is roughly
     uniform (share below 0.05 near 0.05, not near 0 or 1)
  6. the pool statistic runs through relabelled_subset: an effect carried by
     the social descriptors' templates is recovered after relabelling
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r1_analyse as A  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


print("1. pool sizes")
check("size 5", len(A.pool_groups(5)) == 252, len(A.pool_groups(5)))
check("size 4", len(A.pool_groups(4)) == 70, len(A.pool_groups(4)))
check("size 2", len(A.pool_groups(2)) == 15, len(A.pool_groups(2)))
check("size 2 pool", A.POOLS[2][1] == ["gay", "straight", "conservative",
                                       "liberal", "hungry", "sleepy"])

print("2. relabelling")
ner = pd.DataFrame({"Name": ["man", "woman", "hungry man", "hungry woman",
                             "sleepy man", "sleepy woman", "gay man",
                             "lesbian woman", "Muslim man"],
                    "Title": ["[]"] * 9})
sub = A.relabelled_subset(ner, ["hungry", "sleepy"], ["gay", "straight"])
check("names", sorted(sub["Name"]) == sorted(
    ["man", "woman", "gay man", "lesbian woman", "straight man",
     "straight woman"]), sorted(sub["Name"]))
sub2 = A.relabelled_subset(ner, ["gay", "hungry"], ["gay", "straight"])
check("gay keeps lesbian", sorted(sub2["Name"]) == sorted(
    ["man", "woman", "gay man", "lesbian woman", "straight man",
     "straight woman"]), sorted(sub2["Name"]))
check("unrelated template dropped", "Muslim man" not in set(sub["Name"]))

print("3. centred interval")
rng = np.random.default_rng(1)
point = 0.203
biased = list(point + 0.015 + rng.normal(0, 0.004, 100))
ulo, uhi = A.interval(biased)
clo, chi = A.centred_interval(biased, point)
check("unshifted misses the point", not (ulo <= point <= uhi),
      f"[{ulo:.3f}, {uhi:.3f}]")
check("centred contains the point", clo <= point <= chi,
      f"[{clo:.3f}, {chi:.3f}]")
check("width unchanged", abs((chi - clo) - (uhi - ulo)) < 1e-12)
unbiased = list(point + rng.normal(0, 0.004, 100))
clo2, chi2 = A.centred_interval(unbiased, point)
ulo2, uhi2 = A.interval(unbiased)
check("no bias, shift small", abs(clo2 - ulo2) < 0.002,
      f"shift {clo2 - ulo2:+.4f}")

print("4. randomisation p, injected effect")
groups = A.pool_groups(5)
real = tuple(A.SOCIAL["religion"])
effect = {d: (0.30 if d in A.SOCIAL["religion"] else 0.10) for d in A.POOLS[5][1]}
means = [np.mean([effect[d] for d in g]) for g in groups]
p, n = A.randomisation_p(means, means[groups.index(real)])
check("p = 1/252", abs(p - 1 / 252) < 1e-12 and n == 252, f"p {p:.4f}")

print("5. randomisation p under the null")
ps = []
for rep in range(2000):
    vals = {d: rng.normal(0.2, 0.05) for d in A.POOLS[5][1]}
    m = [np.mean([vals[d] for d in g]) for g in groups]
    ps.append(A.randomisation_p(m, m[groups.index(real)])[0])
ps = np.array(ps)
share = (ps <= 0.05).mean()
check("share p <= 0.05 near 0.05", 0.02 <= share <= 0.08, f"{share:.3f}")
check("median p near 0.5", 0.4 <= np.median(ps) <= 0.6, f"{np.median(ps):.3f}")

print("6. pool statistic through relabelling")
rows = []
for d in A.POOLS[5][1]:
    for g in ("man", "woman"):
        signal = 0.3 if d in A.SOCIAL["religion"] else 0.1
        rows += [{"Name": A.template_for(d, g), "signal": signal}] * 3
rows += [{"Name": "man", "signal": 0.0}, {"Name": "woman", "signal": 0.0}]
fake = pd.DataFrame(rows)


def fake_pct(subset, intersection):
    labelled = subset[~subset["Name"].isin(["man", "woman"])]
    return {"mean": float(labelled["signal"].mean())}


A.interaction_pct = fake_pct
labels = A.SOCIAL["religion"]
pm = [A._pool_one((fake, g, labels, "religion")) for g in groups]
p6, _ = A.randomisation_p(pm, pm[groups.index(real)])
check("recovered after relabelling", abs(p6 - 1 / 252) < 1e-12,
      f"real {pm[groups.index(real)]:.3f}, p {p6:.4f}")
neutral = tuple(A.NEUTRAL5)
check("neutral group is the least extreme",
      A.randomisation_p(pm, pm[groups.index(neutral)])[0] == 1.0)

print("=" * 60)
print(f"{'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
