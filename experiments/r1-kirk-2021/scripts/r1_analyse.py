#!/usr/bin/env python3
"""
r1_analyse.py - Kirk et al. 2021: fidelity, the floor, and the decision.

    python r1_analyse.py <kirk repo> <our ner.csv> [--reps 100] [--workers 12]

Preregistration R1 Steps 2 and 3, with Addenda 1 to 4. Written before the
full generation finished; nothing here depends on its results.

THE QUANTITY
------------
Their Pct_Signif for an interaction term woman:X is the share of per-job
logistic regressions in which that term has p < 0.05. It comes from the
interaction hierarchy alone (their regression_results takes p-values only from
the interaction models), so this script calls their initialize_regression with
"interactions" and counts, which gives their number at a third of the cost.
The mean over a group's interaction terms is the statistic compared.

GROUPS
------
    size 5   religion (theirs)       against  hungry thirsty sleepy busy cheerful
    size 4   ethnicity (theirs)      against  hungry thirsty sleepy busy
    size 2   sexuality, political    against  hungry sleepy

Neutral details are relabelled onto the group's category names in a copy of the
data so that their make_regression_df runs unchanged (Addendum 2).

BOOTSTRAP
---------
Completions are resampled with replacement within each template, the frequency
matrix is rebuilt with their make_freq_matrix, and the group's regressions are
refitted. 100 resamples; the percentile interval is the 3rd to the 97th sorted
value.

Addendum 4: resampling pushes a share of p < 0.05 upward, so every interval
used below is the percentile interval shifted by (point estimate - bootstrap
median), centring it on the estimate it describes. The unshifted interval is
printed beside it and decides nothing.

THE DECISION (preregistered)
----------------------------
    floor explains the claim   neutral mean >= social mean, or the social mean
                               lies inside the neutral interval
    claim survives the floor   social mean above the neutral interval at every
                               group size
    partial                    otherwise

RANDOMISATION TEST (Addendum 4, secondary, does not change the decision)
-------------------------------------------------------------------------
For each group size, the social and neutral descriptors of that size form a
pool; every group of that size drawn from the pool is relabelled onto the
size's category names and run through the same regressions (point estimates,
no bootstrap). p = share of pool groups whose mean is at least the real social
group's. Resolution: 252 groups (size 5), 70 (size 4), 15 (size 2, pool gay,
straight, conservative, liberal, hungry, sleepy).
"""

import argparse
import contextlib
import io
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r1_step0_replicate as R0     # noqa: E402

NEUTRAL5 = ["hungry", "thirsty", "sleepy", "busy", "cheerful"]
SOCIAL = {
    "religion": ["Buddhist", "Christian", "Hindu", "Jewish", "Muslim"],
    "ethnicity": ["Asian", "Black", "Hispanic", "White"],
    "sexuality": ["gay", "straight"],
    "political": ["conservative", "liberal"],
}
NEUTRAL_FOR = {
    "religion": NEUTRAL5,
    "ethnicity": NEUTRAL5[:4],
    "sexuality": [NEUTRAL5[0], NEUTRAL5[2]],
    "political": [NEUTRAL5[0], NEUTRAL5[2]],
}
# the neutral size-2 group is run through the sexuality labels
RUN_AS = {"religion": "religion", "ethnicity": "ethnicity",
          "sexuality": "sexuality", "political": "sexuality"}
SIZE = {g: len(v) for g, v in SOCIAL.items()}

_NS = None
_REPO = None


def names_for(categories, lesbian=False):
    out = ["man", "woman"]
    for c in categories:
        out.append(f"{c} man")
        out.append("lesbian woman" if (lesbian and c == "gay") else f"{c} woman")
    return out


def social_subset(ner, group):
    names = names_for(SOCIAL[group], lesbian=(group == "sexuality"))
    return ner[ner["Name"].isin(names)].copy()


def neutral_subset(ner, group):
    """Neutral templates relabelled onto the labels of RUN_AS[group]."""
    target = SOCIAL[RUN_AS[group]]
    mapping = {"man": "man", "woman": "woman"}
    for neu, lab in zip(NEUTRAL_FOR[group], target):
        mapping[f"{neu} man"] = f"{lab} man"
        mapping[f"{neu} woman"] = ("lesbian woman" if lab == "gay"
                                   else f"{lab} woman")
    sub = ner[ner["Name"].isin(mapping)].copy()
    sub["Name"] = sub["Name"].map(mapping)
    return sub


def template_for(descriptor, gender):
    if gender == "woman" and descriptor == "gay":
        return "lesbian woman"
    return f"{descriptor} {gender}"


def relabelled_subset(ner, descriptors, labels):
    """Templates of `descriptors` renamed onto the templates of `labels`."""
    mapping = {"man": "man", "woman": "woman"}
    for d, lab in zip(descriptors, labels):
        for gender in ("man", "woman"):
            mapping[template_for(d, gender)] = template_for(lab, gender)
    sub = ner[ner["Name"].isin(mapping)].copy()
    sub["Name"] = sub["Name"].map(mapping)
    return sub


# Addendum 4 pools: label set the groups run under, and the descriptor pool
POOLS = {
    5: ("religion", SOCIAL["religion"] + NEUTRAL5),
    4: ("ethnicity", SOCIAL["ethnicity"] + NEUTRAL5[:4]),
    2: ("sexuality", SOCIAL["sexuality"] + SOCIAL["political"]
        + [NEUTRAL5[0], NEUTRAL5[2]]),
}


def pool_groups(size):
    """Every group of `size` descriptors from the pool, in a fixed order."""
    return list(itertools.combinations(POOLS[size][1], size))


def randomisation_p(pool_means, real_mean):
    """Share of pool groups at least as extreme as the real social group."""
    vals = np.asarray(pool_means, dtype=float)
    return float((vals >= real_mean - 1e-12).mean()), len(vals)


def centred_interval(values, point):
    """Addendum 4: percentile interval shifted by point - bootstrap median."""
    lo, hi = interval(values)
    shift = point - float(np.median(values))
    return lo + shift, hi + shift


def _pool_one(args):
    ner, descriptors, labels, intersection = args
    return interaction_pct(relabelled_subset(ner, descriptors, labels),
                           intersection)["mean"]


def _init(repo):
    global _NS, _REPO
    R0.compatibility_shim()
    _REPO = repo
    _NS = R0.load_their_functions(repo, ["Generate_Freq_Matrices.py",
                                         "Logistic_Regression.py"])


def interaction_pct(ner_subset, intersection):
    """Their Pct_Signif for each interaction term, and the mean."""
    ns = _NS
    with contextlib.redirect_stdout(io.StringIO()):
        cleaned = ns["clean_up_data"](ner_subset.reset_index(drop=True),
                                      template="identity")
        freq = ns["make_freq_matrix"](cleaned, ns["load_job_replacements"](_REPO))
        threshold = 0.0025 * R0.TOTAL_CALLS[intersection]
        reg = ns["make_regression_df"](freq, None, intersection, "range",
                                       threshold)
        models, jobs = ns["initialize_regression"](reg, "interactions")
    pvals = pd.DataFrame({j: m.pvalues for m, j in zip(models, jobs)})
    inter = pvals.loc[[i for i in pvals.index if i.startswith("woman:")]]
    pct = (inter < 0.05).mean(axis=1)
    zero_cells = 0
    for j in jobs:
        cells = reg.groupby(["category", "gender"])[j].sum()
        zero_cells += int((cells == 0).any())
    return {"terms": pct.to_dict(), "mean": float(pct.mean()),
            "jobs": len(jobs), "jobs_with_empty_cell": zero_cells}


def _boot_one(args):
    ner_subset, intersection, seed = args
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in ner_subset.groupby("Name"):
        idx = rng.integers(0, len(g), size=len(g))
        parts.append(g.iloc[idx])
    return interaction_pct(pd.concat(parts), intersection)["mean"]


def interval(values):
    v = sorted(values)
    n = len(v)
    return v[int(round(0.03 * n)) - 1 if n >= 34 else 0], \
        v[int(round(0.97 * n)) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("ner")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    _init(args.repo)
    ours = pd.read_csv(args.ner, index_col=0)
    theirs = pd.read_csv(os.path.join(args.repo, "data", "GPT-2", "NER_output",
                                      "identity_occupations_template.csv"),
                         index_col=0)

    print("=" * 78)
    print("R1 KIRK ET AL. 2021: the floor test")
    print("=" * 78)

    print("\nUNINTERPRETABILITY CHECK: share of completions with no job title")
    their_missing = (theirs["Title"] == "[]").mean()
    miss = ours.groupby("Name")["Title"].apply(lambda s: (s == "[]").mean())
    print(f"  theirs overall {their_missing:.3f}; limit {2 * their_missing:.3f}")
    for name, m in miss.sort_index().items():
        flag = "  <- ABOVE LIMIT" if m > 2 * their_missing else ""
        print(f"  {name:22}{m:.3f}{flag}")

    jobs = []
    arms = {}
    for group in SOCIAL:
        run_as = RUN_AS[group]
        arms[("social", group)] = (social_subset(ours, group), group)
        arms[("neutral", group)] = (neutral_subset(ours, group), run_as)
        arms[("theirs", group)] = (social_subset(theirs, group), group)

    point = {}
    for key, (subset, inter) in arms.items():
        point[key] = interaction_pct(subset, inter)

    boots = {}
    with Pool(args.workers, initializer=_init, initargs=(args.repo,)) as pool:
        for key, (subset, inter) in arms.items():
            seeds = [(subset, inter, 20260912 + i) for i in range(args.reps)]
            boots[key] = pool.map(_boot_one, seeds)

    print(f"\nSTEP 2. FIDELITY: our religion terms against the interval of "
          f"theirs ({args.reps} resamples)")
    tm = point[("theirs", "religion")]["mean"]
    lo, hi = centred_interval(boots[("theirs", "religion")], tm)
    ulo, uhi = interval(boots[("theirs", "religion")])
    their_terms = point[("theirs", "religion")]["terms"]
    our_terms = point[("social", "religion")]["terms"]
    for term in sorted(their_terms):
        print(f"  {term:18} theirs {their_terms[term]:.3f}   ours "
              f"{our_terms.get(term, float('nan')):.3f}")
    om = point[("social", "religion")]["mean"]
    fid = lo <= om <= hi
    print(f"  mean: theirs {tm:.3f} centred [{lo:.3f}, {hi:.3f}] "
          f"(unshifted [{ulo:.3f}, {uhi:.3f}])   ours {om:.3f}   "
          f"{'PASS' if fid else 'FAIL: pipeline not shown equivalent'}")

    print(f"\nSTEP 3. THE FLOOR: mean share of significant interaction terms")
    print(f"  intervals centred on the point estimate (Addendum 4); "
          f"unshifted in brackets after")
    print(f"  {'group':11}{'size':>5}{'social':>9}{'interval':>18}"
          f"{'neutral':>9}{'interval':>18}{'theirs':>9}   verdict")
    verdicts = []
    for group in SOCIAL:
        s = point[("social", group)]["mean"]
        n = point[("neutral", group)]["mean"]
        t = point[("theirs", group)]["mean"]
        slo, shi = centred_interval(boots[("social", group)], s)
        nlo, nhi = centred_interval(boots[("neutral", group)], n)
        if n >= s or nlo <= s <= nhi:
            v = "floor explains the claim"
        elif s > nhi:
            v = "claim survives the floor"
        else:
            v = "partial"
        verdicts.append(v)
        uslo, ushi = interval(boots[("social", group)])
        unlo, unhi = interval(boots[("neutral", group)])
        print(f"  {group:11}{SIZE[group]:5}{s:9.3f}  [{slo:.3f}, {shi:.3f}]"
              f"{n:9.3f}  [{nlo:.3f}, {nhi:.3f}]{t:9.3f}   {v}")
        print(f"  {'':16}unshifted  [{uslo:.3f}, {ushi:.3f}]"
              f"{'':9}  [{unlo:.3f}, {unhi:.3f}]")
    if all(v == "claim survives the floor" for v in verdicts):
        overall = "CLAIM SURVIVES THE FLOOR"
    elif all(v == "floor explains the claim" for v in verdicts):
        overall = "THE FLOOR EXPLAINS THE CLAIM"
    else:
        overall = "PARTIAL: see the groups"
    print(f"\n  overall: {overall}")

    print("\nSECONDARY (Addendum 4): randomisation over descriptor pools, "
          "point estimates")
    for size in sorted(POOLS, reverse=True):
        labels_group, pool = POOLS[size]
        labels = SOCIAL[labels_group]
        groups = pool_groups(size)
        keep = {"man", "woman"} | {template_for(d, x) for d in pool
                                   for x in ("man", "woman")}
        pool_ner = ours[ours["Name"].isin(keep)]
        with Pool(args.workers, initializer=_init,
                  initargs=(args.repo,)) as pool_exec:
            means = pool_exec.map(
                _pool_one, [(pool_ner, g, labels, labels_group)
                            for g in groups])
        by_group = dict(zip(groups, means))
        for real in [g for g in SOCIAL if SIZE[g] == size]:
            key = tuple(SOCIAL[real])
            p, n = randomisation_p(means, by_group[key])
            print(f"  size {size}  {real:10} mean {by_group[key]:.3f} "
                  f"(decision-rule point {point[('social', real)]['mean']:.3f})"
                  f"   p = {p:.3f}  over {n} groups, resolution {1 / n:.3f}")

    print("\nSECONDARY: regressions with an empty category-by-gender cell")
    for key in sorted(point):
        p = point[key]
        print(f"  {key[0]:8}{key[1]:11} jobs {p['jobs']:3}   with an empty "
              f"cell {p['jobs_with_empty_cell']:3}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
