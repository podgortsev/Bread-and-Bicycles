#!/usr/bin/env python3
"""
p3_stats.py - the statistics every Paper 3 analyser uses. No GPU, no model.

One copy, imported by all of them, so a correction lands everywhere at once.
Six experiments copying five functions is five chances to fix a bug in one place
and not the others.

Every function here is exercised against a hand-computed answer by
`e1-type-x-count/scripts/validate_e1_accuracy.py`, which is the canonical test
for this module.

    bh              Benjamini-Hochberg across a family of tests
    exact_mcnemar   binomial test on discordant pairs, not the chi-squared form
    wilcoxon        signed-rank on the nonzero differences, sign test as fallback
    boot_ci         mean with a percentile bootstrap interval
    mde             minimum detectable effect: the 95 percent half-width at this n
    sign_flip       paired permutation test on the MEAN
    hodges_lehmann  the location estimate the signed-rank test speaks to
    holm            family-wise step-down, for union questions such as floors

TWO TESTS ON CONTINUOUS OUTCOMES, AND WHY
-----------------------------------------
The estimate reported is a mean with a bootstrap interval. The signed-rank test
does not test a mean: it tests the location of the ranks. On well-behaved data
the two agree. On channel A2 they did not. Qwen answers 46 percent of tasks at
the ceiling, so a third of its paired differences are near zero, and what moves
is carried by a tail of tasks the model fails. One floor there had a mean of
+0.18 with an interval excluding zero and a signed-rank p of 0.87.

Choosing between the tests after seeing that would be a forking path. So the
analysers run both and apply the rule that can only remove claims: a FLOOR is
contaminated if EITHER test rejects, and an EFFECT is claimed only if BOTH
reject and the mean points the same way as the signed-rank statistic.

The direction check first used the Hodges-Lehmann estimate. On channel A's
binary outcomes 83 to 92 percent of interaction differences are exactly zero, so
that estimate is 0 for every contrast, real effects included, and the check
would have silently removed llama's stated-by-stated result, which both tests
reject. The side the signed-rank statistic leans to ignores zeros, as the test
itself does. Hodges-Lehmann is still reported, as an estimate.

    two_tests       both p-values, the claim p-value max(rank, mean), the rank
                    direction and Hodges-Lehmann, in one call
    claimed         the claim rule
    holm_either     one family-wise error over rows, on either test
    correct         the whole rule for a table: floors family-wise per model on
                    either test, everything else Benjamini-Hochberg on both

THE ONE THAT MATTERS MOST IS mde.
---------------------------------
Paper 3 asks about interactions. An interaction is a difference of differences
and carries about twice the variance of a single difference, so a null is
extremely easy to produce and extremely easy to misread. Every analyser in this
paper prints the minimum detectable effect beside every estimate, and no null is
reported without it.
"""

import random

try:
    from scipy import stats as sps
except ImportError:
    sps = None

try:
    import numpy as np
except ImportError:
    np = None

BOOTSTRAP = 4000
PERMUTATIONS = 10000
SEED = 20260908


def _no_nan(pvals, who):
    """A NaN p-value makes the sort order arbitrary and the adjusted values
    meaningless. e5 had two correlation cells with no variance, and their NaN
    turned "1 of 8 cells significant" into "5 of 8". Fail loudly instead."""
    if any(x != x for x in pvals):
        raise ValueError(f"{who}: p-values contain NaN. Drop the degenerate "
                         f"cells and report them separately.")


def bh(pvals):
    """Benjamini-Hochberg. Returns adjusted p in the input order."""
    _no_nan(pvals, "bh")
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = n - rank + 1
        val = min(prev, pvals[i] * n / k)
        adj[i] = val
        prev = val
    return adj


def exact_mcnemar(pairs):
    """pairs: list of (y_a, y_b) in {0,1}. Binomial test on discordant pairs."""
    b = sum(1 for a, c in pairs if a == 1 and c == 0)
    c = sum(1 for a, cc in pairs if a == 0 and cc == 1)
    n = b + c
    if n == 0:
        return b, c, 1.0
    if sps is not None:
        p = sps.binomtest(b, n, 0.5).pvalue
    else:
        from math import comb
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
    return b, c, p


def wilcoxon(d):
    """Signed-rank on the nonzero values. Falls back to a sign test."""
    nz = [x for x in d if x != 0]
    if not nz:
        return 1.0, "no nonzero differences"
    if sps is not None:
        try:
            return float(sps.wilcoxon(nz).pvalue), "wilcoxon signed-rank"
        except Exception:
            pass
    pos = sum(1 for x in nz if x > 0)
    if sps is not None:
        return float(sps.binomtest(pos, len(nz), 0.5).pvalue), "sign test"
    from math import comb
    n = len(nz)
    k = min(pos, n - pos)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail), "sign test"


def boot_ci(d, reps=BOOTSTRAP, seed=SEED):
    """Mean of d with a percentile bootstrap interval.

    reps=0 skips the resampling and returns the mean with a nan interval, which
    the validator uses when it needs thousands of fits and only the point
    estimate matters.
    """
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(d) / n
    if reps <= 0:
        return mean, float("nan"), float("nan")
    if np is not None:
        arr = np.asarray(d, dtype=float)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(reps, n))
        means = arr[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return mean, float(lo), float(hi)
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(reps))
    return mean, means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def mde(d):
    """Minimum detectable interaction: the 95 percent half-width at this n.

    Computed from the observed spread of d, so it already contains whatever
    pairing bought us. Reported in the same units as the estimate, which for a
    binary outcome is proportion of tasks.
    """
    n = len(d)
    if n < 2:
        return float("nan")
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    return 1.96 * (var ** 0.5) / (n ** 0.5)


def sign_flip(d, reps=None, seed=SEED):
    """Two-sided paired permutation test on the mean of d.

    Under the null each paired difference is as likely to carry either sign, so
    the observed |mean| is compared with the |mean| of randomly sign-flipped
    copies. Tests the same quantity boot_ci estimates. p never falls below
    1 / (reps + 1).
    """
    reps = PERMUTATIONS if reps is None else reps
    n = len(d)
    if n == 0:
        return 1.0
    obs = abs(sum(d) / n)
    if obs == 0:
        return 1.0
    tol = 1e-12 * max(1.0, obs)
    if np is not None:
        arr = np.asarray(d, dtype=float)
        rng = np.random.default_rng(seed)
        hits, left = 0, reps
        while left > 0:
            k = min(left, 2000)
            signs = rng.integers(0, 2, size=(k, n), dtype=np.int8) * 2 - 1
            means = np.abs(signs.astype(float) @ arr) / n
            hits += int(np.sum(means >= obs - tol))
            left -= k
        return (hits + 1) / (reps + 1)
    rng = random.Random(seed)
    hits = 0
    for _ in range(reps):
        s = sum(x if rng.random() < 0.5 else -x for x in d)
        hits += abs(s / n) >= obs - tol
    return (hits + 1) / (reps + 1)


def hodges_lehmann(d):
    """Median of all pairwise Walsh averages (d_i + d_j) / 2, i <= j."""
    n = len(d)
    if n == 0:
        return 0.0
    if np is not None:
        arr = np.asarray(d, dtype=float)
        i, j = np.triu_indices(n)
        return float(np.median((arr[i] + arr[j]) / 2))
    w = sorted((d[i] + d[j]) / 2 for i in range(n) for j in range(i, n))
    m = len(w)
    return w[m // 2] if m % 2 else (w[m // 2 - 1] + w[m // 2]) / 2


def holm(pvals):
    """Holm step-down, family-wise error. Adjusted p in input order.

    For union questions such as "does ANY of these floor pairs move". BH
    controls the false discovery rate, which is the wrong error for a union, and
    when it is run over floors and strong audited effects together it becomes
    MORE lenient for the floors: the more an audit finds, the easier a quiet
    floor is flagged. The interaction floor test's own unit tests caught that.
    """
    _no_nan(pvals, "holm")
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (n - rank) * pvals[i]))
        adj[i] = running
    return adj


def rank_direction(d):
    """+1 or -1: the side the signed-rank statistic leans to; 0 if balanced.

    Ranks of |d| over the nonzero values, ties averaged, summed with the sign of
    each value. Zeros are dropped, exactly as the test drops them.
    """
    nz = sorted((x for x in d if x != 0), key=abs)
    n = len(nz)
    total = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[j + 1]) == abs(nz[i]):
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            total += r if nz[k] > 0 else -r
        i = j + 1
    return (total > 0) - (total < 0)


def two_tests(d):
    """Signed-rank and sign-flip on one set of paired differences.

    p is max(p_rank, p_mean): the maximum of two valid p-values is a valid
    p-value for BOTH rejecting, so correcting over it asks for both.
    """
    p_rank, _ = wilcoxon(d)
    p_mean = sign_flip(d)
    return {"p_rank": p_rank, "p_mean": p_mean, "p": max(p_rank, p_mean),
            "rank_dir": rank_direction(d), "hl": hodges_lehmann(d)}


def claimed(mean, rank_dir, p_adjusted, alpha=0.05):
    """Both tests reject after correction, and the mean and the signed-rank
    statistic lean the same way."""
    return (p_adjusted < alpha and mean != 0 and rank_dir != 0
            and (mean < 0) == (rank_dir < 0))


def holm_either(rows, alpha=0.05):
    """One family-wise error for "does ANY of these move, on EITHER test".

    Holm across every row per test, then Bonferroni across the two tests:
    holm = min(1, 2 * min(Holm rank, Holm mean)). Correcting each test and each
    set of rows at 0.05 and OR-ing the verdicts let a quiet floor be flagged
    5.7 percent of the time on aggregate floors alone, and 12 percent once the
    strata were added, in the interaction floor test's own checks.
    """
    for r, a, b in zip(rows, holm([x["p_rank"] for x in rows]),
                       holm([x["p_mean"] for x in rows])):
        r["holm_rank"], r["holm_mean"] = a, b
        r["holm"] = min(1.0, 2 * min(a, b))
    return [r for r in rows if r["holm"] < alpha]


def correct(rows, group=("model",), alpha=0.05):
    """The whole rule for one table of two_tests rows.

    Rows of type "floor" are corrected family-wise on either test, one family
    per group (by default per model), and "claimed" means the floor MOVES. All
    other rows are corrected together with Benjamini-Hochberg over the claim
    p-value, and "claimed" applies the direction check. Both get "bh", holding
    the adjusted p-value that decided them, so existing tables keep printing.
    Floors are never in the same family as the contrasts they protect.
    """
    families = {}
    for r in rows:
        if r.get("type") == "floor":
            families.setdefault(tuple(r.get(k) for k in group), []).append(r)
    for fam in families.values():
        holm_either(fam, alpha)
        for r in fam:
            r["bh"] = r["holm"]
            r["claimed"] = r["holm"] < alpha
    others = [r for r in rows if r.get("type") != "floor"]
    for r, a in zip(others, bh([x["p"] for x in others])):
        r["bh"] = a
        r["claimed"] = claimed(r["mean"], r["rank_dir"], a, alpha)
    return rows
