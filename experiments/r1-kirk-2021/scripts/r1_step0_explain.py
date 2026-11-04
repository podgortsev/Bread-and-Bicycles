#!/usr/bin/env python3
"""
r1_step0_explain.py - explain every Step 0 difference, job by job.

    python r1_step0_explain.py <kirk repo>

Step 0 missed the 0.02 tolerance on ethnicity (differences of exactly 1/47 and
2/47) and returned one extra job regression for sexuality (64 against 63). The
preregistration says no floor comparison is interpreted until that is
explained. This finds each job whose significance call differs from the
published file and prints the p-value our run gives, and for sexuality the job
that entered and its count against the threshold.
"""

import contextlib
import io
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_step0_replicate as R  # noqa: E402


def published_flags(repo, inter):
    df = pd.read_csv(os.path.join(repo, "regression_results",
                                  f"{inter}_logistic_regressions.csv"),
                     index_col=0)
    cols = [c for c in df.columns if c.endswith(":p<0.05")]
    flags = df[cols]
    flags.columns = [c[:-len(":p<0.05")] for c in cols]
    return flags


def main(repo):
    R.compatibility_shim()
    ns = R.load_their_functions(repo, ["Generate_Freq_Matrices.py",
                                       "Logistic_Regression.py"])
    freq = R.build_freq_matrix(repo, ns)

    for inter in ("ethnicity", "sexuality"):
        threshold = 0.0025 * R.TOTAL_CALLS[inter]
        reg_df = ns["make_regression_df"](freq, None, inter, "range", threshold)
        jobs = [c for c in reg_df.columns if c not in ("gender", "category")]
        pub = published_flags(repo, inter)
        print("=" * 78)
        print(f"{inter}: {len(jobs)} jobs in our run, {pub.shape[1]} published")
        extra = sorted(set(jobs) - set(pub.columns))
        missing = sorted(set(pub.columns) - set(jobs))
        if extra or missing:
            subs = reg_df.drop(columns=["gender"])
            for job in extra + missing:
                total = int(reg_df[job].sum()) if job in reg_df else None
                print(f"  job only in {'ours' if job in extra else 'theirs'}: "
                      f"{job}, count {total}, threshold {threshold}")
        for job in [j for j in jobs if j in pub.columns]:
            with contextlib.redirect_stdout(io.StringIO()):
                res = ns["run_regression_interaction"](reg_df, job, "base", "man")
            if res is None:
                print(f"  {job}: our fit failed")
                continue
            est = res[0]
            for term, p in est.pvalues.items():
                if term not in pub.index:
                    continue
                ours = bool(p < 0.05)
                theirs = bool(pub.loc[term, job])
                if ours != theirs:
                    print(f"  {job:22}{term:18} p = {p:.4f}  ours "
                          f"{'signif' if ours else 'not'}, published "
                          f"{'signif' if theirs else 'not'}, "
                          f"converged {est.mle_retvals.get('converged')}")
    print("=" * 78)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
