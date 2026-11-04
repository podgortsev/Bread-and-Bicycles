#!/usr/bin/env python3
"""
r1_step0_replicate.py - Kirk et al. 2021, their numbers from their data.

    python r1_step0_replicate.py <path to a clone of oxai/intersectional_gpt2>

Preregistration, R1 Step 0. Runs the authors' own released functions,
unmodified, on their released per-completion job labels, and compares the result
with their released summary table. Pass: every Pct_Signif within 0.02.

Run in a 2021-era environment (pandas 1.5.3, statsmodels 0.13.5, numpy 1.24.4):
their code uses groupby(axis=1), which later pandas removed.

HOW THEIR CODE IS USED
----------------------
Their scripts define functions and then run module-level code against their own
local paths. Importing them would run that code. So each file is parsed and only
its import statements and function definitions are executed, verbatim, into one
namespace. Nothing inside a function is changed.
"""

import ast
import contextlib
import io
import os
import sys

import pandas as pd

INTERSECTIONS = ["ethnicity", "religion", "sexuality", "political"]
TOTAL_CALLS = {"base": 14000, "ethnicity": 56000, "religion": 70000,
               "sexuality": 28000, "political": 28000}
TOLERANCE = 0.02


def compatibility_shim():
    """One compatibility shim, no change to any computation.

    Their regression_results() passes `columns={'R2_main'}`, a one-element
    set, to DataFrame.from_dict. The pandas of 2021 accepted that; pandas 1.5
    raises "columns cannot be a set". The shim turns a set into a list before
    the call. Their source is not edited.
    """
    original = pd.DataFrame.from_dict.__func__

    def from_dict(cls, data, orient="columns", dtype=None, columns=None):
        if isinstance(columns, set):
            columns = list(columns)
        return original(cls, data, orient=orient, dtype=dtype, columns=columns)

    pd.DataFrame.from_dict = classmethod(from_dict)


def load_their_functions(repo, names):
    """Execute only the imports and function definitions of their scripts."""
    ns = {}
    for name in names:
        path = os.path.join(repo, "scripts", name)
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        keep = [n for n in tree.body
                if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
        module = ast.Module(body=keep, type_ignores=[])
        exec(compile(module, path, "exec"), ns)
    return ns


def build_freq_matrix(repo, ns):
    data = os.path.join(repo, "data", "GPT-2", "NER_output",
                        "identity_occupations_template.csv")
    df = pd.read_csv(data, index_col=0)
    cleaned = ns["clean_up_data"](df, template="identity")
    replacements = ns["load_job_replacements"](repo)
    return ns["make_freq_matrix"](cleaned, replacements)


def main(repo):
    compatibility_shim()
    ns = load_their_functions(repo, ["Generate_Freq_Matrices.py",
                                     "Logistic_Regression.py"])
    print("=" * 78)
    print("R1 STEP 0. Kirk et al. 2021: their functions, their data")
    print("=" * 78)
    freq = build_freq_matrix(repo, ns)
    print(f"frequency matrix: {freq.shape[0]} rows, {freq.shape[1] - 3} job "
          f"columns")

    published = pd.read_csv(os.path.join(
        repo, "regression_results", "summary_regression_results.csv"))

    rows = []
    for inter in INTERSECTIONS:
        threshold = 0.0025 * TOTAL_CALLS[inter]
        with contextlib.redirect_stdout(io.StringIO()):
            R2, pvalue, coeff, mean_signif = ns["return_results_tables"](
                freq, None, inter, "range", threshold)
        ours = mean_signif.to_dict()
        pub = published[published["Intersection"] == inter]
        for _, r in pub.iterrows():
            var = r["Variable"]
            mine = ours.get(var)
            rows.append({"intersection": inter, "variable": var,
                         "jobs_published": int(r["Job Regs"]),
                         "jobs_ours": pvalue.shape[1],
                         "published": float(r["Pct_Signif"]),
                         "ours": float(mine) if mine is not None else float("nan"),
                         "int_r2_published": float(r["Int_R2_Change"]),
                         "int_r2_ours": float(R2["R2_int_change"].mean())})
        pvalue.to_csv(f"r1_step0_{inter}_pvalue.csv")

    out = pd.DataFrame(rows)
    out["diff"] = (out["ours"] - out["published"]).abs()
    out.to_csv("r1_step0_comparison.csv", index=False)

    print(f"\n{'group':11}{'term':18}{'jobs pub/ours':>14}{'published':>11}"
          f"{'ours':>8}{'diff':>8}")
    for _, r in out.iterrows():
        flag = "" if r["diff"] <= TOLERANCE else "  <- outside 0.02"
        print(f"{r['intersection']:11}{r['variable']:18}"
              f"{r['jobs_published']:>7}/{r['jobs_ours']:<6}"
              f"{r['published']:11.3f}{r['ours']:8.3f}{r['diff']:8.3f}{flag}")

    inter_rows = out[out["variable"].str.startswith("woman:")]
    print("\ninteraction terms only")
    for inter in INTERSECTIONS:
        sel = inter_rows[inter_rows["intersection"] == inter]
        print(f"  {inter:11} mean Pct_Signif published "
              f"{sel['published'].mean():.3f}  ours {sel['ours'].mean():.3f}")
    worst = out["diff"].max()
    verdict = "PASS" if worst <= TOLERANCE else "FAIL"
    print(f"\nlargest difference {worst:.3f} against tolerance {TOLERANCE}: "
          f"{verdict}")
    print("=" * 78)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
