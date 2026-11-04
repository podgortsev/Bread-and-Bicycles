#!/usr/bin/env python3
"""
r7_step0.py - ABLEIST Step 0 (Addendum 7): their Table 6 for the two open models
from their full dataset. Prints aggregates only; the data are not redistributed.

    python r7_step0.py --data <data-restricted/ableist/labeled_ableism_complete_dataset_full.csv>

Readings A, B, C in the registered order; the first that passes is used.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import ranksums

IST = ["Tokenism", "Inspiration Porn", "Superhumanization Harm"]
ALL8 = ["One-size-fits-all Ableism", "Infantilization", "Technoableism",
        "Anticipated Ableism", "Ability Saviorism"] + IST
MODELS = {"meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
          "allenai/OLMo-2-1124-7B-Instruct": "OLMo-2-7B"}
COMPARISONS = [("dis", "dis_gen"), ("dis", "dis_cas"), ("dis", "dis_nat"),
               ("dis", "dis_gen_cas"), ("dis_gen", "dis_gen_cas"),
               ("dis_cas", "dis_gen_cas")]
# Table 6 (PDF p. 6): (percent change, effect) per comparison, in order
TABLE6 = {
    "OLMo-2-7B": [(32.56, 0.37), (23.26, 0.25), (20.93, 0.22), (53.49, 0.58),
                  (15.79, 0.22), (24.53, 0.31)],
    "Llama-3.1-8B": [(19.61, 0.19), (-10.29, -0.19), (13.24, 0.10),
                     (51.47, 0.47), (26.64, 0.26), (68.85, 0.42)],
}


def groups(df, restrict):
    blank = lambda c: df[c].isna() | (df[c].astype(str).str.strip() == "")
    dis = ~blank("Disability")
    cn = df["Caste/Nationality"].fillna("")
    caste = cn.isin(["Dalit", "Brahmin"])
    nat = cn.isin(["American", "Indian"])
    gen_vals = ["Woman", "Transgender"] if restrict else ["Man", "Woman", "Transgender"]
    cas_vals = ["Dalit"] if restrict else ["Dalit", "Brahmin"]
    gen = df["Gender"].isin(gen_vals)
    cas = cn.isin(cas_vals)
    no_gen = blank("Gender")
    no_cn = cn == ""
    return {
        "dis": dis & no_gen & no_cn,
        "dis_gen": dis & gen & no_cn,
        "dis_cas": dis & no_gen & cas,
        "dis_nat": dis & no_gen & nat,
        "dis_gen_cas": dis & gen & cas,
    }


def table(df, harms, restrict):
    out = {}
    for full, short in MODELS.items():
        d = df[df["LLM"] == full]
        score = d[harms].astype(float).mean(axis=1)
        g = groups(d, restrict)
        rows = []
        for a, b in COMPARISONS:
            xa, xb = score[g[a]], score[g[b]]
            pct = (xb.mean() - xa.mean()) / xa.mean() * 100
            stat = ranksums(xb, xa).statistic
            r = stat / np.sqrt(min(len(xa), len(xb)))
            rows.append((pct, r, len(xa), len(xb)))
        out[short] = rows
    return out


def matches(ours, theirs):
    hits = 0
    for (p, r, _, _), (tp, tr) in zip(ours, theirs):
        hits += abs(round(p, 2) - tp) <= 0.1
        hits += abs(round(r, 2) - tr) <= 0.01
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.data, keep_default_na=True)
    readings = [("A", IST, True), ("B", ALL8, True), ("C", IST, False)]
    print("=" * 78)
    print("R7 ABLEIST STEP 0: Table 6, open models (Addendum 7)")
    print("=" * 78)
    chosen = None
    for name, harms, restrict in readings:
        t = table(df, harms, restrict)
        hits = {m: matches(t[m], TABLE6[m]) for m in t}
        ok = all(h >= 10 for h in hits.values())
        print(f"\nreading {name}: " + ", ".join(f"{m} {h}/12" for m, h in hits.items())
              + f"   {'PASS' if ok else 'fail'}")
        for m in t:
            for (a, b), (p, r, na, nb), (tp, tr) in zip(COMPARISONS, t[m], TABLE6[m]):
                print(f"  {m:13} {a:8}->{b:12} ours {p:7.2f} {r:5.2f}  theirs {tp:7.2f} {tr:5.2f}  n {na}/{nb}")
        if ok:
            chosen = name
            break
    print(f"\nVERDICT: {'PASS with reading ' + chosen if chosen else 'FAIL: Table 6 not reproduced from their data'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
