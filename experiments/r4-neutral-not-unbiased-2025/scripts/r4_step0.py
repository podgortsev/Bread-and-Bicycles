#!/usr/bin/env python3
"""
r4_step0.py - Step 0 for Ghanbari Haez and Dragoni 2025, "Neutral Is Not
Unbiased" (EMNLP Findings 2025): their numbers from their data, before
anything neutral is generated.

    python r4_step0.py --repo <clone of Saba-Gh-H/Neutral_isnot_unbiased> --work <dir>

For each model, the first code cell of their own notebook
"E. Counterfactual Analysis/CounterAnalysis<Model>.ipynb" is executed
unchanged, in a working directory holding their released answers. It writes
their per-row CSV (similarity, sentiment shift). This script then prints the
quantities the paper reports on p. 8:

    mean semantic similarity     (Mistral-Small 3.1: 0.88, Phi-4: 0.85)
    negative vs positive shifts  (Mistral-Small 3.1: 1,268 vs 1,232;
                                  Phi-4: 1,289 vs 1,190)
    mean sentiment shift         (Phi-4: -0.0045)
    per comparison pair: mean similarity and mean sentiment shift

Pinned commit: 6d1df10eeb4b0fc79eb3d18ba6bd99d9600d8bfa.
"""

import argparse
import json
import os
import shutil

import pandas as pd

MODELS = {
    # key: (notebook, answers json, csv the notebook writes)
    "mistral-small3.1": ("CounterAnalysisMistral.ipynb",
                         "mistral-small3.1_answers_RUN.json",
                         "counterfactual_bias_analysis_mistral.csv"),
    "phi4": ("CounterAnalysisPhi4.ipynb", "phi4_answers_RUN.json", None),
}
PAPER = {
    "mistral-small3.1": {"similarity": 0.88, "neg": 1268, "pos": 1232},
    "phi4": {"similarity": 0.85, "neg": 1289, "pos": 1190, "shift": -0.0045},
}


def first_code_cell(path):
    nb = json.load(open(path, encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            return "".join(c["source"])
    raise ValueError(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    for key, (nb, answers, _) in MODELS.items():
        src = first_code_cell(os.path.join(args.repo, "E. Counterfactual Analysis", nb))
        shutil.copy(os.path.join(args.repo, "D. LLM Responses", answers),
                    os.path.join(args.work, answers))
        before = set(os.listdir(args.work))
        cwd = os.getcwd()
        os.chdir(args.work)
        try:
            exec(compile(src, nb, "exec"), {"__name__": "__main__"})
        finally:
            os.chdir(cwd)
        written = sorted(f for f in set(os.listdir(args.work)) - before if f.endswith(".csv"))
        csv = written[0] if written else MODELS[key][2]
        df = pd.read_csv(os.path.join(args.work, csv))

        print("=" * 72)
        print(f"{key}: their notebook {nb}, output {csv}, {len(df)} rows")
        missing = (df["answer_a"] == "MISSING").sum() + (df["answer_b"] == "MISSING").sum()
        print(f"  MISSING placeholders in pairs: {missing}")
        neg = int((df["sentiment_diff"] < 0).sum())
        pos = int((df["sentiment_diff"] > 0).sum())
        zero = int((df["sentiment_diff"] == 0).sum())
        print(f"  mean similarity      ours {df['similarity'].mean():.4f}   paper {PAPER[key]['similarity']}")
        print(f"  negative / positive  ours {neg} / {pos} (zero {zero})   paper {PAPER[key]['neg']} / {PAPER[key]['pos']}")
        print(f"  mean sentiment shift ours {df['sentiment_diff'].mean():+.4f}   paper {PAPER[key].get('shift', 'not stated')}")
        g = df.groupby(["step_from", "step_to"], sort=False).agg(
            similarity=("similarity", "mean"), shift=("sentiment_diff", "mean"),
            n=("similarity", "size"))
        print("  per pair:")
        for (a, b), r in g.iterrows():
            print(f"    {a[:28]:28} -> {b[:40]:40} sim {r.similarity:.4f}  shift {r['shift']:+.4f}  n {int(r.n)}")


if __name__ == "__main__":
    main()
