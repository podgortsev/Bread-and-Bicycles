#!/usr/bin/env python3
"""
r4_analyse.py - the R4 floor for "Neutral Is Not Unbiased": does adding age to
a gendered scenario move the answers more than adding a neutral word in the
same slot, or than asking the same prompt again?

    python r4_analyse.py --answers answers_phi4.json --model-key phi4
                         [--floor-tool <products/interaction-floor-test>]

Preregistration Addendum 6.

INSTRUMENT (theirs, from the first cell of their CounterAnalysis notebooks)
    similarity  cosine of all-MiniLM-L6-v2 embeddings of answer a and answer b
    shift       TextBlob polarity(b) - polarity(a)
Paired by scenario and question, as theirs are.

STEPS
    age      gender -> gender_age        the step their claim is about
    n1, n2   gender -> gender_n1 / _n2   the same step with a neutral word
    again    gender -> gender_again      the same prompt sampled twice
    first    original -> gender          their first step, for fidelity

DECISION (primary statistic: sentiment shift; similarity reported the same way)
    d_k = step age - step n_k, per item, k = 1, 2
    both tests on each d_k: Wilcoxon signed rank and sign-flip permutation
    (the implementations in products/interaction-floor-test)

    survives the floor   for both k: both tests p < 0.05 and mean d_k in the
                         claimed direction (shift more negative; similarity
                         lower). An intersection-union rule: every component
                         must reject, so no multiplicity correction is applied
    floor explains it    for some k: neither test rejects, or mean d_k is not
                         in the claimed direction
    partial              otherwise

RESOLUTION
    every d is printed with its 95 percent half-width ("smallest visible"). If
    the social step's published size is below it, a FLOOR EXPLAINS verdict also
    means the instrument cannot separate the steps at this sample size, and is
    reported in those words.

FIDELITY OF THE NEW GENERATION (against their released answers, Phi-4)
    mean similarity of step first and step age within 0.02 of theirs
    (0.8467 and 0.8398)
"""

import argparse
import json
import os
import sys

import numpy as np

THEIRS = {"phi4": {"first": 0.8467, "age": 0.8398},
          "mistral-small3.1": {"first": 0.8720, "age": 0.8719}}
STEPS = {
    "age": ("gender", "gender_age"),
    "n1": ("gender", "gender_n1"),
    "n2": ("gender", "gender_n2"),
    "again": ("gender", "gender_again"),
    "first": ("original", "gender"),
}
ALPHA = 0.05


def load_floor_tool(path):
    sys.path.insert(0, path)
    import floor_test  # noqa: E402
    return floor_test


def measure(answers, encoder, polarity):
    """Per step: arrays of similarity and shift, aligned on (scenario, question)."""
    keys = []
    for skey, arms in answers.items():
        if not all(a in arms for s in STEPS.values() for a in s):
            continue
        qs = set.intersection(*[set(arms[a]) for s in STEPS.values() for a in s])
        keys += [(skey, q) for q in sorted(qs)]
    texts = {}
    for skey, q in keys:
        for arm in {a for s in STEPS.values() for a in s}:
            texts[(skey, arm, q)] = answers[skey][arm][q]
    order = list(texts)
    emb = encoder([texts[k] for k in order])
    emb = {k: e / np.linalg.norm(e) for k, e in zip(order, emb)}
    pol = {k: polarity(texts[k]) for k in order}
    out = {}
    for name, (a, b) in STEPS.items():
        sim = np.array([float(emb[(s, a, q)] @ emb[(s, b, q)]) for s, q in keys])
        shift = np.array([pol[(s, b, q)] - pol[(s, a, q)] for s, q in keys])
        out[name] = {"similarity": sim, "shift": shift}
    return out, len(keys)


def decide(steps, ft, stat):
    """Addendum 6 rule for one statistic."""
    rows = {}
    for k in ("n1", "n2"):
        d = list(steps["age"][stat] - steps[k][stat])
        rows[k] = {"mean": float(np.mean(d)), "se": float(np.std(d, ddof=1) / np.sqrt(len(d))),
                   "p_rank": ft.signed_rank_p(d), "p_mean": ft.sign_flip_p(d),
                   "mde": ft.mde(d)}
    claimed = lambda r: r["mean"] < 0  # shift more negative, similarity lower
    both = lambda r: r["p_rank"] < ALPHA and r["p_mean"] < ALPHA
    neither = lambda r: r["p_rank"] >= ALPHA and r["p_mean"] >= ALPHA
    if all(both(r) and claimed(r) for r in rows.values()):
        verdict = "CLAIM SURVIVES THE FLOOR"
    elif any(neither(r) or not claimed(r) for r in rows.values()):
        verdict = "THE FLOOR EXPLAINS THE CLAIM"
    else:
        verdict = "PARTIAL"
    return rows, verdict


def pair_test(ft, x, y):
    d = list(x - y)
    return float(np.mean(d)), ft.signed_rank_p(d), ft.sign_flip_p(d)


def report(steps, n, ft, model_key):
    print("=" * 78)
    print(f"R4 FLOOR: Neutral Is Not Unbiased, {model_key}, {n} scenario-question items")
    print("=" * 78)
    print("\nMEANS PER STEP")
    for name, (a, b) in STEPS.items():
        s = steps[name]
        print(f"  {name:6} {a:>13} -> {b:13}  similarity {s['similarity'].mean():.4f}"
              f"   shift {s['shift'].mean():+.4f}")
    if model_key in THEIRS:
        print("\nFIDELITY OF THE NEW GENERATION (similarity within 0.02 of their answers)")
        ok = True
        for name, t in THEIRS[model_key].items():
            o = steps[name]["similarity"].mean()
            good = abs(o - t) <= 0.02
            ok &= good
            print(f"  {name:6} ours {o:.4f}  theirs {t:.4f}  {'pass' if good else 'FAIL'}")
        print(f"  {'PASS' if ok else 'FAIL: new generation not shown equivalent'}")
    verdicts = {}
    for stat in ("shift", "similarity"):
        rows, verdict = decide(steps, ft, stat)
        verdicts[stat] = verdict
        tag = "PRIMARY" if stat == "shift" else "secondary"
        print(f"\n{tag}: {stat}, d = step age - step neutral, per item")
        for k, r in rows.items():
            print(f"  vs {k}: mean d {r['mean']:+.5f} (se {r['se']:.5f})  "
                  f"p signed-rank {r['p_rank']:.4f}  p sign-flip {r['p_mean']:.4f}  "
                  f"smallest visible {r['mde']:.5f}")
        print(f"  verdict: {verdict}")
    print("\nSECONDARY: is each step distinguishable from sampling the same prompt twice?")
    for name in ("age", "n1", "n2"):
        for stat in ("shift", "similarity"):
            m, pr, pm = pair_test(ft, steps[name][stat], steps["again"][stat])
            print(f"  {name:4} vs again  {stat:10} mean d {m:+.5f}  p rank {pr:.4f}  p flip {pm:.4f}")
    print(f"\nDECISION (primary, sentiment shift): {verdicts['shift']}")
    print("=" * 78)
    return verdicts


def real_instrument():
    from sentence_transformers import SentenceTransformer
    from textblob import TextBlob
    model = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda texts: model.encode(texts, batch_size=64)
    return enc, lambda t: TextBlob(t).sentiment.polarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--model-key", required=True)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--floor-tool", default=os.path.join(
        here, "..", "..", "..", "..", "..", "products", "interaction-floor-test"))
    args = ap.parse_args()
    ft = load_floor_tool(os.path.abspath(args.floor_tool))
    answers = json.load(open(args.answers, encoding="utf-8"))
    enc, pol = real_instrument()
    steps, n = measure(answers, enc, pol)
    report(steps, n, ft, args.model_key)


if __name__ == "__main__":
    main()
