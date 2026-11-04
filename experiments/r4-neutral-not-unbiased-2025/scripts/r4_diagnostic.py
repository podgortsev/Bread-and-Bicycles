#!/usr/bin/env python3
"""
r4_diagnostic.py - Addendum 9: step-one similarity (original -> gender) on the
same first 30 scenarios in three sets of answers: theirs, our four-parallel run,
and our one-at-a-time run. Their instrument (all-MiniLM-L6-v2 cosine).

    python r4_diagnostic.py --theirs <phi4_answers_RUN.json> --parallel <answers_phi4.json>
                            --serial <answers_phi4_serial30.json>
"""

import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer

THEIRS_ARMS = ("Original", "Name 1 Female, Name 2 Male")
OURS_ARMS = ("original", "gender")


def pairs(answers, arms, scenarios):
    out = []
    for s in scenarios:
        a, b = answers[s][arms[0]], answers[s][arms[1]]
        for q in sorted(set(a) & set(b)):
            out.append((a[q], b[q]))
    return out


def mean_sim(model, pairs_):
    ea = model.encode([p[0] for p in pairs_], batch_size=64, normalize_embeddings=True)
    eb = model.encode([p[1] for p in pairs_], batch_size=64, normalize_embeddings=True)
    sims = (ea * eb).sum(axis=1)
    return float(sims.mean()), float(sims.std(ddof=1) / np.sqrt(len(sims))), len(sims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theirs", required=True)
    ap.add_argument("--parallel", required=True)
    ap.add_argument("--serial", required=True)
    args = ap.parse_args()
    theirs = json.load(open(args.theirs, encoding="utf-8"))
    par = json.load(open(args.parallel, encoding="utf-8"))
    ser = json.load(open(args.serial, encoding="utf-8"))
    scenarios = [f"Scenario_{i}" for i in range(1, 31)]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    t, tse, tn = mean_sim(model, pairs(theirs, THEIRS_ARMS, scenarios))
    p, pse, pn = mean_sim(model, pairs(par, OURS_ARMS, scenarios))
    s, sse, sn = mean_sim(model, pairs(ser, OURS_ARMS, scenarios))
    print("=" * 72)
    print("R4 DIAGNOSTIC (Addendum 9): original -> gender, scenarios 1 to 30")
    print("=" * 72)
    print(f"  theirs            {t:.4f}  (se {tse:.4f}, n {tn})")
    print(f"  ours, 4 parallel  {p:.4f}  (se {pse:.4f}, n {pn})   diff {p - t:+.4f}")
    print(f"  ours, 1 at a time {s:.4f}  (se {sse:.4f}, n {sn})   diff {s - t:+.4f}")
    serial_ok = abs(s - t) <= 0.02
    parallel_ok = abs(p - t) <= 0.02
    print(f"\n  serial within 0.02 of theirs: {serial_ok};  parallel within 0.02: {parallel_ok}")
    if serial_ok and not parallel_ok:
        print("  LICENSED: repeat the full six-arm run one request at a time")
    else:
        print("  NOT LICENSED: R4 stays as reported (fidelity failed)")
    print("=" * 72)


if __name__ == "__main__":
    main()
