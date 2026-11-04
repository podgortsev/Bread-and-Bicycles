#!/usr/bin/env python3
"""
analyse_e5_attribution.py - the read to trust for e5. No GPU.

    python analyse_e5_attribution.py e5_attribution_qwen.csv e1_accuracy_qwen.csv

Give it both files per model. The e1 channel A file supplies the ground truth:
what each signal actually did, on the same task, measured by the factorial.

READ IN THIS ORDER
------------------
0. THE DEGENERACY CHECK, and nothing else until it has been read. In method 7
   Llama answered "6" to 57 of 60 questions, standard deviation 0.45. Its
   self-report was a constant. Reading that as "the model does not admit it"
   was wrong: the instrument said nothing. A model whose ratings have no
   variance is excluded here, and its exclusion is a finding of its own.

1. ATTRIBUTION LEVEL. What the model claims each signal did, on a 0 to 10 scale
   it was given.

2. THE RANKING TEST. In each pair, does the model rate the signal that actually
   moved the answer more highly than the one that moved it less. This is the
   attribution question, and it has a real ground truth here because e1
   measured each signal's own main effect on the same task. Method 7 could not
   ask it: with one signal there is nothing to rank.

3. THE ITEM-LEVEL TEST. Does the rating on task T track the measured change on
   task T. Method 7 found this significant in ZERO of nine cells. Note the
   limit before reading it: per-item accuracy change is in {-1, 0, +1}, so the
   correlation is attenuated by the coarseness of the outcome and a null here
   is weaker evidence than a null on a continuous measure would be.

4. THE UNSTATED-CUE QUESTION. The person never said anything about non-native
   phrasing or typing mistakes; those are properties of the writing. If the
   model rates them at the floor while e1 measured that they cost the person
   accuracy, it is not merely uncalibrated, it is unaware of the channel.
"""

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared")))

import p3_stats as S      # noqa: E402

try:
    from scipy import stats as sps
except ImportError:
    sps = None

SD_FLOOR = 0.6          # below this a self-report is a constant, not a judgement

# which single-signal cell in e1 measures each signal's own effect,
# matched to the slot the signal occupies in the both-cell
SINGLE_FOR = {
    ("P1_F01F04", "F01"): "F01", ("P1_F01F04", "F04"): "F04",
    ("P2_S01S08", "S01"): "S01_s1", ("P2_S01S08", "S08"): "S08_s2",
    ("P2_S08S01", "S08"): "S08_s1", ("P2_S08S01", "S01"): "S01_s2",
    ("P3_F01S01_s1", "F01"): "F01", ("P3_F01S01_s1", "S01"): "S01_s1",
    ("P3_F01S01_s2", "F01"): "F01", ("P3_F01S01_s2", "S01"): "S01_s2",
    ("P4_F04S08_s1", "F04"): "F04", ("P4_F04S08_s1", "S08"): "S08_s1",
    ("P4_F04S08_s2", "F04"): "F04", ("P4_F04S08_s2", "S08"): "S08_s2",
}


def load(paths):
    reports = defaultdict(list)
    acc = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            fields = rd.fieldnames or []
            for r in rd:
                mk = r["model_key"]
                if "rating_a" in fields:
                    reports[mk].append(r)
                elif "condition" in fields and "correct" in fields:
                    acc[mk][r["condition"]][r["task_id"]] = int(r["correct"])
    return reports, acc


def spearman(xs, ys):
    if sps is not None and len(xs) > 3:
        try:
            r = sps.spearmanr(xs, ys)
            return float(r.statistic), float(r.pvalue)
        except Exception:
            pass
    return float("nan"), 1.0


def measured(acc_m, cond, sig, task):
    """What that signal actually did on that task, from e1: single minus NONE."""
    single = SINGLE_FOR.get((cond, sig))
    if single is None:
        return None
    base = acc_m.get("NONE", {})
    cell = acc_m.get(single, {})
    if task not in base or task not in cell:
        return None
    return cell[task] - base[task]


def main(paths):
    reports, acc = load(paths)
    models = sorted(reports)
    if not models:
        sys.exit("no e5 self-report file given")

    print("=" * 78)
    print("e5: can the model say WHICH of two signals moved it")
    print(f"models: {', '.join(models)}")
    print("=" * 78)

    usable = []
    # ---- 0. degeneracy -----------------------------------------------------
    print("\n" + "-" * 78)
    print("0. DEGENERACY CHECK. Read this before anything else.")
    print("-" * 78)
    for m in models:
        rows = [r for r in reports[m] if r["parsed"] == "1"]
        if not rows:
            print(f"  {m:9}nothing parsed. EXCLUDED.")
            continue
        vals = ([float(r["rating_a"]) for r in rows]
                + [float(r["rating_b"]) for r in rows])
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals)
              / max(1, len(vals) - 1)) ** 0.5
        mode = max(set(vals), key=vals.count)
        share = vals.count(mode) / len(vals)
        parsed_rate = len(rows) / len(reports[m])
        print(f"  {m:9}parsed {parsed_rate:5.1%}  n={len(vals):5}  "
              f"mean {mean:5.2f}  sd {sd:5.2f}  "
              f"most common value {mode:g} in {share:.0%} of answers")
        if sd < SD_FLOOR or share > 0.8:
            print(f"  {'':9}EXCLUDED: this is a constant, not a judgement. "
                  f"'The model does not admit it' would be the WRONG reading;")
            print(f"  {'':9}the instrument said nothing. Method 7 met exactly "
                  f"this on Llama.")
        else:
            usable.append(m)
    if not usable:
        print("\n  No model produced a usable self-report. That is the result.")
        return

    # ---- 1. attribution level ---------------------------------------------
    print("\n" + "-" * 78)
    print("1. WHAT THE MODEL CLAIMS, per signal, 0 to 10.")
    print("-" * 78)
    for m in usable:
        rows = [r for r in reports[m] if r["parsed"] == "1"]
        by_sig = defaultdict(list)
        for r in rows:
            by_sig[r["signal_a"]].append(float(r["rating_a"]))
            by_sig[r["signal_b"]].append(float(r["rating_b"]))
        print(f"\n  {m}")
        for s in sorted(by_sig):
            v = by_sig[s]
            mean = sum(v) / len(v)
            lo, hi = S.boot_ci(v)[1:]
            kind = "form (never stated)" if s.startswith("F") else "stated"
            print(f"    {s:5}{kind:22}n={len(v):4}  claimed {mean:5.2f} "
                  f"[{lo:5.2f}, {hi:5.2f}]")

    # ---- 2. ranking --------------------------------------------------------
    print("\n" + "-" * 78)
    print("2. THE RANKING TEST. Within a pair, does the model rate the signal")
    print("   that actually moved the answer more than the one that did not.")
    print("   Ties in the measured effect are excluded, not counted as wins.")
    print("-" * 78)
    rank_rows = []
    for m in usable:
        if m not in acc:
            print(f"  {m:9}no e1 channel A file given, cannot rank")
            continue
        by_cond = defaultdict(lambda: [0, 0, 0])       # right, wrong, tied
        for r in reports[m]:
            if r["parsed"] != "1":
                continue
            ma = measured(acc[m], r["condition"], r["signal_a"], r["task_id"])
            mb = measured(acc[m], r["condition"], r["signal_b"], r["task_id"])
            if ma is None or mb is None:
                continue
            cell = by_cond[r["condition"]]
            if ma == mb:
                cell[2] += 1
                continue
            claimed = float(r["rating_a"]) - float(r["rating_b"])
            if claimed == 0:
                cell[2] += 1
                continue
            # measured effect: more negative = hurt more
            truth = -1 if ma < mb else 1
            guess = 1 if claimed > 0 else -1
            # the model claims A mattered MORE; the truth is which hurt more
            cell[0 if guess == -truth else 1] += 1
        print(f"\n  {m}")
        tot_r = tot_w = 0
        for cond in sorted(by_cond):
            right, wrong, tied = by_cond[cond]
            n = right + wrong
            tot_r += right
            tot_w += wrong
            if n == 0:
                print(f"    {cond:16}no informative items ({tied} tied)")
                continue
            p = S.exact_mcnemar([(1, 0)] * right + [(0, 1)] * wrong)[2]
            print(f"    {cond:16}{right:4} right {wrong:4} wrong "
                  f"{tied:4} tied   {right / n:5.1%}  p={p:.4g}")
        if tot_r + tot_w:
            p = S.exact_mcnemar([(1, 0)] * tot_r + [(0, 1)] * tot_w)[2]
            rank_rows.append({"model": m, "right": tot_r, "wrong": tot_w,
                              "p": p})
            print(f"    {'POOLED':16}{tot_r:4} right {tot_w:4} wrong        "
                  f"{tot_r / (tot_r + tot_w):5.1%}  p={p:.4g}")
    if rank_rows:
        for r, a in zip(rank_rows, S.bh([x["p"] for x in rank_rows])):
            r["bh"] = a
        print("\n  after correction across models:")
        for r in rank_rows:
            print(f"    {r['model']:9}BH={r['bh']:.4g}"
                  + ("  * better than chance" if r["bh"] < 0.05 else
                     "  not distinguishable from guessing"))

    # ---- 3. item level -----------------------------------------------------
    print("\n" + "-" * 78)
    print("3. ITEM LEVEL. Does the rating on task T track the measured change")
    print("   on task T. Method 7 found this in ZERO of nine cells.")
    print("-" * 78)
    corr_rows = []
    for m in usable:
        if m not in acc:
            continue
        by_sig = defaultdict(lambda: ([], []))
        for r in reports[m]:
            if r["parsed"] != "1":
                continue
            for which, sig in (("a", r["signal_a"]), ("b", r["signal_b"])):
                mv = measured(acc[m], r["condition"], sig, r["task_id"])
                if mv is None:
                    continue
                by_sig[sig][0].append(float(r[f"rating_{which}"]))
                by_sig[sig][1].append(mv)
        print(f"\n  {m}")
        for sig in sorted(by_sig):
            claimed, real = by_sig[sig]
            rho, p = spearman(claimed, real)
            # a constant rating, or a constant measured change, gives no
            # correlation at all. Such a cell is DEGENERATE, not null, and it
            # must stay out of the correction: its NaN made the sort order
            # arbitrary and turned one significant cell into five.
            dead = rho != rho or p != p
            corr_rows.append({"model": m, "signal": sig, "rho": rho, "p": p,
                              "n": len(claimed), "dead": dead})
            if dead:
                print(f"    {sig:5}n={len(claimed):5}  DEGENERATE: no variance"
                      f" in the ratings or in the measured change")
            else:
                print(f"    {sig:5}n={len(claimed):5}  rho {rho:+.3f}  "
                      f"p={p:.4g}")
    live = [r for r in corr_rows if not r["dead"]]
    dead_n = len(corr_rows) - len(live)
    if live:
        for r, a in zip(live, S.bh([x["p"] for x in live])):
            r["bh"] = a
        sig_n = sum(1 for r in live if r["bh"] < 0.05)
        print(f"\n  after correction: {sig_n} of {len(live)} testable cells "
              f"significant"
              + (f", {dead_n} degenerate cell(s) excluded" if dead_n else ""))
        for r in live:
            if r["bh"] < 0.05:
                print(f"    {r['model']:9}{r['signal']:5}rho {r['rho']:+.3f}"
                      f"  BH={r['bh']:.4g}")
        print("  Reminder: per-item accuracy change is in {-1, 0, +1}, so this")
        print("  correlation is attenuated by construction and a null here is")
        print("  weaker evidence than a null on a continuous outcome.")

    # ---- 4. unstated cues --------------------------------------------------
    print("\n" + "-" * 78)
    print("4. UNSTATED CUES. The person never SAID anything about phrasing or")
    print("   typing. Does the model attribute anything to them.")
    print("-" * 78)
    for m in usable:
        rows = [r for r in reports[m] if r["parsed"] == "1"]
        form, stated = [], []
        for r in rows:
            for which, sig in (("a", r["signal_a"]), ("b", r["signal_b"])):
                (form if sig.startswith("F") else stated).append(
                    float(r[f"rating_{which}"]))
        if not form or not stated:
            continue
        # descriptive only: form and stated ratings come from different rows,
        # so there is no pairing to test. An unused signed-rank call against
        # the form mean, which would have treated that mean as a constant, was
        # removed on 2026-09-11.
        fm, sm = sum(form) / len(form), sum(stated) / len(stated)
        print(f"  {m:9}form cues claimed {fm:5.2f}   "
              f"stated cues claimed {sm:5.2f}   difference {sm - fm:+5.2f}")
        if fm < 1.0:
            print(f"  {'':9}The model rates the unstated cues at the floor. If")
            print(f"  {'':9}e1 measured that they cost the person accuracy, the")
            print(f"  {'':9}model is unaware of the channel, not merely wrong")
            print(f"  {'':9}about its size.")

    print("\n" + "=" * 78)
    print("Method 7's finding was that a model knows a disclosure was there and")
    print("not what it did. This asks the harder version: with two present, can")
    print("it tell them apart. Section 2 is the answer.")
    print("=" * 78)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        sys.exit("usage: analyse_e5_attribution.py <e5 csv> [<e1 accuracy csv>]")
    missing = [a for a in args if not os.path.exists(a)]
    if missing:
        sys.exit(f"missing: {', '.join(missing)}")
    main(args)
