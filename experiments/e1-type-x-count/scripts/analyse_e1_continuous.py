#!/usr/bin/env python3
"""
analyse_e1_continuous.py - the read to trust for e1 channels B and C. No GPU.

    python analyse_e1_continuous.py e1_traits_qwen.csv e1_traits_llama.csv
    python analyse_e1_continuous.py e1_choice_qwen.csv

Needs e1_conditions.json in the working directory. The channel is detected from
the CSV header; the two are never mixed in one call because their units differ.

WHY THESE CHANNELS CARRY THE INTERACTION QUESTION
-------------------------------------------------
Channel A is binary. At 600 tasks it detects an interaction of fifteen accuracy
points with 90 percent power and one of five points with 13 percent. These two
produce a continuous value per item, so the same design sees far smaller
effects. Channel A is the behavioural evidence; these are the sensitive
instruments.

WHAT EACH ITEM IS

  channel B   one profile. The value is the mean log probability of six
              positive trait words, averaged over three phrasings. Higher is
              better for the person.
  channel C   one profile-by-question pair. The value is the margin in favour
              of the condition against an otherwise identical person who wrote
              the signal-free message, averaged over both slot orders. Negative
              means the condition loses the choice.

For channel C the NONE cell is zero by construction, so it is not collected and
the interaction is margin(both) - margin(A) - margin(B). A degenerate control
proves nothing, and in Paper 2 including one inflated an error estimate
fivefold.

ORDER OF READING, WHICH MATTERS
-------------------------------
1. the gate (channel C): how much data carries too little mass on the letters
2. coherence (channel B): a condition that raises "intelligent" must not also
   raise "stupid"
3. the floor: two socially neutral details stacked the same way
4. main effects
5. interactions, with the minimum detectable effect beside every estimate
6. order agreement
7. spread across phrasings (channel B)
"""

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared")))

import p3_stats as S      # noqa: E402  bh, wilcoxon, boot_ci, mde


def _conditions_path(name):
    """Find a condition file whether the analyser is run from its own folder,
    from the repository root, or from a directory of downloaded CSVs."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (name,
                 os.path.join(here, name),
                 os.path.join(here, "..", "..", "shared", "conditions", name),
                 os.path.join(here, "..", "shared", "conditions", name)):
        if os.path.exists(cand):
            return cand
    raise SystemExit(
        f"{name} not found. Build it with the matching builder in "
        f"shared/conditions/, or run this from a directory that has it.")



MASS_GATE = 0.5


# --------------------------------------------------------------------------

def detect(path):
    with open(path, encoding="utf-8") as f:
        header = next(csv.reader(f))
    if "trait" in header:
        return "B"
    if "margin_signal" in header:
        return "C"
    raise SystemExit(f"{path}: not an e1 channel B or C file")


def load_traits(paths):
    """value[model][condition][profile] and per-phrasing detail."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    per_phrasing = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    negatives = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                mk, cid, prof = r["model_key"], r["condition"], r["profile"]
                v = float(r["logprob"])
                if v != v:                       # nan
                    continue
                if r["valence"] == "positive":
                    acc[mk][cid][prof].append(v)
                    per_phrasing[mk][cid][r["phrasing"]].append(v)
                else:
                    negatives[mk][cid][prof].append(v)
    value = defaultdict(lambda: defaultdict(dict))
    for mk in acc:
        for cid in acc[mk]:
            for prof, vals in acc[mk][cid].items():
                value[mk][cid][prof] = sum(vals) / len(vals)
    neg_value = defaultdict(lambda: defaultdict(dict))
    for mk in negatives:
        for cid in negatives[mk]:
            for prof, vals in negatives[mk][cid].items():
                neg_value[mk][cid][prof] = sum(vals) / len(vals)
    return value, per_phrasing, neg_value


def load_choice(paths):
    """value[model][condition][profile:question], gated, orders averaged."""
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    mass = defaultdict(list)
    gated_out = defaultdict(int)
    total = defaultdict(int)
    slot = defaultdict(list)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                mk = r["model_key"]
                total[mk] += 1
                m = float(r["mass"])
                mass[mk].append(m)
                slot[mk].append(float(r["logp_A"]) - float(r["logp_B"]))
                if m < MASS_GATE:
                    gated_out[mk] += 1
                    continue
                item = f"{r['profile']}:{r['question']}"
                raw[mk][r["condition"]][item].append(float(r["margin_signal"]))
    value = defaultdict(lambda: defaultdict(dict))
    dropped_unpaired = defaultdict(int)
    for mk in raw:
        for cid in raw[mk]:
            for item, vals in raw[mk][cid].items():
                # both orders must survive the gate, or the slot term does not
                # cancel and the number is not comparable with the others
                if len(vals) < 2:
                    dropped_unpaired[mk] += 1
                    continue
                value[mk][cid][item] = sum(vals) / len(vals)
    return value, mass, gated_out, total, slot, dropped_unpaired


# --------------------------------------------------------------------------

def contrast_values(value_model, ct, channel):
    """Per-item interaction. For channel C the none-cell is zero by design."""
    cells = [ct["both"], ct["single_a"], ct["single_b"]]
    if channel == "B":
        cells.append(ct["none"])
    if any(c not in value_model for c in cells):
        return []
    ids = sorted(set.intersection(*[set(value_model[c]) for c in cells]))
    if channel == "B":
        return [(value_model[ct["both"]][i] - value_model[ct["single_a"]][i])
                - (value_model[ct["single_b"]][i] - value_model[ct["none"]][i])
                for i in ids]
    return [value_model[ct["both"]][i] - value_model[ct["single_a"]][i]
            - value_model[ct["single_b"]][i] for i in ids]


def main(paths):
    channels = {detect(p) for p in paths}
    if len(channels) > 1:
        sys.exit("channel B and channel C files have different units; "
                 "analyse them in separate calls")
    channel = channels.pop()

    with open(_conditions_path("e1_conditions.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    conditions, contrasts = cfg["conditions"], cfg["contrasts"]

    per_phrasing = neg_value = None
    if channel == "B":
        value, per_phrasing, neg_value = load_traits(paths)
        unit = "log prob of a favourable word"
    else:
        (value, mass, gated_out, total, slot,
         dropped_unpaired) = load_choice(paths)
        unit = "logits of choice margin"
    models = sorted(value)

    print("=" * 76)
    print(f"e1 CHANNEL {channel}: {'trait log probability' if channel == 'B' else 'head-to-head choice'}")
    print(f"models: {', '.join(models)}   unit: {unit}")
    print("=" * 76)

    # ---- 0. the gate, channel C only ---------------------------------------
    if channel == "C":
        print("\n" + "-" * 76)
        print("0. THE LETTER-MASS GATE. Reading logP(A) against logP(B) is a")
        print("   judgement only if the model was about to emit a letter.")
        print("-" * 76)
        for m in models:
            mm = mass[m]
            print(f"  {m:9}mean mass {sum(mm) / len(mm):.3f}   "
                  f"gated out {gated_out[m]} of {total[m]} reads "
                  f"({gated_out[m] / total[m]:.1%})   "
                  f"items lost for want of both orders: {dropped_unpaired[m]}")
            sl = slot[m]
            print(f"  {'':9}raw first-slot advantage {sum(sl) / len(sl):+.3f} "
                  f"logits, which is why only within-slot differences are used")
            if gated_out[m] / total[m] > 0.5:
                print(f"  {'':9}MORE THAN HALF THIS MODEL'S READS ARE UNUSABLE."
                      f" Report it as a refusal, not as a result.")

    # ---- 0b. coherence, channel B only -------------------------------------
    if channel == "B":
        print("\n" + "-" * 76)
        print("0. COHERENCE. A condition that lifts the favourable words must")
        print("   not also lift their opposites. This caught a broken")
        print("   instrument twice in Paper 2.")
        print("-" * 76)
        for m in models:
            if m not in neg_value or "NONE" not in neg_value[m]:
                print(f"  {m:9}no coherence sub-run in this file")
                continue
            base_p = value[m].get("NONE", {})
            base_n = neg_value[m].get("NONE", {})
            for cid in sorted(neg_value[m]):
                if cid == "NONE":
                    continue
                ids = sorted(set(value[m].get(cid, {})) & set(base_p)
                             & set(neg_value[m][cid]) & set(base_n))
                if not ids:
                    continue
                dp = sum(value[m][cid][i] - base_p[i] for i in ids) / len(ids)
                dn = sum(neg_value[m][cid][i] - base_n[i] for i in ids) / len(ids)
                verdict = ("coherent" if dp * dn <= 0 else
                           "INCOHERENT, both directions move together")
                print(f"  {m:9}{cid:14}positive {dp:+.4f}  negative {dn:+.4f}"
                      f"   {verdict}")

    # ---- 1. floor ----------------------------------------------------------
    print("\n" + "-" * 76)
    print("1. FLOOR CHECK. Two socially neutral details, stacked the same way.")
    print("-" * 76)
    floor = {}
    ct_floor = next(c for c in contrasts if c["type"] == "floor")
    for m in models:
        d = contrast_values(value[m], ct_floor, channel)
        if not d:
            print(f"  {m:9}no data")
            continue
        mean, lo, hi = S.boot_ci(d)
        t = S.two_tests(d)
        floor[m] = abs(mean)
        print(f"  {m:9}n={len(d):5}  interaction {mean:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  p rank {t['p_rank']:.4f} "
              f"mean {t['p_mean']:.4f}  MDE +/-{S.mde(d):.4f}")

    # ---- 2. main effects ---------------------------------------------------
    print("\n" + "-" * 76)
    print("2. MAIN EFFECTS. Channel B against NONE; channel C is already a")
    print("   margin against NONE, so the value IS the main effect.")
    print("-" * 76)
    singles = [c["id"] for c in conditions
               if c["n_social_signals"] == 1 or c["id"].startswith("ALT")]
    for m in models:
        print(f"\n  {m}")
        for cid in singles:
            if cid not in value[m]:
                continue
            if channel == "B":
                base = value[m].get("NONE", {})
                ids = sorted(set(value[m][cid]) & set(base))
                if not ids:
                    continue
                d = [value[m][cid][i] - base[i] for i in ids]
            else:
                ids = sorted(value[m][cid])
                d = [value[m][cid][i] for i in ids]
            mean, lo, hi = S.boot_ci(d)
            t = S.two_tests(d)
            print(f"    {cid:14}n={len(d):5}  {mean:+9.4f} "
                  f"[{lo:+8.4f},{hi:+8.4f}]  p rank {t['p_rank']:.4g} "
                  f"mean {t['p_mean']:.4g}")

    # ---- 3. interactions ---------------------------------------------------
    print("\n" + "-" * 76)
    print("3. INTERACTIONS. NEGATIVE = super-additive, the pair costs MORE")
    print("   than the sum of its parts. POSITIVE = sub-additive.")
    print("-" * 76)
    rows = []
    for m in models:
        for ct in contrasts:
            d = contrast_values(value[m], ct, channel)
            if not d:
                continue
            mean, lo, hi = S.boot_ci(d)
            row = {"model": m, "name": ct["name"], "type": ct["type"],
                   "n": len(d), "mean": mean, "lo": lo, "hi": hi,
                   "mde": S.mde(d)}
            row.update(S.two_tests(d))
            rows.append(row)
    # floors family-wise per model on either test; contrasts BH on both tests
    S.correct(rows)

    print(f"\n  {'model':9}{'contrast':22}{'type':17}{'n':>5}  "
          f"{'interaction':>12}  {'95% CI':>22}  {'MDE':>8}  {'adj':>9}")
    print("  adj: BH on both tests for contrasts, Holm on either test for the")
    print("  floor, where * means the floor MOVES")
    for r in rows:
        star = " *" if r["claimed"] else ""
        print(f"  {r['model']:9}{r['name']:22}{r['type']:17}{r['n']:5}  "
              f"{r['mean']:+12.4f}  [{r['lo']:+9.4f},{r['hi']:+9.4f}]  "
              f"{r['mde']:8.4f}  {r['bh']:9.5f}{star}")

    # ---- 4. by type --------------------------------------------------------
    print("\n" + "-" * 76)
    print("4. BY SIGNAL TYPE. The thesis: form signals compound, stated do not.")
    print("-" * 76)
    for m in models:
        print(f"\n  {m}")
        for t in ("form_x_form", "stated_x_stated", "cross", "floor"):
            sel = [r for r in rows if r["model"] == m and r["type"] == t]
            if not sel:
                continue
            mean = sum(r["mean"] for r in sel) / len(sel)
            sig = sum(1 for r in sel if r["claimed"])
            vis = sum(1 for r in sel if abs(r["mean"]) > r["mde"])
            print(f"    {t:18}{len(sel)} contrast(s)  mean {mean:+.4f}  "
                  f"{sig} significant after BH, {vis} above MDE")

    # ---- 5. order ----------------------------------------------------------
    print("\n" + "-" * 76)
    print("5. ORDER. Pairs carrying a stated signal ran in both slot orders.")
    print("-" * 76)
    groups = defaultdict(list)
    for r in rows:
        if r["name"][:2] in ("P2", "P3", "P4"):
            groups[(r["model"], r["name"][:2])].append(r)
    for (m, stem), sel in sorted(groups.items()):
        if len(sel) < 2:
            continue
        signs = {1 if r["mean"] > 0 else -1 if r["mean"] < 0 else 0
                 for r in sel}
        verdict = "AGREE" if len(signs) == 1 else "DISAGREE, slot-dependent"
        vals = ", ".join(f"{r['name']}={r['mean']:+.4f}" for r in sel)
        print(f"  {m:9}{stem:5}{verdict:26}{vals}")

    # ---- 6. phrasing spread, channel B -------------------------------------
    if channel == "B" and per_phrasing:
        print("\n" + "-" * 76)
        print("6. PHRASING SPREAD. Three phrasings, kept as separate")
        print("   estimates. A conclusion that depends on the phrasing is a")
        print("   fact about the phrasing.")
        print("-" * 76)
        for m in models:
            print(f"\n  {m}")
            for cid in ["NONE", "F01", "F04", "P1_F01F04",
                        "S01_s1", "S08_s2", "P2_S01S08"]:
                if cid not in per_phrasing[m]:
                    continue
                means = {pid: sum(v) / len(v)
                         for pid, v in per_phrasing[m][cid].items()}
                spread = max(means.values()) - min(means.values())
                cells = "  ".join(f"{k}={v:.3f}" for k, v in sorted(means.items()))
                print(f"    {cid:14}spread {spread:.4f}   {cells}")

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 76)
    print("VERDICT on this channel, against the registered expectations")
    print("=" * 76)
    for m in models:
        ff = [r for r in rows if r["model"] == m and r["type"] == "form_x_form"]
        ss = [r for r in rows if r["model"] == m
              and r["type"] == "stated_x_stated"]
        if not ff or not ss:
            continue
        ff_super = all(r["claimed"] and r["mean"] < 0
                       and abs(r["mean"]) > r["mde"] for r in ff)
        ss_null = not any(r["claimed"] for r in ss)
        # the floor verdict is the TEST, not a ratio of magnitudes
        fl_row = next((r for r in rows if r["model"] == m
                       and r["type"] == "floor"), None)
        floor_ok = not (fl_row and fl_row["claimed"])
        parts = ["form x form super-additive" if ff_super
                 else "form x form NOT super-additive",
                 "stated x stated null" if ss_null
                 else "stated x stated NOT null",
                 "floor clean" if floor_ok else "FLOOR CONTAMINATED"]
        print(f"  {m:9}{'SUPPORTS' if (ff_super and ss_null and floor_ok) else 'does not support'}"
              f" the thesis here: {'; '.join(parts)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        sys.exit("usage: analyse_e1_continuous.py <csv> [<csv> ...]")
    missing = [a for a in args if not os.path.exists(a)]
    if missing:
        sys.exit(f"missing: {', '.join(missing)}")
    main(args)
