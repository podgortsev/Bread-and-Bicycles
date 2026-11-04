#!/usr/bin/env python3
"""
validate_e1_channels.py - offline check of e1 channels B and C. No GPU, no
torch, no network.

    python validate_e1_channels.py

Channel A has its own validator, validate_e1_accuracy.py. This one covers the
two continuous channels and the analysis they share.

WHAT IT CHECKS
--------------
A. job construction: the right number of prompts, NONE excluded from channel C,
   both slot orders always present, the lead-in actually reaching the prompt
B. synthetic channel B with a known interaction on the form pair and none on the
   stated pair, written as a real CSV and read back through the real analyser
C. synthetic channel C, same, plus the zero-by-construction NONE cell
D. the letter-mass gate: low-mass reads must be removed, and an item that loses
   one of its two slot orders must be dropped rather than half-counted
E. the coherence check: an instrument where the favourable and unfavourable
   words move together must be flagged
F. false positives on both channels, measured over many seeds
G. power at plausible effect sizes, so a null on these channels can be read
"""

import contextlib
import csv
import io
import json
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = os.path.abspath(os.path.join(HERE, "..", "..", "shared", "conditions"))
sys.path.insert(0, HERE)
sys.path.insert(0, CONDS)

import analyse_e1_accuracy as S        # noqa: E402
import analyse_e1_continuous as AC     # noqa: E402
import build_e1_conditions as B        # noqa: E402
import run_e1_choice as C             # noqa: E402
import run_e1_traits as T             # noqa: E402

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def close(a, b, tol):
    return abs(a - b) <= tol


CONDITIONS = B.build()
CONTRASTS = [{"name": n, "type": t, "both": c[0], "single_a": c[1],
              "single_b": c[2], "none": c[3]} for n, t, c in B.CONTRASTS]
IDS = [c["id"] for c in CONDITIONS]
PROFILES = [f"{i} years in a field, does a job, leads something" for i in range(20)]

# ==========================================================================
print("=" * 74)
print("A. JOB CONSTRUCTION")
print("=" * 74)

jobs_b = T.build_jobs(CONDITIONS, PROFILES)
want_b = (len(CONDITIONS) * len(PROFILES) * len(T.PHRASINGS) * len(T.POSITIVE)
          + len(T.COHERENCE_CONDITIONS) * len(PROFILES) * len(T.NEGATIVE))
ok("channel B job count is exactly the design", len(jobs_b) == want_b,
   f"{len(jobs_b)} vs {want_b}")
ok("channel B scores positives on every condition",
   len({j[1] for j in jobs_b if j[4] == "positive"}) == len(CONDITIONS))
ok("channel B scores negatives only in the coherence sub-run",
   {j[1] for j in jobs_b if j[4] == "negative"} == set(T.COHERENCE_CONDITIONS))
ok("channel B prompt carries the condition's lead-in",
   all(dict((c["id"], c["lead_in"]) for c in CONDITIONS)[j[1]] in j[5]
       for j in jobs_b[:400]))

jobs_c = C.build_jobs(CONDITIONS, PROFILES)
want_c = (len(CONDITIONS) - 1) * len(PROFILES) * len(C.QUESTIONS) * 2
ok("channel C job count is exactly the design", len(jobs_c) == want_c,
   f"{len(jobs_c)} vs {want_c}")
ok("channel C never runs NONE against itself",
   "NONE" not in {j[0] for j in jobs_c},
   "a degenerate control is zero by construction and proves nothing")
orders = {}
for cid, prof, qid, order, _p in jobs_c:
    orders.setdefault((cid, prof, qid), set()).add(order)
ok("channel C runs BOTH slot orders for every item",
   all(v == {"sig_first", "sig_second"} for v in orders.values()))
none_lead = next(c["lead_in"] for c in CONDITIONS if c["id"] == "NONE")
ok("channel C prompt contains both messages",
   all(none_lead in j[4] for j in jobs_c[:200]))

# ==========================================================================
# synthetic writers
# ==========================================================================

N_PROF = 100
N_Q = len(C.QUESTIONS)

EFF_B = {"F01": -1.0, "F04": -1.0, "S01_s1": -1.0, "S01_s2": -1.0,
         "S08_s1": -1.0, "S08_s2": -1.0, "ALT1": 0.0, "ALT2_ONLY": 0.0,
         "P1_F01F04": -2.0, "P2_S01S08": -2.0, "P2_S08S01": -2.0,
         "P3_F01S01_s1": -2.0, "P3_F01S01_s2": -2.0,
         "P4_F04S08_s1": -2.0, "P4_F04S08_s2": -2.0, "ALT_BOTH": 0.0}

# channel C values are margins AGAINST NONE, so NONE is zero and the singles
# already carry their main effect
EFF_C = dict(EFF_B)
EFF_C["NONE"] = 0.0


def extra_from(interactions):
    extra = {}
    for ct in CONTRASTS:
        v = interactions.get(ct["name"], 0.0)
        if v:
            extra[ct["both"]] = extra.get(ct["both"], 0.0) + v
    return extra


def write_traits_csv(path, interactions, seed, noise=0.35,
                     incoherent=(), model="qwen"):
    """Trait-level rows, exactly the shape run_e1_traits.py writes."""
    rng = random.Random(seed)
    extra = extra_from(interactions)
    prof_base = {f"P{i + 1:03d}": rng.gauss(-4.0, 0.6) for i in range(N_PROF)}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "model_key", "phrasing", "condition", "profile",
                    "trait", "valence", "logprob"])
        for pid, _tpl in T.PHRASINGS:
            for cid in IDS:
                shift = EFF_B.get(cid, 0.0) + extra.get(cid, 0.0)
                for prof, base in prof_base.items():
                    for tr in T.POSITIVE:
                        w.writerow([model, model, pid, cid, prof, tr,
                                    "positive",
                                    f"{base + shift + rng.gauss(0, noise):.6f}"])
        cp = T.COHERENCE_PHRASING
        for cid in T.COHERENCE_CONDITIONS:
            shift = EFF_B.get(cid, 0.0) + extra.get(cid, 0.0)
            # a coherent instrument moves the negatives the OTHER way
            neg_shift = -shift if cid not in incoherent else shift
            for prof, base in prof_base.items():
                for tr in T.NEGATIVE:
                    w.writerow([model, model, cp, cid, prof, tr, "negative",
                                f"{base - 3.0 + neg_shift + rng.gauss(0, noise):.6f}"])


def write_choice_csv(path, interactions, seed, noise=1.2, mass=0.95,
                     low_mass_conditions=(), drop_one_order=(), model="qwen"):
    rng = random.Random(seed)
    extra = extra_from(interactions)
    items = [(f"P{i + 1:03d}", q) for i in range(N_PROF) for q, _ in C.QUESTIONS]
    item_base = {it: rng.gauss(0, 0.8) for it in items}
    slot_bias = 6.0                     # a large first-slot advantage, as real
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "model_key", "condition", "profile", "question",
                    "order", "logp_A", "logp_B", "mass", "margin_signal"])
        for cid in IDS:
            if cid == "NONE":
                continue
            shift = EFF_C.get(cid, 0.0) + extra.get(cid, 0.0)
            for prof, q in items:
                for order in ("sig_first", "sig_second"):
                    if cid in drop_one_order and order == "sig_second":
                        continue
                    margin = shift + item_base[(prof, q)] + rng.gauss(0, noise)
                    if order == "sig_first":
                        la, lb = slot_bias + margin, 0.0
                    else:
                        la, lb = slot_bias, margin
                    m = 0.12 if cid in low_mass_conditions else mass
                    w.writerow([model, model, cid, prof, q, order,
                                f"{la:.6f}", f"{lb:.6f}", f"{m:.6f}",
                                f"{margin:.6f}"])


def run_analyser(paths):
    """Runs the real analyser and returns its printed output."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        AC.main(paths)
    return buf.getvalue()


def contrasts_from(value_model, channel):
    out = {}
    for ct in CONTRASTS:
        d = AC.contrast_values(value_model, ct, channel)
        if not d:
            continue
        p, _ = S.wilcoxon(d)
        out[ct["name"]] = {"type": ct["type"], "mean": sum(d) / len(d),
                           "p": p, "mde": S.mde(d), "n": len(d)}
    names = list(out)
    for n, a in zip(names, S.bh([out[n]["p"] for n in names])):
        out[n]["bh"] = a
    return out


tmp = tempfile.mkdtemp(prefix="e1val_")
shutil.copy(os.path.join(CONDS, "e1_conditions.json"),
            os.path.join(tmp, "e1_conditions.json"))
cwd = os.getcwd()
os.chdir(tmp)

try:
    # ======================================================================
    print()
    print("=" * 74)
    print("B. CHANNEL B, synthetic, through the real analyser")
    print("=" * 74)
    print("   injected: form x form -0.60 logits, stated x stated 0.00")

    p = "e1_traits_qwen.csv"
    write_traits_csv(p, {"P1_FORMxFORM": -0.60}, seed=3)
    out = run_analyser([p])
    val, _ph, _neg = AC.load_traits([p])
    res = contrasts_from(val["qwen"], "B")

    ff = res["P1_FORMxFORM"]
    ok("channel B recovers the injected form x form interaction",
       close(ff["mean"], -0.60, 0.15), f"got {ff['mean']:+.4f}")
    ok("channel B finds it significant after BH", ff["bh"] < 0.05,
       f"BH={ff['bh']:.2e}")
    ok("channel B: stated x stated stays null",
       all(res[k]["bh"] >= 0.05 for k in res
           if res[k]["type"] == "stated_x_stated"))
    ok("channel B: floor stays null", res["FLOOR_ALTxALT"]["bh"] >= 0.05)
    ok("analyser prints the verdict line", "SUPPORTS the thesis here" in out)
    ok("analyser prints the coherence section", "COHERENCE" in out)
    ok("analyser prints the phrasing spread", "PHRASING SPREAD" in out)

    # ======================================================================
    print()
    print("=" * 74)
    print("C. CHANNEL C, synthetic, through the real analyser")
    print("=" * 74)
    print("   injected: form x form -1.20 logits, stated x stated 0.00,")
    print("   plus a first-slot advantage of 6 logits that must cancel")

    p = "e1_choice_qwen.csv"
    write_choice_csv(p, {"P1_FORMxFORM": -1.20}, seed=5)
    out = run_analyser([p])
    val, mass, gated, total, slot, unpaired = AC.load_choice([p])
    res = contrasts_from(val["qwen"], "C")

    ff = res["P1_FORMxFORM"]
    ok("channel C recovers the injected form x form interaction",
       close(ff["mean"], -1.20, 0.35), f"got {ff['mean']:+.4f}")
    ok("channel C finds it significant after BH", ff["bh"] < 0.05,
       f"BH={ff['bh']:.2e}")
    ok("channel C: stated x stated stays null",
       all(res[k]["bh"] >= 0.05 for k in res
           if res[k]["type"] == "stated_x_stated"))
    ok("channel C: the slot bias does not leak into the interaction",
       abs(ff["mean"] + 1.20) < 0.35,
       "a 6-logit first-slot advantage was present in the raw reads")
    ok("analyser reports the raw first-slot advantage",
       "raw first-slot advantage" in out)

    # single-signal main effects come back as margins against NONE
    ids = sorted(val["qwen"]["S01_s1"])
    m_s01 = sum(val["qwen"]["S01_s1"][i] for i in ids) / len(ids)
    ok("channel C main effect recovers the injected -1.0",
       close(m_s01, -1.0, 0.25), f"got {m_s01:+.4f}")

    # ======================================================================
    print()
    print("=" * 74)
    print("D. THE LETTER-MASS GATE")
    print("=" * 74)

    p = "e1_choice_llama.csv"
    write_choice_csv(p, {}, seed=6, low_mass_conditions=("F01",),
                     drop_one_order=("F04",), model="llama")
    val2, mass2, gated2, total2, slot2, unpaired2 = AC.load_choice([p])
    ok("low-mass reads are gated out",
       "F01" not in val2["llama"] or len(val2["llama"].get("F01", {})) == 0,
       f"gated {gated2['llama']} of {total2['llama']} reads")
    ok("the gate count is reported, not silently applied",
       gated2["llama"] == N_PROF * N_Q * 2,
       f"{gated2['llama']}")
    ok("an item missing one slot order is DROPPED, not half-counted",
       len(val2["llama"].get("F04", {})) == 0 and unpaired2["llama"] > 0,
       f"{unpaired2['llama']} items dropped for want of both orders")
    out2 = run_analyser([p])
    ok("analyser prints the gate section", "LETTER-MASS GATE" in out2)

    # ======================================================================
    print()
    print("=" * 74)
    print("E. THE COHERENCE CHECK")
    print("=" * 74)

    p = "e1_traits_mistral.csv"
    write_traits_csv(p, {}, seed=8, incoherent=("S01_s1",), model="mistral")
    out3 = run_analyser([p])
    ok("an incoherent condition is flagged", "INCOHERENT" in out3,
       "positive and negative words moving together")
    p = "e1_traits_llama.csv"
    write_traits_csv(p, {}, seed=9, model="llama")
    out4 = run_analyser([p])
    ok("a coherent instrument is NOT flagged", "INCOHERENT" not in out4)

    # ======================================================================
    print()
    print("=" * 74)
    print("F. FALSE POSITIVES, no interaction anywhere")
    print("=" * 74)

    for label, writer, loader, channel in (
            ("B", write_traits_csv, AC.load_traits, "B"),
            ("C", write_choice_csv, AC.load_choice, "C")):
        fp = tested = 0
        for s in range(20):
            fn = f"fp_{label}_{s}.csv"
            writer(fn, {}, seed=2000 + s)
            v = loader([fn])[0]
            r = contrasts_from(v["qwen"], channel)
            for name in r:
                tested += 1
                if r[name]["bh"] < 0.05:
                    fp += 1
            os.remove(fn)
        rate = fp / tested
        ok(f"channel {label} false-positive rate near nominal",
           rate <= 0.06, f"{fp}/{tested} = {rate:.2%}")

    # ======================================================================
    print()
    print("=" * 74)
    print("G. POWER, so a null on these channels can be read")
    print("=" * 74)

    def power(label, writer, loader, channel, size, trials=20):
        hits = 0
        for s in range(trials):
            fn = f"pw_{label}_{s}.csv"
            writer(fn, {"P1_FORMxFORM": size}, seed=3000 + s)
            v = loader([fn])[0]
            r = contrasts_from(v["qwen"], channel)
            if r["P1_FORMxFORM"]["bh"] < 0.05:
                hits += 1
            os.remove(fn)
        return hits / trials

    print("   channel B, n=100 profiles, item noise 0.35 logits:")
    for size in (-0.10, -0.20, -0.40):
        print(f"     {size:+.2f} logits   {power('B', write_traits_csv, AC.load_traits, 'B', size):.0%}")
    print("   channel C, n=200 profile-question items, item noise 1.2 logits:")
    for size in (-0.40, -0.80, -1.20):
        print(f"     {size:+.2f} logits   {power('C', write_choice_csv, AC.load_choice, 'C', size):.0%}")

    v = AC.load_traits(["e1_traits_qwen.csv"])[0]
    mde_b = S.mde(AC.contrast_values(v["qwen"], CONTRASTS[0], "B"))
    v = AC.load_choice(["e1_choice_qwen.csv"])[0]
    mde_c = S.mde(AC.contrast_values(v["qwen"], CONTRASTS[0], "C"))
    print()
    print(f"   MDE, channel B: {mde_b:.4f} logits")
    print(f"   MDE, channel C: {mde_c:.4f} logits")
    print("   Compare with channel A, whose MDE is 0.077 in units of accuracy.")
    print("   Paper 2 single-signal effects: method 1 ran 0.95 to 2.92 logits,")
    print("   method 6b ran 0.58 to 6.93. These channels see a fraction of that,")
    print("   which is the whole reason they carry the interaction question.")
    print()
    print("   CAVEAT, and it matters. Channel A's power is fixed by the outcome:")
    print("   a binary variable has a variance you cannot argue with. These two")
    print("   numbers instead depend on the item noise assumed here, 0.35 logits")
    print("   for B and 1.2 for C, which are guesses until real data arrives.")
    print("   Treat them as showing the ORDER of sensitivity, not its value. The")
    print("   analyser recomputes the MDE from the collected data, and that is")
    print("   the number that goes in the paper.")
    ok("channel B MDE is a small fraction of a Paper 2 main effect",
       mde_b < 0.30, f"{mde_b:.4f} against main effects of 0.95 and up")
    ok("channel C MDE is a small fraction of a Paper 2 main effect",
       mde_c < 0.80, f"{mde_c:.4f} against main effects of 0.58 and up")

    # ======================================================================
    print()
    print("=" * 74)
    print("H. THE REBUILT CHOICE CONTRAST (run_e1_choice_pairs)")
    print("=" * 74)
    print("   The first contrast differenced margins against a common")
    print("   reference. This proves the new one survives the exact failure")
    print("   mode that broke it: a SATURATING pairwise read.")

    import math
    import run_e1_choice_pairs as CP

    comps = CP.build_comparisons(CONTRASTS)
    ok("every contrast gets both directions",
       len(comps) == 2 * len(CONTRASTS)
       and {c[1] for c in comps} == {"vs_a", "vs_b"},
       f"{len(comps)} comparisons")
    ok("the both-cell is always the LEFT side",
       all(c[2] == ct["both"] for c, ct in
           zip(comps[::2], CONTRASTS)))
    ok("the floor gets rebuilt too, not inherited",
       any(c[0] == "FLOOR_ALTxALT" for c in comps),
       "being rebuilt is not the same as being right")

    conds = {c["id"]: c for c in CONDITIONS}
    jobs = CP.build_jobs(conds, CONTRASTS, PROFILES)
    ok("pairs job count is contrasts x 2 x profiles x questions x 2 orders",
       len(jobs) == 2 * len(CONTRASTS) * len(PROFILES) * len(CP.QUESTIONS) * 2,
       f"{len(jobs)}")
    orders = {}
    for name, side, l, r, pid, qid, order, _p in jobs:
        orders.setdefault((name, side, pid, qid), set()).add(order)
    ok("both slot orders on every comparison",
       all(v == {"left_first", "left_second"} for v in orders.values()))

    # a world where the STRENGTHS are exactly additive but the READ saturates
    K = 5.0
    strength = {"NONE": 0.0, "ALT1": -6.0, "ALT2_ONLY": -2.0,
                "ALT_BOTH": -8.0}

    def read(x, y):
        return K * math.tanh((strength[x] - strength[y]) / K)

    old = (read("ALT_BOTH", "NONE") - read("ALT1", "NONE")
           - read("ALT2_ONLY", "NONE"))
    new_a = read("ALT_BOTH", "ALT1") - read("ALT2_ONLY", "NONE")
    new_b = read("ALT_BOTH", "ALT2_ONLY") - read("ALT1", "NONE")
    print()
    print(f"   strengths are perfectly additive: "
          f"ALT_BOTH {strength['ALT_BOTH']:+.0f} = "
          f"{strength['ALT1']:+.0f} {strength['ALT2_ONLY']:+.0f}")
    print(f"   old contrast reports {old:+.4f} anyway")
    print(f"   new contrast reports {new_a:+.4f} and {new_b:+.4f}")
    ok("the OLD contrast invents an interaction where there is none",
       abs(old) > 1.0, f"{old:+.4f} out of nothing")
    ok("the NEW contrast returns zero, both directions",
       abs(new_a) < 1e-9 and abs(new_b) < 1e-9,
       f"{new_a:+.2e} and {new_b:+.2e}")

    # and it still finds a real interaction under the same saturation
    strength["ALT_BOTH"] = -11.0            # 3 logits worse than additive
    new_a = read("ALT_BOTH", "ALT1") - read("ALT2_ONLY", "NONE")
    ok("the NEW contrast still detects a genuine super-additive effect",
       new_a < -1.0, f"{new_a:+.4f} for an injected -3.0")
    ok("and it is attenuated, not inflated, by the saturation",
       new_a > -3.0, f"{new_a:+.4f}, so the estimate is conservative")

    # ======================================================================
    print()
    print("=" * 74)
    print("I. THE LIMIT: context dependence defeats every pairwise contrast")
    print("=" * 74)
    print("   Section H proved robustness to saturation. The real data failed")
    print("   differently: the effect of a clause depends on its neighbour.")
    print("   This world has NO social interaction at all, and one context term")
    print("   between the two neutral clauses.")

    a = {"n1": 0.0, "alt1": -2.0, "S01": -1.0}
    b = {"n2": 0.0, "alt2": -1.5, "S08": -1.0}
    ctx = {("alt1", "alt2"): 3.0}

    def strength(s1, s2):
        return a[s1] + b[s2] + ctx.get((s1, s2), 0.0)

    def m(x, y):
        return strength(*x) - strength(*y)

    none = ("n1", "n2")
    floor_new = m(("alt1", "alt2"), ("alt1", "n2")) - m(("n1", "alt2"), none)
    social_new = m(("S01", "S08"), ("S01", "n2")) - m(("n1", "S08"), none)
    corrected = social_new - floor_new
    print()
    print(f"   rebuilt floor {floor_new:+.2f}, rebuilt social {social_new:+.2f}, "
          f"floor-corrected social {corrected:+.2f}")
    ok("with no social interaction, the social contrast is correctly zero",
       abs(social_new) < 1e-9, f"{social_new:+.4f}")
    ok("but the floor is NOT zero, because neutral clauses interact",
       abs(floor_new - 3.0) < 1e-9, f"{floor_new:+.4f}")
    ok("so correcting by the floor MANUFACTURES a compounding effect",
       abs(corrected + 3.0) < 1e-9,
       f"{corrected:+.4f} out of nothing; subtraction is not a defence")
    ok("the only defence is the floor check itself: a nonzero floor means the "
       "instrument cannot identify an interaction", abs(floor_new) > 0.5)

finally:
    os.chdir(cwd)
    shutil.rmtree(tmp, ignore_errors=True)

# ==========================================================================
print()
print("=" * 74)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("e1 channels B and C are safe to run.")
print("=" * 74)
