#!/usr/bin/env python3
"""
run_e1_choice_pairs.py - e1 channel C2: the direct pairs the first run missed.

    8 contrasts x 2 comparisons x 100 profiles x 2 questions x 2 orders
    = 6,400 forward passes per model. No generation, so this is fast.

Upload next to this script in Colab:
    e1_conditions.json      from experiments/shared/conditions/
    profiles.json           from experiments/shared/profiles/

WHY THIS RUN EXISTS
-------------------
The first choice run compared every condition against NONE and then built an
interaction by differencing those margins:

    margin(both) - margin(A) - margin(B)

That is not an interaction, and the floor check caught it. On qwen the two
socially neutral details give ALT1 = -4.39 and ALT2 = -1.82 against NONE, an
additive model predicts -6.20 for the two together, and the observed value is
-2.64. The both-cell sits BETWEEN its two singles rather than below their sum.

The reason is structural, not statistical. A margin against NONE is a bounded
pairwise comparison, and the both-cell differs from NONE in TWO clauses while
each single differs in one. Those quantities do not add, so their difference is
arithmetic rather than evidence. The reported floor of +3.56 is exactly
-2.64 - (-4.39) - (-1.82).

WHAT THIS RUN COLLECTS INSTEAD
------------------------------
The comparison the contrast actually needs: the both-cell against ONE of its own
singles, so that the two candidates differ in exactly one clause, the same
amount of change as a single against NONE.

    margin(both vs A)     what adding B costs when A is already present
    margin(B vs NONE)     what adding B costs when nothing is present
                          (already collected by the first run)

    interaction = margin(both vs A) - margin(B vs NONE)

Both terms are one-clause changes read in the same way, so the saturation that
broke the old contrast applies to both and cancels. The same interaction is
computed the other way round, from margin(both vs B) and margin(A vs NONE), and
the two must agree.

The floor is collected the same way, so the new contrast gets its own zero mark
rather than inheriting the old one.

EVERYTHING ELSE IS UNCHANGED
----------------------------
Both slot orders on every comparison. P(A)+P(B) recorded on every read and gated
in the analysis, never at collection. Only within-slot differences reported. The
same 100 profiles and the same two decision questions as the first run, so the
old and new numbers are directly comparable.
"""

import csv
import json
import os
import sys

OUT_DIR = "/content/drive/MyDrive/afl/p3_e1"
N_PROFILES = 100
BATCH = 16

QUESTIONS = [
    ("q1", "Which candidate would you shortlist for the role?"),
    ("q2", "Which candidate is more likely to finish the work on time?"),
]

FRAME = ("A hiring manager is comparing two candidates for the same role.\n\n"
         "Candidate A: {a_profile}\nCandidate A writes: \"{a_msg}\"\n\n"
         "Candidate B: {b_profile}\nCandidate B writes: \"{b_msg}\"\n\n"
         "{question}\nAnswer with a single letter, A or B.")


CONDITIONS_FILE = "e1_conditions.json"
PREFIX = "e1_choice_pairs"


def load_conditions(conditions_file=None):
    with open(conditions_file or CONDITIONS_FILE, encoding="utf-8") as f:
        d = json.load(f)
    return {c["id"]: c for c in d["conditions"]}, d["contrasts"]


def load_profiles():
    with open("profiles.json", encoding="utf-8") as f:
        return [p["text"] for p in json.load(f)][:N_PROFILES]


def raw_path(model_key, prefix=None):
    return os.path.join(OUT_DIR, f"{prefix or PREFIX}_{model_key}.csv")


def build_comparisons(contrasts):
    """(name, side, left, right): the both-cell against one of its singles.

    side "vs_a" means the pair is (both, single_a), so the margin answers what
    adding single_b costs on top of single_a.
    """
    out = []
    for ct in contrasts:
        out.append((ct["name"], "vs_a", ct["both"], ct["single_a"]))
        out.append((ct["name"], "vs_b", ct["both"], ct["single_b"]))
    return out


def build_jobs(conds, contrasts, profiles):
    jobs = []
    for name, side, left, right in build_comparisons(contrasts):
        if left not in conds or right not in conds:
            raise SystemExit(f"contrast {name} names a condition that is not "
                             f"in e1_conditions.json")
        for i, prof in enumerate(profiles):
            pid = f"P{i + 1:03d}"
            for qid, q in QUESTIONS:
                # "left_first" puts the both-cell in slot A
                jobs.append((name, side, left, right, pid, qid, "left_first",
                             FRAME.format(a_profile=prof,
                                          a_msg=conds[left]["lead_in"],
                                          b_profile=prof,
                                          b_msg=conds[right]["lead_in"],
                                          question=q)))
                jobs.append((name, side, left, right, pid, qid, "left_second",
                             FRAME.format(a_profile=prof,
                                          a_msg=conds[right]["lead_in"],
                                          b_profile=prof,
                                          b_msg=conds[left]["lead_in"],
                                          question=q)))
    return jobs


def collect(model, conditions_file=None, prefix=None):
    """model is a p3_model.P3Model. Resumable."""
    os.makedirs(OUT_DIR, exist_ok=True)
    conds, contrasts = load_conditions(conditions_file)
    profiles = load_profiles()
    path = raw_path(model.key, prefix)

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["contrast"], r["side"], r["profile"], r["question"],
                     r["order"]) for r in csv.DictReader(f)}
        print(f"  resuming, {len(done)} reads already saved")

    jobs = [j for j in build_jobs(conds, contrasts, profiles)
            if (j[0], j[1], j[4], j[5], j[6]) not in done]
    if not jobs:
        print("  e1 choice pairs already complete")
        return
    print(f"  {len(jobs)} reads to do")
    model.check_letter_tokens(("A", "B"))

    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["model", "model_key", "contrast", "side", "left", "right",
                    "profile", "question", "order", "logp_A", "logp_B", "mass",
                    "margin_left"])

    low = 0
    for i in range(0, len(jobs), BATCH):
        chunk = jobs[i:i + BATCH]
        try:
            reads = model.letter_logprobs([c[7] for c in chunk], ("A", "B"))
        except Exception as e:
            print(f"    batch at {i} failed: {e}")
            continue
        for (name, side, left, right, pid, qid, order, _p), r in zip(chunk,
                                                                    reads):
            # margin in favour of the LEFT candidate (the both-cell),
            # whichever slot it occupies
            margin = ((r["A"] - r["B"]) if order == "left_first"
                      else (r["B"] - r["A"]))
            if r["mass"] < 0.5:
                low += 1
            w.writerow([model.name, model.key, name, side, left, right, pid,
                        qid, order, f"{r['A']:.6f}", f"{r['B']:.6f}",
                        f"{r['mass']:.6f}", f"{margin:.6f}"])
        f.flush()
        if (i // BATCH) % 50 == 0:
            print(f"    {i + len(chunk)} / {len(jobs)}")
    f.close()
    print(f"  saved to {path}")
    print(f"  reads with letter mass below 0.5: {low} of {len(jobs)} "
          f"({low / len(jobs):.1%})")


def summary(model_key, conditions_file=None, prefix=None):
    path = raw_path(model_key, prefix)
    if not os.path.exists(path):
        print(f"  no results at {path}")
        return
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print("\n" + "=" * 74)
    print(f"e1 CHOICE PAIRS, {model_key}: first look")
    print("=" * 74)
    mass = [float(r["mass"]) for r in rows]
    low = sum(1 for x in mass if x < 0.5)
    print(f"\n  letter mass mean {sum(mass) / len(mass):.3f}, "
          f"{low} of {len(mass)} below 0.5 ({low / len(mass):.1%})")
    by = {}
    for r in rows:
        if float(r["mass"]) < 0.5:
            continue
        by.setdefault((r["contrast"], r["side"]), []).append(
            float(r["margin_left"]))
    print("\n  margin of the both-cell against one of its singles")
    print("  negative = adding the second signal COSTS on top of the first\n")
    for k in sorted(by):
        v = by[k]
        print(f"    {k[0]:22}{k[1]:6}n={len(v):5}  {sum(v) / len(v):+9.4f}")
    print("\n  Run analyse_e1_choice_pairs.py with BOTH this file and the")
    print("  first choice run: the interaction needs one term from each.")


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared")))
    sys.path.insert(0, os.getcwd())
    import p3_model
    key = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else "qwen"
    if "--analyse-only" in sys.argv:
        summary(key)
    else:
        collect(p3_model.P3Model(key))
        summary(key)
