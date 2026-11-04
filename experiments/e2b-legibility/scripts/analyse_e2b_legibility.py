#!/usr/bin/env python3
"""
analyse_e2b_legibility.py - the read to trust for e2b. No GPU.

    python analyse_e2b_legibility.py e2b_legibility_qwen.csv e1_accuracy_qwen.csv

Give it BOTH files per model: the arm B file collected by e2b and the arm A file
collected by e1 channel A. With only one of them it reports that arm alone and
says so; the comparison is the experiment.

THE QUESTION
------------
Both arms estimate the same form-by-form interaction on the same 600 tasks. They
differ only in whether the form style touches the task text.

    arm A   task byte-identical           from e1: NONE, F01, F04, P1_F01F04
    arm B   task styled the same way      from e2b: std, esl, typo, esl_typo

The difference between the two interactions is the legibility component. It is
tested directly, paired within task, because both arms saw the same items:

    legibility = interaction(arm B) - interaction(arm A)

WHY THIS IS WORTH ITS 2,400 GENERATIONS
---------------------------------------
Liu et al. (arXiv:2604.04723) report, from a design like arm B, a combination
larger than either factor alone and in most settings smaller than the sum. No
published work separates corrupting the task from styling only the person's
message, which is what the two arms do. (Corrected 2026-09-12: this docstring
first described their result as super-additive.)

THE 59 TASKS THAT DO NOT CHANGE
-------------------------------
The typo rewriter needs a long enough word that is not one of the words carrying
what the question asks. On 59 of the 600 tasks there is none, so the typo arm's
text equals std there. Every interaction below is reported twice: over all
tasks, and over the subset where the text really changed. If those two disagree,
the arm-B number is being diluted by rows that were never in arm B at all.
"""

import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared")))

import p3_stats as S      # noqa: E402

# arm A cell -> the arm B cell it corresponds to
ARM_A_CELLS = {"std": "NONE", "esl": "F01", "typo": "F04",
               "esl_typo": "P1_F01F04"}


def load(paths):
    """Returns armB[model][cell][task], armA[model][cell][task], changed set."""
    arm_b = defaultdict(lambda: defaultdict(dict))
    arm_a = defaultdict(lambda: defaultdict(dict))
    changed = defaultdict(set)
    refused = defaultdict(lambda: defaultdict(list))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            fields = rd.fieldnames or []
            for r in rd:
                mk = r["model_key"]
                if "arm" in fields:
                    arm_b[mk][r["arm"]][r["task_id"]] = int(r["correct"])
                    refused[mk][f"B:{r['arm']}"].append(int(r["refused"]))
                    # only the TYPO arm can leave a task unchanged, so
                    # that is the arm whose flag defines the subset. Using the
                    # union across arms made the subset all 600 tasks and the
                    # line meaningless.
                    if r["arm"] == "typo" and r.get("text_changed") == "1":
                        changed[mk].add(r["task_id"])
                elif "condition" in fields:
                    cid = r["condition"]
                    for arm, want in ARM_A_CELLS.items():
                        if cid == want:
                            arm_a[mk][arm][r["task_id"]] = int(r["correct"])
                            refused[mk][f"A:{arm}"].append(int(r["refused"]))
    return arm_b, arm_a, changed, refused


def interaction(cells, ids=None):
    need = ["esl_typo", "esl", "typo", "std"]
    if any(c not in cells for c in need):
        return [], []
    have = set.intersection(*[set(cells[c]) for c in need])
    keys = sorted(have if ids is None else (have & ids))
    d = [(cells["esl_typo"][i] - cells["esl"][i])
         - (cells["typo"][i] - cells["std"][i]) for i in keys]
    return d, keys


def report(label, d):
    if not d:
        return None
    mean, lo, hi = S.boot_ci(d)
    t = S.two_tests(d)
    print(f"    {label:34}n={len(d):4}  {mean:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]  p rank {t['p_rank']:.4f} "
          f"mean {t['p_mean']:.4f}  MDE +/-{S.mde(d):.4f}")
    out = {"n": len(d), "mean": mean, "lo": lo, "hi": hi,
           "mde": S.mde(d), "d": d}
    out.update(t)
    return out


def main(paths):
    arm_b, arm_a, changed, refused = load(paths)
    models = sorted(set(arm_b) | set(arm_a))

    print("=" * 78)
    print("e2b: does corrupting the task change the form-by-form interaction")
    print(f"models: {', '.join(models)}")
    print("=" * 78)

    for m in models:
        print(f"\n{'-' * 78}\n{m}\n{'-' * 78}")
        has_a = all(c in arm_a[m] for c in ARM_A_CELLS)
        has_b = all(c in arm_b[m] for c in ARM_A_CELLS)
        if not has_b:
            print("  no arm B data in the files given")
        if not has_a:
            print("  no arm A data. Pass the e1 channel A CSV for this model")
            print("  as well; the comparison is the experiment.")

        print("\n  accuracy per cell")
        for cell in ("std", "esl", "typo", "esl_typo"):
            a = list(arm_a[m].get(cell, {}).values())
            b = list(arm_b[m].get(cell, {}).values())
            sa = f"{sum(a) / len(a):7.1%}" if a else f"{'-':>7}"
            sb = f"{sum(b) / len(b):7.1%}" if b else f"{'-':>7}"
            print(f"    {cell:10}arm A (task clean) {sa}    "
                  f"arm B (task styled) {sb}")

        print("\n  the form-by-form interaction, negative = super-additive")
        ra = rb = None
        if has_a:
            da, _ = interaction(arm_a[m])
            ra = report("arm A, task byte-identical", da)
        if has_b:
            db, _ = interaction(arm_b[m])
            rb = report("arm B, task styled too", db)
            if changed[m]:
                dbc, _ = interaction(arm_b[m], changed[m])
                report("arm B, only tasks that changed", dbc)
                skipped = len(db) - len(dbc)
                if skipped:
                    print(f"    ({skipped} tasks had no eligible word for a "
                          f"typo, so their typo cell equals std)")

        if ra and rb:
            print("\n  THE LEGIBILITY COMPONENT, paired within task")
            ids = sorted(set.intersection(
                *[set(arm_a[m][c]) for c in ARM_A_CELLS],
                *[set(arm_b[m][c]) for c in ARM_A_CELLS]))
            da_i, _ = interaction(arm_a[m], set(ids))
            db_i, _ = interaction(arm_b[m], set(ids))
            diff = [b - a for a, b in zip(da_i, db_i)]
            r = report("arm B minus arm A", diff)

            print("\n  VERDICT")
            a_super = ra["mean"] < 0 and abs(ra["mean"]) > ra["mde"]
            b_super = rb["mean"] < 0 and abs(rb["mean"]) > rb["mde"]
            sep = r and abs(r["mean"]) > r["mde"]
            if b_super and not a_super and sep:
                print("    The interaction appears only when the TASK is")
                print("    corrupted. On this model the published")
                print("    super-additive result is a legibility effect, not a")
                print("    statement about how the model treats a person.")
            elif b_super and a_super:
                print("    Super-additive in both arms. The form pair compounds")
                print("    even with a byte-identical task, which is a stronger")
                print("    claim than the literature has established.")
            elif not b_super and not a_super:
                print("    Neither arm is super-additive. Liu et al. report a")
                print("    combination that is mostly SUB-additive when the task")
                print("    itself is corrupted, so read arm B's sign against that,")
                print("    and the MDE above for how large an effect this run could")
                print("    have seen. (Corrected 2026-09-12: an earlier version read")
                print("    their result as super-additive.)")
            else:
                print("    Mixed. Read the two MDE values before concluding")
                print("    anything: at this n the difference between the arms")
                print("    may simply not be resolvable.")

        print("\n  refusal by cell")
        for k in sorted(refused[m]):
            v = refused[m][k]
            print(f"    {k:16}{sum(v) / len(v):6.2%}  ({sum(v)} of {len(v)})")

    print("\n" + "=" * 78)
    print("Both arms use the same 600 tasks, the same keys, the same lead-ins")
    print("and the same parser. The only difference is the task text.")
    print("=" * 78)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        sys.exit("usage: analyse_e2b_legibility.py <e2b csv> [<e1 accuracy csv>]")
    missing = [a for a in args if not os.path.exists(a)]
    if missing:
        sys.exit(f"missing: {', '.join(missing)}")
    main(args)
