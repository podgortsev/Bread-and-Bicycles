#!/usr/bin/env python3
"""
run_e2b_legibility.py - e2b: is the published super-additivity a legibility
effect.

    4 cells x 600 tasks = 2,400 generations per model. The cheapest experiment
    in Paper 3 and possibly the most valuable.

Upload next to this script in Colab:
    e1_conditions.json      from experiments/shared/conditions/
    e2b_tasks.json          built by build_e2b_tasks.py

THE COMPARISON
--------------
Two arms, the same four cells, differing in ONE thing: whether the form style
touches the task or only the person's message.

    arm A   collected by e1 channel A. Lead-in styled, task byte-identical.
            cells NONE, F01, F04, P1_F01F04
    arm B   collected here. Lead-in styled AND task text styled the same way.
            cells std, esl, typo, esl_typo

Liu et al. (arXiv:2604.04723) run something like arm B and report a
super-additive drop. If arm B reproduces that and arm A does not, then the
published compounding result is about the model failing to read a corrupted
input, not about it treating a person worse for two signals. Those are
different claims and only one of them is about fairness.

Three outcomes, all worth having:

    arm B super-additive, arm A not     the effect is legibility
    both super-additive                 the effect survives a clean task, which
                                        is a stronger version of the compounding
                                        claim than the literature has shown
    neither                             the form pair does not compound at all,
                                        and Liu et al. do not replicate here

WHAT IS HELD CONSTANT
---------------------
Both arms use the same 600 tasks, the same answer keys, the same two-clause
lead-ins and the same parser. The only difference is the task text.

WHAT IS NOT PERFECT, AND IS RECORDED PER ROW
--------------------------------------------
The typo rewriter needs a word of five or more letters that is not one of the
meaning-carrying words it must never touch. On 59 of the 600 tasks there is no
such word, so the typo arm's text is identical to std there. Those rows are
flagged `text_changed=0` and the analysis reports the interaction with and
without them. Dropping them silently would overstate the arm; keeping them
silently would dilute it.
"""

import csv
import json
import os
import sys

OUT_DIR = "/content/drive/MyDrive/afl/p3_e2b"
N_TASKS = 600
MAX_NEW_TOKENS = 160
BATCH = 8

INSTRUCTION = ("Answer with the final answer only. Do not show working. "
               "If the answer is a number, give just the number.")

# arm B cell -> the e1 condition supplying the lead-in for that style
CELLS = [
    ("std", "NONE"),
    ("esl", "F01"),
    ("typo", "F04"),
    ("esl_typo", "P1_F01F04"),
]


def load_leadins():
    with open("e1_conditions.json", encoding="utf-8") as f:
        conds = {c["id"]: c["lead_in"] for c in json.load(f)["conditions"]}
    missing = [cid for _, cid in CELLS if cid not in conds]
    if missing:
        raise SystemExit(f"e1_conditions.json is missing {missing}")
    return {arm: conds[cid] for arm, cid in CELLS}


def load_tasks():
    with open("e2b_tasks.json", encoding="utf-8") as f:
        return json.load(f)[:N_TASKS]


def raw_path(model_key):
    return os.path.join(OUT_DIR, f"e2b_legibility_{model_key}.csv")


def build_jobs(leadins, tasks):
    jobs = []
    for arm, _cid in CELLS:
        for t in tasks:
            q = t["text"][arm]
            changed = int(q != t["text"]["std"])
            prompt = f"{leadins[arm]}\n\n{q}\n\n{INSTRUCTION}"
            jobs.append((arm, t["id"], t["answer"], t["answer_kind"],
                         t["family"], changed, prompt))
    return jobs


def collect(model):
    """model is a p3_model.P3Model. Resumable."""
    import run_e1_accuracy as A          # one parser, one copy
    os.makedirs(OUT_DIR, exist_ok=True)
    leadins = load_leadins()
    tasks = load_tasks()
    path = raw_path(model.key)

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["arm"], r["task_id"]) for r in csv.DictReader(f)}
        print(f"  resuming, {len(done)} rows already saved")

    jobs = [j for j in build_jobs(leadins, tasks) if (j[0], j[1]) not in done]
    if not jobs:
        print("  e2b already complete")
        return
    print(f"  {len(jobs)} generations to do")

    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["model", "model_key", "arm", "task_id", "family",
                    "text_changed", "correct", "refused", "answer_key",
                    "raw_output"])

    for i in range(0, len(jobs), BATCH):
        chunk = jobs[i:i + BATCH]
        try:
            outs = model.generate([c[6] for c in chunk], MAX_NEW_TOKENS)
        except Exception as e:
            print(f"    batch at {i} failed: {e}")
            continue
        for (arm, tid, key, kind, fam, changed, _p), o in zip(chunk, outs):
            ok, ref = A.parse(o, key, kind)
            w.writerow([model.name, model.key, arm, tid, fam, changed,
                        int(ok), int(ref), key,
                        o.strip().replace("\n", " ")[:400]])
        f.flush()
        if (i // BATCH) % 50 == 0:
            print(f"    {i + len(chunk)} / {len(jobs)}")
    f.close()
    print(f"  saved to {path}")


def summary(model_key):
    path = raw_path(model_key)
    if not os.path.exists(path):
        print(f"  no results at {path}")
        return
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    acc = {}
    for r in rows:
        acc.setdefault(r["arm"], {})[r["task_id"]] = int(r["correct"])
    print("\n" + "=" * 74)
    print(f"e2b LEGIBILITY, {model_key}: first look, arm B (task styled too)")
    print("=" * 74)
    for arm, _cid in CELLS:
        if arm not in acc:
            continue
        v = list(acc[arm].values())
        print(f"  {arm:10}{sum(v) / len(v):8.1%}   n={len(v)}")
    if all(a in acc for a, _ in CELLS):
        ids = sorted(set.intersection(*[set(acc[a]) for a, _ in CELLS]))
        d = [(acc["esl_typo"][i] - acc["esl"][i])
             - (acc["typo"][i] - acc["std"][i]) for i in ids]
        print(f"\n  arm B interaction: {sum(d) / len(d):+.4f} over {len(d)} tasks")
        print("  negative = super-additive")
    print("\n  Run analyse_e2b_legibility.py with BOTH this file and the e1")
    print("  channel A file for the arm A against arm B comparison, which is")
    print("  the whole point of the experiment.")


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "e1-type-x-count", "scripts")))
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared")))
    import p3_model
    key = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else "qwen"
    if "--analyse-only" in sys.argv:
        summary(key)
    else:
        collect(p3_model.P3Model(key))
        summary(key)
