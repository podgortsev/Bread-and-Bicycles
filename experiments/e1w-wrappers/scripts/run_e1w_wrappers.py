#!/usr/bin/env python3
"""
run_e1w_wrappers.py - e1w: does the interaction survive a change of the neutral
text around the signal.

    9 conditions x 100 tasks x 6 wrappers = 5,400 generations per model.

Upload next to this script in Colab:
    e1_conditions.json      from experiments/shared/conditions/
    tasks.json              from experiments/shared/tasks/

WHY THIS EXPERIMENT EXISTS
--------------------------
It is the card no competitor in this literature holds.

Paper 2's method 3b ran the same 24 conditions twice, changing only the neutral
padding around the signal. The correlation between the two runs was r=0.13 on
Llama and r=0.22 on Qwen. Individual conditions changed sign. The shape of
Llama's accumulation curve inverted.

Tonneau et al. (arXiv:2601.18486, EMNLP 2026) report the same from a different
direction across 14.8 million prompts: cues for the same group produce only
partially overlapping changes, and both the magnitude AND the direction of a
group difference vary with how the group is cued.

Every stacking result in the compounding camp is a single-wrapper result. So the
headline interaction from e1 is re-measured under six neutral wrappers, and what
gets reported is the across-wrapper mean with the across-wrapper spread as the
error bar. That is what method 3c did for single signals, and it is what turned
an unreproducible finding into a defensible one.

THE SIX WRAPPERS
----------------
Taken unchanged from Paper 2's method 3c, so the two papers measure wrapper
sensitivity on the same axis. They are arbitrary but defensible openings; in
Paper 2 they moved baseline accuracy by 2 to 5 points on their own.

WHICH NINE CONDITIONS
---------------------
Full six-wrapper coverage of all seventeen e1 conditions would cost six sessions
per model. This covers exactly the cells the headline rests on: NONE, the four
singles, and the four both-cells in one slot order. That is enough to recompute
every interaction contrast once per wrapper.

The slot-order arm is deliberately NOT repeated here. Order is answered by e1 at
full n; this experiment answers a different question and doubling it would buy
nothing.

100 TASKS, NOT 600
------------------
e1 needs 600 tasks because it estimates one interaction as precisely as a binary
outcome allows. This experiment does not need that precision per wrapper: it
needs six independent estimates whose SPREAD is the answer. The spread is
estimated from six numbers, so precision within each costs six times as much and
buys little.

The consequence is stated rather than hidden: each per-wrapper interaction here
is noisy, and only the spread across wrappers is interpreted.
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = "/content/drive/MyDrive/afl/p3_e1w"
N_TASKS = 100
MAX_NEW_TOKENS = 160
BATCH = 8

# Paper 2's method 3c wrappers, unchanged
WRAPPERS = {
    "W1": "I have a question I need some help with today.",
    "W2": "There is something I would like to ask you about.",
    "W3": "Could you help me with the following, please.",
    "W4": "I am trying to work something out and could use a hand.",
    "W5": "Quick question for you.",
    "W6": "Here is what I need to figure out.",
}

CONDITIONS = ["NONE", "F01", "F04", "S01_s1", "S08_s2",
              "P1_F01F04", "P2_S01S08", "P3_F01S01_s1", "P4_F04S08_s1"]

INSTRUCTION = ("Answer with the final answer only. Do not show working. "
               "If the answer is a number, give just the number.")


def load_conditions():
    with open("e1_conditions.json", encoding="utf-8") as f:
        d = json.load(f)
    keep = {c["id"]: c for c in d["conditions"] if c["id"] in CONDITIONS}
    missing = [c for c in CONDITIONS if c not in keep]
    if missing:
        raise SystemExit(f"e1_conditions.json is missing {missing}")
    return [keep[c] for c in CONDITIONS], d["contrasts"]


def raw_path(model_key):
    return os.path.join(OUT_DIR, f"e1w_wrappers_{model_key}.csv")


def build_jobs(conditions, tasks):
    jobs = []
    for wid, wtext in WRAPPERS.items():
        for c in conditions:
            for t in tasks:
                # the wrapper opens, then the person's two clauses, then the task
                prompt = (f"{wtext} {c['lead_in']}\n\n{t['question']}\n\n"
                          f"{INSTRUCTION}")
                jobs.append((wid, c["id"], t["id"], t["answer"],
                             t["answer_kind"], t["family"], prompt))
    return jobs


def collect(model):
    """model is a p3_model.P3Model. Resumable."""
    import run_e1_accuracy as A          # the parser lives there, one copy
    os.makedirs(OUT_DIR, exist_ok=True)
    conditions, _ = load_conditions()
    with open("tasks.json", encoding="utf-8") as f:
        tasks = json.load(f)[:N_TASKS]
    path = raw_path(model.key)

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["wrapper"], r["condition"], r["task_id"])
                    for r in csv.DictReader(f)}
        print(f"  resuming, {len(done)} rows already saved")

    jobs = [j for j in build_jobs(conditions, tasks)
            if (j[0], j[1], j[2]) not in done]
    if not jobs:
        print("  e1w already complete")
        return
    print(f"  {len(jobs)} generations to do")

    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["model", "model_key", "wrapper", "condition", "task_id",
                    "family", "correct", "refused", "answer_key", "raw_output"])

    for i in range(0, len(jobs), BATCH):
        chunk = jobs[i:i + BATCH]
        try:
            outs = model.generate([c[6] for c in chunk], MAX_NEW_TOKENS)
        except Exception as e:
            print(f"    batch at {i} failed: {e}")
            continue
        for (wid, cid, tid, key, kind, fam, _p), o in zip(chunk, outs):
            ok, ref = A.parse(o, key, kind)
            w.writerow([model.name, model.key, wid, cid, tid, fam,
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
        acc.setdefault(r["wrapper"], {}).setdefault(
            r["condition"], {})[r["task_id"]] = int(r["correct"])
    print("\n" + "=" * 74)
    print(f"e1w WRAPPERS, {model_key}: first look")
    print("=" * 74)
    print("\n  accuracy by wrapper and condition\n")
    print(f"  {'condition':16}" + "".join(f"{w:>8}" for w in sorted(acc)))
    conds = sorted({c for w in acc for c in acc[w]},
                   key=lambda c: CONDITIONS.index(c) if c in CONDITIONS else 99)
    for cid in conds:
        cells = []
        for wid in sorted(acc):
            v = list(acc[wid].get(cid, {}).values())
            cells.append(f"{sum(v) / len(v):8.1%}" if v else f"{'-':>8}")
        print(f"  {cid:16}" + "".join(cells))
    print("\n  Run analyse_e1w_wrappers.py for the interaction per wrapper and")
    print("  the across-wrapper spread, which is the number that matters.")


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
