#!/usr/bin/env python3
"""
run_e1_answer_logprob.py - e1 channel A2: how readily the model produces the
RIGHT answer to the person's own task.

    23 conditions x 600 tasks = 13,800 scored answers per model. No generation.

Upload (the floor bundle carries all of these):
    e1_conditions.json  e1_floor_conditions.json  tasks.json
    run_e1_accuracy.py  (the instruction is imported from it, one copy)

WHY THIS CHANNEL
----------------
Channel A asks whether the task was solved. Its floor is clean on all three
models, which makes it the only instrument in Paper 3 that can identify an
interaction. But it is binary, and single signals cost accuracy only on llama,
so on qwen and mistral there is no effect for an interaction to modify.

This channel reads the same prompt and scores the log probability of the correct
answer key. Same 600 tasks, same lead-ins, same instruction, so it differs from
channel A only in what is read: a continuous quantity per task instead of 0 or 1.

WHY IT MIGHT HAVE A CLEAN FLOOR WHERE B AND C DID NOT
----------------------------------------------------
Channels B and C read the model's judgement ABOUT the person or the message, so
the words of the message are the thing being scored, and any two clauses
interact. This reads the answer to an arithmetic question, which is what channel
A reads, and channel A's floor was clean. That is a reason to expect a clean
floor, not a result. The floor runs first in the analysis, over three neutral
pairs, and if it is contaminated this is reported as a third instrument that
cannot identify an interaction.

WHAT IS SCORED
--------------
log P(answer key | prompt), summed over the key's tokens: the log probability of
emitting exactly the correct answer as the reply. Summed rather than averaged,
because the question is the probability of the whole answer. The key is the same
string in every condition for a given task, so every within-task comparison
cancels its length.

A low value is not a refusal and not an error. It is only ever read as a
difference between conditions on the same task.

THE GPU SELF-CHECK
------------------
The token indexing cannot be exercised offline. So before a single row is
written, a few prompts of different lengths are scored by the scorer and by a
plain full forward pass, and scored again in reverse order; all must agree. If
they do not, the run stops. See P3Model.check_answer_scoring.

The first run, on 2026-09-10, scored 16 answers per batch and this check
stopped it on all three models: batching alone moved scores by up to 0.56 nats.
Every answer is now scored in a forward pass of its own. Slower, about an hour
per model, and the only way a score cannot depend on its batch neighbours.
"""

import csv
import json
import os
import sys

OUT_DIR = "/content/drive/MyDrive/afl/p3_e1"
N_TASKS = 600
FLUSH_EVERY = 32    # rows written per flush; scoring is one answer per pass
PREFIX = "e1_answer_logprob"
CONDITION_FILES = ("e1_conditions.json", "e1_floor_conditions.json")


def load_conditions(files=CONDITION_FILES):
    """The union of the main and the floor condition sets, NONE counted once."""
    conds, contrasts, seen = [], [], {}
    for name in files:
        with open(name, encoding="utf-8") as f:
            d = json.load(f)
        for c in d["conditions"]:
            if c["id"] in seen:
                if seen[c["id"]] != c["lead_in"]:
                    raise SystemExit(f"{c['id']} has different text in {name}; "
                                     f"the union would not be one condition")
                continue
            seen[c["id"]] = c["lead_in"]
            conds.append(c)
        names = {x["name"] for x in contrasts}
        for ct in d["contrasts"]:
            if ct["name"] not in names:
                contrasts.append(ct)
    return conds, contrasts


def raw_path(model_key):
    return os.path.join(OUT_DIR, f"{PREFIX}_{model_key}.csv")


def build_jobs(conds, tasks):
    """(condition, task, family, answer key, prompt). The prompt is built
    exactly as channel A builds it, from the instruction channel A defines."""
    import run_e1_accuracy as A
    jobs = []
    for c in conds:
        for t in tasks:
            prompt = f"{c['lead_in']}\n\n{t['question']}\n\n{A.INSTRUCTION}"
            jobs.append((c["id"], t["id"], t["family"], t["answer"], prompt))
    return jobs


def collect(model):
    """model is a p3_model.P3Model. Resumable."""
    os.makedirs(OUT_DIR, exist_ok=True)
    conds, _ = load_conditions()
    with open("tasks.json", encoding="utf-8") as f:
        tasks = json.load(f)[:N_TASKS]
    path = raw_path(model.key)

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["condition"], r["task_id"]) for r in csv.DictReader(f)}
        print(f"  resuming, {len(done)} scores already saved")

    all_jobs = build_jobs(conds, tasks)
    jobs = [j for j in all_jobs if (j[0], j[1]) not in done]
    if not jobs:
        print("  channel A2 already complete")
        return
    print(f"  {len(jobs)} answers to score across {len(conds)} conditions")

    # self-check on five prompts spread over the lengths, before any row
    by_len = sorted(all_jobs, key=lambda j: len(j[4]) + len(str(j[3])))
    last = len(by_len) - 1
    sample = [by_len[round(last * q)] for q in (0, 0.25, 0.5, 0.75, 1)]
    print("  checking answer scoring against a plain forward pass")
    model.check_answer_scoring([j[4] for j in sample], [j[3] for j in sample])

    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["model", "model_key", "condition", "task_id", "family",
                    "answer_key", "n_answer_tokens", "logprob_sum",
                    "logprob_mean"])
    for i in range(0, len(jobs), FLUSH_EVERY):
        chunk = jobs[i:i + FLUSH_EVERY]
        try:
            scores = model.score_answers([j[4] for j in chunk],
                                         [j[3] for j in chunk])
        except Exception as e:
            print(f"    rows from {i} failed: {e}")
            continue
        for (cid, tid, fam, key, _p), (total, n) in zip(chunk, scores):
            per = total / n if n else float("nan")
            w.writerow([model.name, model.key, cid, tid, fam, key, n,
                        f"{total:.6f}", f"{per:.6f}"])
        f.flush()
        if (i // FLUSH_EVERY) % 50 == 0:
            print(f"    {i + len(chunk)} / {len(jobs)}")
    f.close()
    print(f"  saved to {path}")


def summary(model_key):
    """A first look. analyse_e1_answer_logprob.py is the read to trust."""
    path = raw_path(model_key)
    if not os.path.exists(path):
        print(f"  no results at {path}")
        return
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    conds, _ = load_conditions()
    by = {}
    for r in rows:
        v = float(r["logprob_sum"])
        if v == v:
            by.setdefault(r["condition"], {})[r["task_id"]] = v
    print()
    print("=" * 74)
    print(f"e1 CHANNEL A2, {model_key}: first look")
    print("=" * 74)
    print("  mean log P(correct answer), and the paired difference from NONE")
    print("  negative = the condition makes the right answer LESS likely")
    print()
    base = by.get("NONE", {})
    for c in conds:
        cells = by.get(c["id"])
        if not cells:
            continue
        ids = sorted(set(cells) & set(base))
        mean = sum(cells.values()) / len(cells)
        delta = (sum(cells[i] - base[i] for i in ids) / len(ids)) if ids else 0.0
        print(f"    {c['id']:14}n={len(cells):4}  {mean:9.4f}   vs NONE {delta:+.4f}")
    print()
    print("  Run analyse_e1_answer_logprob.py for the floors, the validity check")
    print("  against channel A and the interactions.")


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
