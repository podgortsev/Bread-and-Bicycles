#!/usr/bin/env python3
"""
run_e2_depth.py - e2: does the penalty grow with the NUMBER of signals.

    17 conditions x 400 tasks = 6,800 generations per model.

Upload next to this script in Colab:
    e2_conditions.json      from experiments/shared/conditions/
    tasks.json              from experiments/shared/tasks/

Paper 2's method 3b rebuilt. The design and every change from m3b are documented
in `shared/conditions/build_e2_conditions.py`; the short version is two ladders
instead of one mixed set, three models instead of two, signals fixed by type
rather than picked for having shown an effect, a constant three clauses at every
depth, and a shape judged by a fitted second difference rather than by a
threshold someone chose.

WHY 400 TASKS AND NOT 600
-------------------------
e1 needs 600 because it estimates a single interaction as precisely as a binary
outcome allows. e2 estimates a CURVE from four points, and the question it
answers is about the shape across routes rather than the precision of any one
point. 400 keeps this to a single session per model and leaves the compute for
e1, which is where the paper's claim lives.

The first 400 rows of tasks.json are used, which includes all 200 of Paper 2's
own tasks, so the depth-0 baseline here is directly comparable with method 3b's.

WHAT m3b GOT WRONG AND THIS DOES NOT
------------------------------------
Its own result document lists the defects, and they are worth repeating because
this run exists to avoid them:

  * depth 1 and depth 3 were compared with a Mann-Whitney U across routes, which
    treats them as independent groups when the same routes carry both. Corrected
    to a paired Wilcoxon, the headline p went from 0.031 to 0.0625.
  * the shape was labelled by a heuristic with arbitrary thresholds.
  * no multiplicity correction was applied at depth.
  * the eight routes were not eight independent signal sets.

The analyser here does the paired test, fits the shape, corrects across the
family, and prints how many independent routes stand behind every point.
"""

import csv
import json
import os
import sys

OUT_DIR = "/content/drive/MyDrive/afl/p3_e2"
N_TASKS = 400
MAX_NEW_TOKENS = 160
BATCH = 8

INSTRUCTION = ("Answer with the final answer only. Do not show working. "
               "If the answer is a number, give just the number.")


def load_conditions():
    with open("e2_conditions.json", encoding="utf-8") as f:
        return json.load(f)["conditions"]


def raw_path(model_key):
    return os.path.join(OUT_DIR, f"e2_depth_{model_key}.csv")


def collect(model):
    """model is a p3_model.P3Model. Resumable."""
    import run_e1_accuracy as A
    os.makedirs(OUT_DIR, exist_ok=True)
    conditions = load_conditions()
    with open("tasks.json", encoding="utf-8") as f:
        tasks = json.load(f)[:N_TASKS]
    path = raw_path(model.key)

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["condition"], r["task_id"]) for r in csv.DictReader(f)}
        print(f"  resuming, {len(done)} rows already saved")

    todo = [(c, t) for c in conditions for t in tasks
            if (c["id"], t["id"]) not in done]
    if not todo:
        print("  e2 already complete")
        return
    print(f"  {len(todo)} generations to do")

    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["model", "model_key", "condition", "ladder", "depth",
                    "route", "task_id", "family", "correct", "refused",
                    "answer_key", "raw_output"])

    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        prompts = [f"{c['lead_in']}\n\n{t['question']}\n\n{INSTRUCTION}"
                   for c, t in chunk]
        try:
            outs = model.generate(prompts, MAX_NEW_TOKENS)
        except Exception as e:
            print(f"    batch at {i} failed: {e}")
            continue
        for (c, t), o in zip(chunk, outs):
            ok, ref = A.parse(o, t["answer"], t["answer_kind"])
            w.writerow([model.name, model.key, c["id"], c["ladder"],
                        c["depth"], c["route"], t["id"], t["family"],
                        int(ok), int(ref), t["answer"],
                        o.strip().replace("\n", " ")[:400]])
        f.flush()
        if (i // BATCH) % 50 == 0:
            print(f"    {i + len(chunk)} / {len(todo)}")
    f.close()
    print(f"  saved to {path}")


def summary(model_key):
    path = raw_path(model_key)
    if not os.path.exists(path):
        print(f"  no results at {path}")
        return
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    acc, meta = {}, {}
    for r in rows:
        acc.setdefault(r["condition"], {})[r["task_id"]] = int(r["correct"])
        meta[r["condition"]] = (r["ladder"], int(r["depth"]))

    print("\n" + "=" * 74)
    print(f"e2 DEPTH CURVE, {model_key}: first look")
    print("=" * 74)
    base = acc.get("D0", {})
    b = sum(base.values()) / len(base) if base else 0.0
    print(f"\n  depth 0 baseline: {b:.1%}\n")
    for ladder in ("form", "stated"):
        print(f"  {ladder} ladder")
        for depth in (1, 2, 3):
            sel = [c for c in acc if meta.get(c, ("", 0)) == (ladder, depth)]
            if not sel:
                continue
            nets = []
            for c in sel:
                ids = sorted(set(acc[c]) & set(base))
                nets.append(sum(acc[c][i] - base[i] for i in ids) / len(ids))
            print(f"    depth {depth}: mean net {sum(nets) / len(nets):+.4f} "
                  f"over {len(nets)} condition(s)  "
                  f"[{min(nets):+.3f}, {max(nets):+.3f}]")
        print()
    print("  Run analyse_e2_depth.py for the fitted shape, the second")
    print("  difference with an interval, and the paired depth comparisons.")


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
