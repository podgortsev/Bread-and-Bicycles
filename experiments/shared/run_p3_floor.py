#!/usr/bin/env python3
"""
run_p3_floor.py - the floor extension for one model, in one notebook.

    !python run_p3_floor.py qwen        then llama, then mistral

Two additions agreed after the first analysis, collected together so the model
is loaded once:

  1. CHANNEL A2, new. The log probability of the correct answer, on the same
     prompts as channel A. A continuous behavioural read that may show single
     signal effects on qwen and mistral, where channel A's 0 or 1 showed none.
  2. TWO MORE NEUTRAL PAIRS for the floor, on every channel that has a floor.
     Until now each floor verdict rested on one pair, cycling and bread.

Every extension step writes to a file of its OWN. Nothing already collected is
reopened, appended to or recollected, whatever state Drive is in.

ORDER, CHEAPEST AND MOST FRAGILE FIRST

    e1c_floor   choice against NONE, new details    2,400 forward passes
    e1p_floor   choice pairs, new floors            1,600 forward passes
    e1b_floor   trait log prob, new floors         12,600 scored words
    e1a2        answer log prob, all conditions    13,800 scored answers
    e1a_floor   task accuracy, new floors           4,200 generations

Resumable like everything else: when the session dies, run the same command.

    !python run_p3_floor.py qwen --only=e1a2
"""

import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.getcwd(), HERE, os.path.join(ROOT, "e1-type-x-count", "scripts")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


FLOOR = "e1_floor_conditions.json"
OUT = "/content/drive/MyDrive/afl/p3_e1"


def steps():
    import run_e1_accuracy as ACC
    import run_e1_answer_logprob as ANS
    import run_e1_choice as CHOICE
    import run_e1_choice_pairs as PAIRS
    import run_e1_traits as TRAITS

    choice_kw = dict(conditions_file=FLOOR, prefix="e1_choice_floor",
                     reference="NONE", out_dir=OUT)
    pairs_kw = dict(conditions_file=FLOOR, prefix="e1_choice_pairs_floor")
    traits_kw = dict(conditions_file=FLOOR, prefix="e1_traits_floor",
                     coherence=False)
    acc_kw = dict(conditions_file=FLOOR, prefix="e1_accuracy_floor")

    return [
        ("e1c_floor", "choice against NONE, new neutral details",
         lambda m: CHOICE.collect(m, **choice_kw),
         lambda k: CHOICE.summary(k, **choice_kw),
         [FLOOR, "profiles.json"], "e1_choice_floor"),
        ("e1p_floor", "choice pairs, new floors",
         lambda m: PAIRS.collect(m, **pairs_kw),
         lambda k: PAIRS.summary(k, **pairs_kw),
         [FLOOR, "profiles.json"], "e1_choice_pairs_floor"),
        ("e1b_floor", "trait log probability, new floors",
         lambda m: TRAITS.collect(m, **traits_kw),
         lambda k: TRAITS.summary(k, **traits_kw),
         [FLOOR, "profiles.json"], "e1_traits_floor"),
        ("e1a2", "answer log probability, all 23 conditions",
         lambda m: ANS.collect(m),
         lambda k: ANS.summary(k),
         ["e1_conditions.json", FLOOR, "tasks.json"], "e1_answer_logprob"),
        ("e1a_floor", "task accuracy, new floors",
         lambda m: ACC.collect(m, **acc_kw),
         lambda k: ACC.summary(k, **acc_kw),
         [FLOOR, "tasks.json"], "e1_accuracy_floor"),
    ]


def check_inputs(selected):
    need = sorted({f for s in selected for f in s[4]})
    missing = [f for f in need if not os.path.exists(f)]
    print("input files")
    for f in need:
        print(f"  {'ok     ' if f not in missing else 'MISSING'}  {f}")
    if missing:
        print()
        print("Nothing was run. Unzip p3_floor.zip next to this script.")
        sys.exit(1)


def main():
    # imported here, not at the top, so the validator can import the step
    # list offline without torch
    import p3_model
    if len(sys.argv) < 2 or sys.argv[1] not in p3_model.MODELS:
        sys.exit(f"usage: run_p3_floor.py <{'|'.join(p3_model.MODELS)}> "
                 f"[--only=tag,tag]")
    key = sys.argv[1]
    only = None
    for a in sys.argv[2:]:
        if a.startswith("--only="):
            only = {x.strip().lower() for x in a.split("=", 1)[1].split(",")}

    selected = [s for s in steps() if only is None or s[0] in only]
    if not selected:
        sys.exit("no steps selected")

    print("=" * 78)
    print(f"PAPER 3 FLOOR EXTENSION, MODEL {key}")
    print("=" * 78)
    print(f"steps: {', '.join(s[0] for s in selected)}")
    print()
    check_inputs(selected)

    if not os.path.isdir("/content/drive/MyDrive"):
        print()
        print("WARNING: Drive does not look mounted, so results would be lost")
        print("when the session ends. Mount it first. Continuing in 10 seconds.")
        time.sleep(10)

    model = p3_model.P3Model(key)
    results = []
    for tag, name, collect, _summary, _files, _stem in selected:
        print()
        print("=" * 78)
        print(f"STEP {tag}: {name}")
        print("=" * 78)
        t0 = time.time()
        try:
            collect(model)
            results.append((tag, "done", time.time() - t0))
        except KeyboardInterrupt:
            print(f"  interrupted during {tag}; rerun to continue")
            results.append((tag, "interrupted", time.time() - t0))
            break
        except Exception:
            traceback.print_exc()
            print(f"  step {tag} FAILED. The remaining steps still run.")
            results.append((tag, "failed", time.time() - t0))

    print()
    print("=" * 78)
    print(f"COLLECTION SUMMARY, MODEL {key}")
    print("=" * 78)
    for tag, state, secs in results:
        print(f"  {tag:10}{state:12}{secs / 60:7.1f} min")

    for tag, _name, _collect, summary, _files, _stem in selected:
        try:
            summary(key)
        except Exception as e:
            print(f"  summary for {tag} unavailable: {e}")

    print()
    print("=" * 78)
    print("WHAT TO SEND BACK")
    print("=" * 78)
    for s in selected:
        print(f"    {OUT}/{s[5]}_{key}.csv")
    print("  and the console text from the first ==== line onwards.")


if __name__ == "__main__":
    main()
