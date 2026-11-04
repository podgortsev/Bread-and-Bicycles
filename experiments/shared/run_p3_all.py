#!/usr/bin/env python3
"""
run_p3_all.py - the whole of Paper 3 for one model, in one notebook.

    !python run_p3_all.py qwen        then llama, then mistral

Three notebooks in total. The model is loaded once and every experiment reads
the same object, so a 4-bit load and its several minutes of disk traffic are
paid once instead of nine times.

ORDER, AND WHY IT IS THIS ORDER
-------------------------------
Cheapest and most fragile first, slowest last.

    1  e1 C   choice          6,400 forward passes   reads a letter probability
    2  e3 C   choice          4,000 forward passes   same instrument
    3  e1 B   traits         35,400 scored words     no generation
    4  e2b    legibility      2,400 generations
    5  e1 A   accuracy       10,200 generations      the paper's core channel
    6  e5     attribution       420 generations      REQUIRES e1 A
    7  e1w    wrappers        5,400 generations
    8  e2     depth curve     6,800 generations
    9  e3 D   numeric         6,600 generations

The two choice steps come first because they are the ones that can come back
unusable: they read logP("A") against logP("B"), and Paper 2 saw that fail on
Mistral three separate times, once at 61 percent of reads carrying too little
probability mass on the letters. Learning that in the first ten minutes is worth
more than learning it after the generation channels have eaten the session.

e5 must follow e1 channel A because it replays that channel's exchange as
history. If channel A has not finished, e5 says so and does nothing rather than
inventing a first turn.

RESUMABILITY
------------
Every step skips work already written to Drive. A free Colab session will not
finish a model in one sitting. When it dies: open a new session, redo the setup,
run the same command. Nothing is ever recomputed.

Roughly 38,000 generations and 46,000 forward passes per model, so expect three
to five sessions each. Use --only to pick up a single step.

    !python run_p3_all.py qwen --only=e1a
    !python run_p3_all.py qwen --skip=e2,e3d
"""

import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

# In Colab everything is uploaded flat, so the working directory is first.
for p in (os.getcwd(), HERE,
          os.path.join(ROOT, "e1-type-x-count", "scripts"),
          os.path.join(ROOT, "e1w-wrappers", "scripts"),
          os.path.join(ROOT, "e2-depth-curve", "scripts"),
          os.path.join(ROOT, "e2b-legibility", "scripts"),
          os.path.join(ROOT, "e3-offsetting", "scripts"),
          os.path.join(ROOT, "e5-attribution", "scripts")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import p3_model      # noqa: E402

# tag, human name, module name, function attribute, files it needs
STEPS = [
    ("e1c", "e1 channel C, head-to-head choice", "run_e1_choice",
     ["e1_conditions.json", "profiles.json"]),
    ("e3c", "e3 channel C, choice with a credential", "run_e3_choice",
     ["e3_conditions.json", "profiles.json"]),
    ("e1b", "e1 channel B, trait log probability", "run_e1_traits",
     ["e1_conditions.json", "profiles.json"]),
    ("e2b", "e2b, legibility test", "run_e2b_legibility",
     ["e1_conditions.json", "e2b_tasks.json"]),
    ("e1a", "e1 channel A, task accuracy", "run_e1_accuracy",
     ["e1_conditions.json", "tasks.json"]),
    ("e5", "e5, attribution under a stack", "run_e5_attribution",
     ["e1_conditions.json", "tasks.json"]),
    ("e1w", "e1w, six neutral wrappers", "run_e1w_wrappers",
     ["e1_conditions.json", "tasks.json"]),
    ("e2", "e2, depth curve", "run_e2_depth",
     ["e2_conditions.json", "tasks.json"]),
    ("e3d", "e3 channel D, numeric estimates", "run_e3_numeric",
     ["e3_conditions.json", "profiles.json"]),
]


def check_inputs(steps):
    need = sorted({f for _t, _n, _m, files in steps for f in files})
    missing = [f for f in need if not os.path.exists(f)]
    print("input files")
    for f in need:
        print(f"  {'ok  ' if f not in missing else 'MISSING'}  {f}")
    if missing:
        print("\nNothing was run. Unzip the bundle next to this script, or")
        print("upload the missing files. See HOW-TO.md.")
        sys.exit(1)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in p3_model.MODELS:
        sys.exit(f"usage: run_p3_all.py <{'|'.join(p3_model.MODELS)}> "
                 f"[--only=tag,tag] [--skip=tag,tag]")
    key = sys.argv[1]
    only = skip = None
    for a in sys.argv[2:]:
        if a.startswith("--only="):
            only = {x.strip().lower() for x in a.split("=", 1)[1].split(",")}
        elif a.startswith("--skip="):
            skip = {x.strip().lower() for x in a.split("=", 1)[1].split(",")}

    steps = [s for s in STEPS
             if (only is None or s[0] in only)
             and (skip is None or s[0] not in skip)]
    if not steps:
        sys.exit("no steps selected")

    print("=" * 78)
    print(f"PAPER 3, ALL EXPERIMENTS, MODEL {key}")
    print("=" * 78)
    print(f"steps: {', '.join(s[0] for s in steps)}\n")
    check_inputs(steps)

    if not os.path.isdir("/content/drive/MyDrive"):
        print("\nWARNING: Drive does not look mounted, so results would be")
        print("lost when the session ends. In a cell, first:")
        print("    from google.colab import drive; drive.mount('/content/drive')")
        print("Continuing in 10 seconds; interrupt to stop.")
        time.sleep(10)

    model = p3_model.P3Model(key)

    results = []
    modules = {}
    for tag, name, mod_name, _files in steps:
        print("\n" + "=" * 78)
        print(f"STEP {tag}: {name}")
        print("=" * 78)
        t0 = time.time()
        try:
            mod = __import__(mod_name)
            modules[tag] = mod
            mod.collect(model)
            results.append((tag, "done", time.time() - t0))
        except KeyboardInterrupt:
            print(f"\n  interrupted during {tag}. Everything already written")
            print("  is safe; rerun this command to continue.")
            results.append((tag, "interrupted", time.time() - t0))
            break
        except Exception:
            traceback.print_exc()
            print(f"\n  step {tag} FAILED. The remaining steps still run.")
            print("  Completed rows are never redone, so a rerun costs only")
            print("  what this step had left.")
            results.append((tag, "failed", time.time() - t0))

    print("\n" + "=" * 78)
    print(f"COLLECTION SUMMARY, MODEL {key}")
    print("=" * 78)
    total = 0.0
    for tag, state, secs in results:
        total += secs
        print(f"  {tag:6}{state:12}{secs / 60:7.1f} min")
    print(f"  {'TOTAL':6}{'':12}{total / 60:7.1f} min")

    for tag, name, mod_name, _files in steps:
        mod = modules.get(tag)
        if mod is None:
            continue
        try:
            mod.summary(key)
        except Exception as e:
            print(f"\n  summary for {tag} unavailable: {e}")

    print("\n" + "=" * 78)
    print("WHAT TO SEND BACK")
    print("=" * 78)
    print("  every CSV under /content/drive/MyDrive/afl/ for this model:")
    for sub, names in (("p3_e1", ["e1_choice", "e1_traits", "e1_accuracy"]),
                       ("p3_e1w", ["e1w_wrappers"]),
                       ("p3_e2", ["e2_depth"]),
                       ("p3_e2b", ["e2b_legibility"]),
                       ("p3_e3", ["e3_choice", "e3_numeric"]),
                       ("p3_e5", ["e5_attribution"])):
        for n in names:
            print(f"    /content/drive/MyDrive/afl/{sub}/{n}_{key}.csv")
    print("  and the console text from the first ==== line onwards.")
    print("  Not the model-loading logs, they carry nothing.")


if __name__ == "__main__":
    main()
