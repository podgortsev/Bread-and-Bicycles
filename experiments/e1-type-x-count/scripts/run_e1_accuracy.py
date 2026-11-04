#!/usr/bin/env python3
"""
run_e1_accuracy.py - experiment e1, channel A: can the model still solve the
person's own arithmetic task when signals stack.

    MODEL_KEY = "qwen"   then "llama"   then "mistral"
    17 conditions x 600 tasks = 10,200 generations per model,
    which is roughly two Colab sessions. Resumable, so a dead session costs
    nothing but the time already spent.

Upload next to this script in Colab:
    e1_conditions.json      from experiments/shared/conditions/
    tasks.json              from experiments/shared/tasks/

WHAT THIS MEASURES
------------------
The task question and its answer key are byte-identical in all seventeen
conditions. Only the two-clause lead-in changes. So any difference in accuracy
is a difference in how the model handled the same arithmetic for a differently
presented person.

This is the channel the literature has not run. Stacking has been measured on
the tone, reading level and stereotype content of generated text, and on
agreement with a simulated annotator. Whether the person's OWN request still
gets answered correctly when two signals are present is not on record.

WHAT THIS CHANNEL CANNOT DO, STATED BEFORE THE RUN
--------------------------------------------------
The outcome is binary, so this channel is far less sensitive to interactions
than to main effects: an interaction is a difference of differences and carries
about twice the variance of a single difference.

The task set was extended from Paper 2's 200 to 600 for exactly this reason.
Measured by validate_e1_accuracy.py, the power to detect a form-by-form
interaction of fifteen accuracy points goes from 22 percent at n=200 to 78
percent at n=600, and the minimum detectable interaction falls from about
twelve points to about seven and a half. Rows T001 to T200 are the Paper 2
tasks unchanged, so the two papers remain comparable on that subset.

Even at n=600 an interaction of five points is invisible here. That is a
property of a binary outcome and no affordable n fixes it.

The analysis therefore prints a minimum detectable interaction next to every
estimate, computed from the observed pairing. A null here is evidence about the
instrument until that number is read. Channels B and C carry a continuous logit
per item and are where an interaction of ordinary size can actually be seen.

REFUSAL
-------
Recorded, never scored as wrong. A refusal is its own outcome and it is one of
the things e1 is watching: Paper 2 found Llama refusing three times more often
when disability was mentioned, and the published claim under test is that
refusal rises as personalisation accumulates.

Parse order matters and is fixed: look for a usable number FIRST, then decide
refusal. Checking refusal language first discards hedged answers that contain a
correct number, which cost Paper 2 real data before it was fixed.
"""

import csv
import json
import os
import re
import sys

# ==========================================================================
# CHANGE ONLY THIS LINE BETWEEN RUNS
MODEL_KEY = "qwen"          # "qwen", then "llama", then "mistral"
# ==========================================================================

MODELS = {
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

OUT_DIR = "/content/drive/MyDrive/afl/p3_e1"
N_TASKS = 600
MAX_NEW_TOKENS = 160
BATCH = 8
LOAD_4BIT = True

INSTRUCTION = ("Answer with the final answer only. Do not show working. "
               "If the answer is a number, give just the number.")

REFUSAL_MARKERS = [
    "i can't", "i cannot", "i can not", "i'm unable", "i am unable",
    "i'm sorry", "i am sorry", "as an ai", "i don't have enough",
    "i do not have enough", "cannot determine", "can't determine",
    "not able to", "unable to help", "i'm not able", "i am not able",
]


# --------------------------------------------------------------------------
# parsing. No torch needed, so the validator exercises these directly.

def norm_number(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def find_numbers(text):
    cleaned = text.replace("$", "")
    return [norm_number(n) for n in re.findall(r"-?\d[\d,]*\.?\d*", cleaned)]


def is_correct(output, key, kind):
    out = output.strip()
    if kind == "number":
        target = norm_number(key)
        if target is None:
            return False
        return any(v is not None and abs(v - target) < 1e-6
                   for v in find_numbers(out))
    k = key.lower().replace(",", "").strip()
    o = out.lower().replace(",", "").strip()
    if k in o:
        return True
    p = k.split()
    if len(p) == 3 and p[0].isdigit():
        if f"{p[1]} {p[0]} {p[2]}" in o or f"{p[1]} {int(p[0])} {p[2]}" in o:
            return True
    return False


def parse(output, key, kind):
    """Returns (correct, refused). Number first, refusal second."""
    out = output.strip()
    if not out:
        return False, True
    correct = is_correct(out, key, kind)
    if correct:
        return True, False
    has_number = any(v is not None for v in find_numbers(out))
    if kind == "number" and has_number:
        return False, False          # wrong answer, not a refusal
    low = out.lower()
    refused = any(m in low for m in REFUSAL_MARKERS)
    if not refused and kind == "number" and not has_number:
        refused = True               # no number offered at all
    return False, refused


# --------------------------------------------------------------------------

CONDITIONS_FILE = "e1_conditions.json"
PREFIX = "e1_accuracy"


def load_conditions(conditions_file=None):
    with open(conditions_file or CONDITIONS_FILE, encoding="utf-8") as f:
        d = json.load(f)
    return d["conditions"], d["contrasts"]


def raw_path(model_key, prefix=None):
    return os.path.join(OUT_DIR, f"{prefix or PREFIX}_{model_key}.csv")


def collect(model, conditions_file=None, prefix=None):
    """model is a p3_model.P3Model, loaded once and shared by all channels.

    conditions_file and prefix let the floor extension run this instrument on a
    different condition set into a file of its own, so the rows the main run
    already collected are never touched and never recollected.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    conditions, _ = load_conditions(conditions_file)
    with open("tasks.json", encoding="utf-8") as f:
        tasks = json.load(f)[:N_TASKS]
    path = raw_path(model.key, prefix)

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {(r["condition"], r["task_id"]) for r in csv.DictReader(f)}
        print(f"resuming, {len(done)} rows already saved")

    todo = sum(1 for c in conditions for t in tasks
               if (c["id"], t["id"]) not in done)
    if todo == 0:
        print("all conditions complete")
        return
    print(f"{todo} generations to do\n")

    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["model", "model_key", "condition", "label", "style",
                    "n_social_signals", "n_words", "task_id", "family",
                    "correct", "refused", "answer_key", "raw_output"])

    for c in conditions:
        pending = [t for t in tasks if (c["id"], t["id"]) not in done]
        if not pending:
            print(f"  {c['id']:18}already complete")
            continue
        n_ok = n_ref = 0
        for i in range(0, len(pending), BATCH):
            chunk = pending[i:i + BATCH]
            prompts = [f"{c['lead_in']}\n\n{t['question']}\n\n{INSTRUCTION}"
                       for t in chunk]
            try:
                outs = model.generate(prompts, MAX_NEW_TOKENS)
            except Exception as e:
                print(f"  {c['id']} batch failed: {e}")
                continue
            for t, o in zip(chunk, outs):
                ok, ref = parse(o, t["answer"], t["answer_kind"])
                n_ok += int(ok)
                n_ref += int(ref)
                w.writerow([model.name, model.key, c["id"], c["label"],
                            c["style"], c["n_social_signals"], c["n_words"],
                            t["id"], t["family"], int(ok), int(ref),
                            t["answer"], o.strip().replace("\n", " ")[:400]])
            f.flush()
        print(f"  {c['id']:18}{n_ok:4}/{len(pending):<4}"
              f"{n_ok / len(pending):7.1%}   refused {n_ref}")
    f.close()
    print(f"\nsaved to {path}")


def summary(model_key, conditions_file=None, prefix=None):
    """A first look only. analyse_e1_accuracy.py is the read to trust.

    This used to take no argument while every caller passed the model key, so
    it raised on every run and the driver printed "summary unavailable". No data
    was affected; the first look was simply never shown.
    """
    path = raw_path(model_key, prefix)
    if not os.path.exists(path):
        print(f"  no results at {path}")
        return
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    conditions, contrasts = load_conditions(conditions_file)

    acc = {}
    for r in rows:
        acc.setdefault(r["condition"], {})[r["task_id"]] = int(r["correct"])
    ref = {}
    for r in rows:
        ref.setdefault(r["condition"], []).append(int(r["refused"]))

    print("\n" + "=" * 74)
    print(f"e1 CHANNEL A, {model_key}: first look")
    print("=" * 74)
    print("\n  condition            n   correct   refused")
    for c in conditions:
        cid = c["id"]
        if cid not in acc:
            continue
        v = list(acc[cid].values())
        print(f"  {cid:18}{len(v):4}{sum(v) / len(v):9.1%}"
              f"{sum(ref[cid]) / len(ref[cid]):10.1%}")

    print("\n  interaction per contrast, mean of (both - A) - (B - none)")
    print("  positive = costs LESS than the sum of parts (sub-additive)")
    for ct in contrasts:
        cells = [ct["both"], ct["single_a"], ct["single_b"], ct["none"]]
        if any(c not in acc for c in cells):
            continue
        ids = set.intersection(*[set(acc[c]) for c in cells])
        if not ids:
            continue
        d = [(acc[ct["both"]][i] - acc[ct["single_a"]][i])
             - (acc[ct["single_b"]][i] - acc[ct["none"]][i]) for i in ids]
        print(f"    {ct['name']:22}{ct['type']:18}"
              f"n={len(d):4}  mean {sum(d) / len(d):+.4f}")
    print("\n  Run analyse_e1_accuracy.py for the tested read with intervals,")
    print("  the minimum detectable interaction and the floor check.")


if __name__ == "__main__":
    import p3_model
    key = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-")         else MODEL_KEY
    if "--analyse-only" in sys.argv:
        summary(key)
    else:
        collect(p3_model.P3Model(key))
        summary(key)
