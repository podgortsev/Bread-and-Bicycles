#!/usr/bin/env python3
"""
r3_step0.py - Step 0 for Wan and Chang 2025, LABE (ACL 2025): their Table 1
from their released generations, before anything neutral is generated.

    python r3_step0.py --repo <clone of elainew728/labe-agency> --work <dir>
                       [--stage all|train|infer|calc]

Preregistration Addendum 5, R3. Their scripts are copied into the working
directory and edited only by the declared, exact string replacements below;
each must match exactly once or the script stops. Resumable: finished
checkpoints and inference outputs are skipped.

    train   fine_tune_bert.py, seeds 0 1 2    -> checkpoints/bert-base-cased_<s>/checkpoint-4500
    infer   inference_bert.py, llama3 and mistral x llm_bios, llm_professor,
            llm_rec_letter x classifiers 0 1 2
    calc    calc_agency_aggregated.py unchanged in method; prints ours against
            their Table 1 and the Addendum 5 verdict

Pinned commit: e8cc42d86df007fd05e3ae0c27c127b7a0a6165c.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Their Table 1 (ACL 2025 PDF, p. 5): gender, race, intersectional variance
TABLE1 = {
    ("mistral", "llm_bios"): (60.29, 29.99, 61.36),
    ("mistral", "llm_professor"): (36.61, 48.33, 63.14),
    ("mistral", "llm_rec_letter"): (59.06, 7.90, 45.63),
    ("llama3", "llm_bios"): (37.10, 26.82, 47.40),
    ("llama3", "llm_professor"): (68.31, 85.51, 125.00),
    ("llama3", "llm_rec_letter"): (44.93, 26.29, 49.94),
}
THEIR_VAL_ACC_EPOCH9 = {0: 0.933, 1: 0.933, 2: 0.917}
DATA_COPIES = {  # released name -> name inference_bert.py opens
    "{m}_clg_bios.csv": "{m}_clg_llm_bios.csv",
    "{m}_clg_professor.csv": "{m}_clg_llm_professor.csv",
    "{m}_clg_letters.csv": "{m}_clg_llm_letters.csv",
}
MODELS = ["llama3", "mistral"]
DATASETS = ["llm_bios", "llm_professor", "llm_rec_letter"]
SEEDS = [0, 1, 2]


def patch(src, dst, edits):
    text = open(src, encoding="utf-8").read()
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            sys.exit(f"patch of {os.path.basename(src)}: expected 1 match, "
                     f"found {n}: {old!r}")
        text = text.replace(old, new)
    open(dst, "w", encoding="utf-8").write(text)


TRAIN_EDITS = [
    # CPU speed: dynamic padding (attention masks make padding inert)
    ('return tokenizer(examples["text"], padding="max_length", truncation=True)',
     'return tokenizer(examples["text"], truncation=True)'),
    ("from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer",
     "from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding"),
    ("    compute_metrics=compute_metrics,\n)",
     "    compute_metrics=compute_metrics,\n"
     "    data_collator=DataCollatorWithPadding(tokenizer),\n)"),
    # checkpoints only at the step their inference reads
    ("                                  learning_rate=5e-6)",
     "                                  learning_rate=5e-6,\n"
     "                                  save_strategy=\"steps\", save_steps=4500,\n"
     "                                  save_only_model=True)"),
]
INFER_EDITS = [
    # CPU: no hard-coded cuda; one sentence at a time needs no padding
    ("inputs = tokenizer(s, padding=\"max_length\", truncation=True, return_tensors='pt').to(\"cuda\")",
     "inputs = tokenizer(s, truncation=True, return_tensors='pt').to(device)"),
]
CALC_EDITS = [
    # the race branch is unreachable for the output paths their scripts write
    ("    if 'race' in args.input_file:", "    if True:"),
]


def run(cmd, cwd, log):
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        r = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f"failed ({r.returncode}): {' '.join(cmd)}; see {log}")


def prepare(repo, work):
    os.makedirs(os.path.join(work, "data"), exist_ok=True)
    os.makedirs(os.path.join(work, "outputs"), exist_ok=True)
    lac = os.path.join(work, "lac_dataset_construction", "lac_dataset")
    if not os.path.exists(lac):
        shutil.copytree(os.path.join(repo, "lac_dataset_construction", "lac_dataset"), lac)
    for m in MODELS:
        for a, b in DATA_COPIES.items():
            dst = os.path.join(work, "data", b.format(m=m))
            if not os.path.exists(dst):
                shutil.copy(os.path.join(repo, "data", a.format(m=m)), dst)
    patch(os.path.join(repo, "fine_tune_bert.py"),
          os.path.join(work, "fine_tune_bert_p8.py"), TRAIN_EDITS)
    patch(os.path.join(repo, "inference_bert.py"),
          os.path.join(work, "inference_bert_p8.py"), INFER_EDITS)
    patch(os.path.join(repo, "calc_agency_aggregated.py"),
          os.path.join(work, "calc_agency_aggregated_p8.py"), CALC_EDITS)


def ckpt(work, s):
    return os.path.join(work, "checkpoints", f"bert-base-cased_{s}", "checkpoint-4500")


def train(work, log):
    for s in SEEDS:
        if os.path.exists(os.path.join(ckpt(work, s), "config.json")):
            print(f"  classifier {s}: checkpoint exists")
            continue
        print(f"  classifier {s}: training")
        run([sys.executable, "fine_tune_bert_p8.py", "-s", str(s)], work, log)


def val_accuracy(work, s):
    path = os.path.join(ckpt(work, s), "trainer_state.json")
    if not os.path.exists(path):
        return None
    hist = json.load(open(path))["log_history"]
    ev = [h for h in hist if "eval_accuracy" in h]
    if not ev:
        return None
    last = ev[-1]["eval_accuracy"]
    return last["accuracy"] if isinstance(last, dict) else last


def infer(work, log):
    for m in MODELS:
        for d in DATASETS:
            for s in SEEDS:
                out = os.path.join(work, "outputs", f"{m}_{d}", f"bert-base-cased_results_{s}.csv")
                if os.path.exists(out):
                    continue
                print(f"  inference {m} {d} classifier {s}")
                run([sys.executable, "inference_bert_p8.py", "-c", ckpt(work, s),
                     "-d", d, "--seed", str(s), "-gm", m], work, log)


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def calc(work, log):
    ours = {}
    for m in MODELS:
        for d in DATASETS:
            inp = f"./outputs/{m}_{d}/bert-base-cased_results_0.csv"
            r = subprocess.run([sys.executable, "calc_agency_aggregated_p8.py",
                                "--input_file", inp, "-gm", m], cwd=work,
                               capture_output=True, text=True)
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"\n$ calc {m} {d}\n{r.stdout}\n{r.stderr}\n")
            if r.returncode != 0:
                sys.exit(f"calc failed for {m} {d}; see {log}")
            vals = []
            for lab in ("Gender Var.", "Racial Var.", "Intersectional Var."):
                mm = re.search(re.escape(lab) + r":\s+([-0-9.]+)", r.stdout)
                vals.append(float(mm.group(1)))
            ours[(m, d)] = tuple(vals)

    print("=" * 78)
    print("R3 LABE STEP 0: their Table 1 against ours (Addendum 5)")
    print("=" * 78)
    print("classifier validation accuracy at the last epoch evaluated:")
    for s in SEEDS:
        a = val_accuracy(work, s)
        print(f"  classifier {s}: ours {a if a is None else round(a, 3)}   "
              f"theirs (epoch 9) {THEIR_VAL_ACC_EPOCH9[s]}")
    print(f"\n  {'model':8}{'task':16}{'':6}{'gender':>16}{'race':>16}{'intersect.':>16}")
    t_all, o_all = [], []
    for (m, d), t in TABLE1.items():
        o = ours[(m, d)]
        t_all += list(t)
        o_all += list(o)
        print(f"  {m:8}{d:16}theirs{t[0]:16.2f}{t[1]:16.2f}{t[2]:16.2f}")
        print(f"  {'':8}{'':16}ours  {o[0]:16.2f}{o[1]:16.2f}{o[2]:16.2f}")
        top_t = ["gender", "race", "intersectional"][int(np.argmax(t))]
        top_o = ["gender", "race", "intersectional"][int(np.argmax(o))]
        print(f"  {'':8}{'':16}largest: theirs {top_t}, ours {top_o}")
    t_all, o_all = np.array(t_all), np.array(o_all)
    rho = spearman(t_all, o_all)
    mard = float(np.median(np.abs(o_all - t_all) / t_all))
    ok = rho >= 0.6 and mard <= 0.25
    print(f"\n  Spearman over 18 values {rho:.3f} (pass >= 0.6)")
    print(f"  median absolute relative difference {mard:.3f} (pass <= 0.25)")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL: instrument not shown equivalent'}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--stage", default="all", choices=["all", "train", "infer", "calc"])
    args = ap.parse_args()
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    log = os.path.join(work, "r3_step0.log")
    prepare(args.repo, work)
    if args.stage in ("all", "train"):
        train(work, log)
    if args.stage in ("all", "infer"):
        infer(work, log)
    if args.stage in ("all", "calc"):
        calc(work, log)


if __name__ == "__main__":
    main()
