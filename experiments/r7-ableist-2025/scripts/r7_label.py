#!/usr/bin/env python3
"""
r7_label.py - Addendum 10: run the ABLEIST authors' open LoRA labeller over the
conversations their two open models produced, and compare with their labels.

    python r7_label.py --data <full dataset csv> --ableist <clone of ABLEIST>
                       --out <dir> [--limit N] [--model ableist:q4]

Their prompt goes to a local Ollama serving their adapter merged into
Llama-3.1-8B-Instruct and quantised to Q4_K_M. Two bitsandbytes routes were
tried first on this 6 GB GPU: the output layer on the CPU, and the output layer
quantised on the GPU. Both ran at minutes per conversation, because a 128k-row
output layer is moved or dequantised for every generated token.

Prints aggregates only. The conversations are not copied, quoted or written out;
the per-conversation file holds the eight predicted labels and the row index.
Resumable: rows already in the output are skipped.
"""

import argparse
import csv
import os
import re
import shutil
import sys
import time

import pandas as pd
import requests

OPEN_MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "allenai/OLMo-2-1124-7B-Instruct"]
METRICS = ["One-size-fits-all Ableism", "Infantilization", "Technoableism",
           "Anticipated Ableism", "Ability Saviorism", "Tokenism",
           "Inspiration Porn", "Superhumanization Harm"]


def their_instruction(ableist_repo):
    # the fine-tuned model has its own prompt file, next to its inference script
    path = os.path.join(ableist_repo, "data-labeling", "ableist-labeling",
                        "lora-inference", "utils", "prompt.py")
    src = open(path, encoding="utf-8").read()
    m = re.search(r"ZERO_SHOT_COMBINED_PROMPT = f'''(.*?)'''", src, re.S)
    if not m:
        sys.exit("ZERO_SHOT_COMBINED_PROMPT not found in their prompts.py")
    return m.group(1)


def parse(response):
    """Their reading of the YAML block: metric -> 0 or 1, absent counts as 0."""
    out = {}
    for metric in METRICS:
        m = re.search(re.escape(metric) + r"\s*:\s*([01])", response)
        out[metric] = int(m.group(1)) if m else 0
    return out


def macro_f1(pred, true):
    """Unweighted mean over metrics of the F1 for the positive class."""
    scores = []
    for metric in METRICS:
        tp = sum(1 for p, t in zip(pred, true) if p[metric] == 1 and t[metric] == 1)
        fp = sum(1 for p, t in zip(pred, true) if p[metric] == 1 and t[metric] == 0)
        fn = sum(1 for p, t in zip(pred, true) if p[metric] == 0 and t[metric] == 1)
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ableist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="ableist:q4")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--mirror", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "r7_labeller_predictions.csv")

    df = pd.read_csv(args.data)
    df = df[df["LLM"].isin(OPEN_MODELS)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    done = {}
    if os.path.exists(path):
        for row in csv.DictReader(open(path, encoding="utf-8")):
            done[int(row["index"])] = {m: int(row[m]) for m in METRICS}
    print(f"{len(df)} conversations from the open models, {len(done)} already labelled",
          flush=True)

    instruction = their_instruction(args.ableist)
    sep = "### Response:"

    def label(conversation):
        prompt = ("### Instruction:\n" + instruction + "\n\n### Input:\n"
                  + str(conversation) + "\n\n" + sep + "\n")
        payload = {"model": args.model, "prompt": prompt, "raw": True,
                   "stream": False,
                   "options": {"temperature": 0.1, "num_predict": 512,
                               "num_ctx": 4096}}
        for attempt in range(3):
            try:
                r = requests.post(args.url.rstrip("/") + "/api/generate",
                                  json=payload, timeout=900)
                r.raise_for_status()
                return r.json()["response"]
            except requests.exceptions.RequestException as e:
                print(f"  request failed ({attempt + 1}): {e}", flush=True)
                time.sleep(5)
        sys.exit("the server stopped answering")

    new_file = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["index", "LLM"] + METRICS)
    t0 = time.time()
    made = 0
    for _, row in df.iterrows():
        if int(row["index"]) in done:
            continue
        text = label(row["Response"])
        if sep in text:
            text = text.split(sep)[-1]
        pred = parse(text.strip())
        done[int(row["index"])] = pred
        made += 1
        w.writerow([int(row["index"]), row["LLM"]] + [pred[m] for m in METRICS])
        f.flush()
        if made % 25 == 0:
            rate = made / (time.time() - t0) * 3600
            print(f"  {len(done)}/{len(df)}  {rate:.0f} per hour", flush=True)
            if args.mirror:
                shutil.copy(path, args.mirror + ".tmp")
                os.replace(args.mirror + ".tmp", args.mirror)
    f.close()
    if args.mirror:
        shutil.copy(path, args.mirror)

    truth = {int(r["index"]): {m: int(float(r[m])) for m in METRICS}
             for _, r in df.iterrows()}
    keys = [k for k in truth if k in done]
    pred = [done[k] for k in keys]
    true = [truth[k] for k in keys]
    overall, per_metric = macro_f1(pred, true)
    print("=" * 78)
    print(f"R7 LABELLER CHECK (Addendum 10): {len(keys)} conversations")
    print("=" * 78)
    for metric, score in zip(METRICS, per_metric):
        base_rate = sum(t[metric] for t in true) / len(true)
        ours = sum(p[metric] for p in pred) / len(pred)
        agree = sum(1 for p, t in zip(pred, true) if p[metric] == t[metric]) / len(true)
        print(f"  {metric:28} F1 {score:.3f}   agreement {agree:.3f}   "
              f"their positives {base_rate:.3f}   ours {ours:.3f}")
    print(f"\n  macro F1 {overall:.3f} (pass at 0.80)")
    print(f"  VERDICT: {'PASS' if overall >= 0.80 else 'FAIL: their labeller not reproduced'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
