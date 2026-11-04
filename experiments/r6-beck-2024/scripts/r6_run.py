#!/usr/bin/env python3
"""
r6_run.py - drive their sd_prompting.py, unchanged, over the Addendum 11 grid:
six models x seven conditions (no profile, five single attributes, all
attributes from the annotators' own profiles), prompt format 0, Diaz
(Addendum 11) or Diverse Perspectives (Addendum 13).
Skips any run whose output file already exists, so it resumes.

    python r6_run.py --repo <their clone> --data-dir <dir of the sample jsonl>
                     --out <dir> --cache <hf cache> [--formats 0]
                     [--task sentiment_diaz|toxicity_diverse_perspectives]
"""

import argparse
import os
import subprocess
import sys
import time

MODELS = ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
          "allenai/tk-instruct-small-def-pos", "allenai/tk-instruct-base-def-pos",
          "allenai/tk-instruct-large-def-pos"]
ATTRIBUTES = ["gender", "race", "education", "age", "political_affiliation", "ratings"]
TASKS = {  # task name -> their sample file, without the .jsonl
    "sentiment_diaz": "sentiment_diaz_filtered_sample_n1000_seed24",
    "toxicity_diverse_perspectives": "toxicity_diverse_perspectives_filtered_sample_n1000_seed24",
}


def out_name(stem, model, fmt, att):
    m = model.replace("/", "-")
    if att is None:
        return f"{stem}_without_sociodemographics_{m}_format{fmt}.jsonl"
    return f"{stem}_with_sociodemographics_{att}_{m}_format{fmt}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--formats", default="0")
    ap.add_argument("--task", default="sentiment_diaz", choices=sorted(TASKS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stem = TASKS[args.task]
    runs = []
    for fmt in args.formats.split(","):
        for model in MODELS:
            runs.append((model, fmt, None))
            runs += [(model, fmt, a) for a in ATTRIBUTES]
    for i, (model, fmt, att) in enumerate(runs, 1):
        if os.path.exists(os.path.join(args.out, out_name(stem, model, fmt, att))):
            print(f"[{i}/{len(runs)}] exists: {model} format {fmt} {att}", flush=True)
            continue
        cmd = [sys.executable, "sd_prompting.py", f"task_name={args.task}",
               f"input_dir={args.data_dir}", f"input_file={stem}", f"output_dir={args.out}",
               f"model_name={model}", f"format_id={fmt}", f"cache_dir={args.cache}",
               "test=false", f"sociodemographics={'false' if att is None else 'true'}"]
        if att is not None:
            cmd.append(f"sociodemographic_attribute={att}")
        t0 = time.time()
        print(f"[{i}/{len(runs)}] running: {model} format {fmt} {att}", flush=True)
        r = subprocess.run(cmd, cwd=args.repo, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-2000:], flush=True)
            sys.exit(f"failed: {model} {fmt} {att}")
        print(f"    done in {time.time() - t0:.0f} s", flush=True)
    print("all runs complete", flush=True)


if __name__ == "__main__":
    main()
