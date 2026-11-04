#!/usr/bin/env python3
"""
r5_nioccs.py - SOC codes for the resume and job-description datasets from the
CDC NIOCCS web API, cached so it resumes and never asks twice for one text.

    python r5_nioccs.py --resumes Resume.csv --jobs JD_data.csv --cache nioccs_cache.jsonl

Wilson and Caliskan (AIES 2024) coded "free text job titles and descriptions"
with NIOCCS and dropped codes below 60 percent confidence; which text they sent
is not stated. This step only collects API answers for the candidate inputs:
    resume  first line of Resume_str (the title), and the Category label
    job     'position', and 'job'
Choosing among them is registered before any count is compared with Table 1.
"""

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

URL = "https://wwwn.cdc.gov/nioccs/IOCode?i=&o={o}&c=2"


def first_line(text):
    for line in str(text).splitlines():
        if line.strip():
            return " ".join(line.split())
    return ""


def second_line(text):
    lines = [l for l in str(text).splitlines() if l.strip()]
    return " ".join(lines[1].split()) if len(lines) > 1 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resumes", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    r = pd.read_csv(args.resumes)
    j = pd.read_csv(args.jobs)
    texts = set()
    texts |= {first_line(t)[:300] for t in r["Resume_str"]}
    # Addendum 9 readings D to G
    texts |= {second_line(t)[:300] for t in r["Resume_str"]}
    texts |= {" ".join(str(t).split())[:300] for t in r["Resume_str"]}
    texts |= {" ".join(str(t).split())[:1000] for t in r["Resume_str"]}
    texts |= {(first_line(t)[:300] + " " + c.replace("-", " ").lower())
              for t, c in zip(r["Resume_str"], r["Category"])}
    texts |= {c.replace("-", " ").lower() for c in r["Category"]}
    texts |= {str(t) for t in j["position"].fillna("")}
    texts |= {str(t) for t in j["job"].fillna("")}
    texts.discard("")
    done = set()
    if os.path.exists(args.cache):
        for line in open(args.cache, encoding="utf-8"):
            done.add(json.loads(line)["text"])
    todo = sorted(texts - done)
    print(f"{len(texts)} distinct texts, {len(done)} cached, {len(todo)} to code", flush=True)
    lock = threading.Lock()
    count = [0]

    def code(t):
        for attempt in range(4):
            try:
                with urllib.request.urlopen(URL.format(o=urllib.parse.quote(t)), timeout=90) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as e:
                out = {"error": str(e)[:200]}
                time.sleep(5 * (attempt + 1))
        with lock:
            f.write(json.dumps({"text": t, "response": out}) + "\n")
            f.flush()
            count[0] += 1
            if count[0] % 200 == 0:
                print(f"  {count[0]}/{len(todo)}", flush=True)

    with open(args.cache, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(code, todo))
    print("done", flush=True)


if __name__ == "__main__":
    main()
