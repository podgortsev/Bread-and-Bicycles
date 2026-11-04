#!/usr/bin/env python3
"""
r1_ner.py - job titles from completions, with Stanford CoreNLP 4.2.0 NER.

    python r1_ner.py --completions completions.csv --out ner.csv \
        --corenlp <dir with stanford-corenlp-4.2.0 jars> --java <java.exe>

Preregistration R1, Step 2. Kirk et al. extracted job titles with the Stanford
CoreNLP named entity recognizer and kept every title a sentence returned. This
starts a local CoreNLP server, sends sentences in batches, and keeps every
entity mention whose type is TITLE.

The output has their released file's shape: an index column, `Name`, and `Title`
as a Python list literal (for example "['waitress']", or "[]" when no title was
found), so their own `clean_up_data` and `make_freq_matrix` read it unchanged.

Resumable: sentences already in the output are skipped by row number.
"""

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request

PORT = 9011
BATCH = 400
PROPS = {"annotators": "tokenize,ssplit,pos,lemma,ner",
         "ssplit.eolonly": "true", "outputFormat": "json",
         "ner.useSUTime": "false"}


def wait_for(port, seconds=180):
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(2)
    return False


def annotate(lines):
    url = (f"http://127.0.0.1:{PORT}/?properties="
           + urllib.parse.quote(json.dumps(PROPS)))
    data = "\n".join(lines).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        doc = json.loads(resp.read().decode("utf-8"))
    sents = doc["sentences"]
    if len(sents) != len(lines):
        raise RuntimeError(f"{len(lines)} lines in, {len(sents)} sentences out")
    out = []
    for s in sents:
        titles = [m["text"] for m in s.get("entitymentions", [])
                  if m.get("ner") == "TITLE"]
        out.append(titles)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--completions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corenlp", required=True)
    ap.add_argument("--java", required=True)
    args = ap.parse_args()

    with open(args.completions, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    done = 0
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = sum(1 for _ in csv.DictReader(f))
    todo = rows[done:]
    print(f"{len(rows)} completions, {done} already tagged, {len(todo)} to tag")
    if not todo:
        return 0

    # one sentence per line: a newline inside a completion would split it
    lines = [" ".join(r["sentence"].split()) for r in todo]

    server = subprocess.Popen(
        [args.java, "-mx6g", "-cp", os.path.join(args.corenlp, "*"),
         "edu.stanford.nlp.pipeline.StanfordCoreNLPServer",
         "-port", str(PORT), "-timeout", "600000", "-quiet"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_for(PORT):
            raise RuntimeError("CoreNLP server did not start")
        new_file = done == 0
        f = open(args.out, "a", newline="", encoding="utf-8")
        w = csv.writer(f)
        if new_file:
            w.writerow(["", "Name", "Title"])
        index = done
        for start in range(0, len(lines), BATCH):
            chunk = lines[start:start + BATCH]
            titles = annotate(chunk)
            for r, t in zip(todo[start:start + BATCH], titles):
                w.writerow([index, r["Name"], repr(t)])
                index += 1
            f.flush()
            if (start // BATCH) % 25 == 0:
                print(f"  {index} / {len(rows)}")
        f.close()
    finally:
        server.terminate()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
