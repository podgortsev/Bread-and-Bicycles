#!/usr/bin/env python3
"""
check_provenance.py - proves that everything Paper 3 inherited from Paper 2 is
still byte for byte what Paper 2 published.

    python check_provenance.py                 verify
    python check_provenance.py --record        write PROVENANCE.json (first time
                                               only, or after a deliberate change)

WHY A COPY AND NOT A POINTER
----------------------------
Paper 3 holds its own copies of Paper 2's datasets rather than referring to the
Paper 2 repository. That is deliberate, for three reasons.

1. Paper 2 is archived with a DOI. Its files cannot change, ever. Paper 3 needs
   600 tasks where Paper 2 has 200, so a shared file is not possible without
   modifying a published artefact.
2. Each paper is published as its own repository and archive. A reader who
   downloads one must be able to reproduce it without fetching another.
3. The programme compares numbers across papers, so the shared parts must be
   provably identical rather than merely similar.

The cost of a copy is silent drift, and this script is the answer to it. It
records the SHA-256 of every inherited file at the moment it was copied and
fails loudly if any of them changes afterwards.

WHAT IS INHERITED AND WHAT IS NEW

  profiles.json, profiles.csv   inherited whole, must never change
  axes.json, axes.csv           inherited whole, must never change
  axes_design.md                inherited whole, must never change
  tasks_p2_200.json             inherited whole, must never change
  tasks.json                    NEW: 600 rows whose first 200 are
                                tasks_p2_200.json unchanged. Checked by
                                build_tasks_p3.py at build time and again here.

Run this before any run that will be reported, and after any merge.
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = os.path.join(HERE, "PROVENANCE.json")

INHERITED = [
    "profiles/profiles.json",
    "profiles/profiles.csv",
    "axes/axes.json",
    "axes/axes.csv",
    "axes/axes_design.md",
    "tasks/tasks_p2_200.json",
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def prefix_check():
    """The 600-row set must open with Paper 2's 200 rows, unmodified."""
    p2 = os.path.join(HERE, "tasks", "tasks_p2_200.json")
    p3 = os.path.join(HERE, "tasks", "tasks.json")
    if not (os.path.exists(p2) and os.path.exists(p3)):
        return ["tasks_p2_200.json or tasks.json is missing"]
    a = json.load(open(p2, encoding="utf-8"))
    b = json.load(open(p3, encoding="utf-8"))
    problems = []
    if len(a) != 200:
        problems.append(f"tasks_p2_200.json has {len(a)} rows, expected 200")
    if len(b) < len(a):
        problems.append(f"tasks.json has {len(b)} rows, fewer than the 200 "
                        f"it must contain")
        return problems
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            problems.append(
                f"tasks.json row {i + 1} ({y.get('id')}) differs from the "
                f"Paper 2 row it must reproduce. Every accuracy number in "
                f"Paper 2's methods 3a, 3b and 3c was measured on that row.")
            break
    return problems


def main():
    record_mode = "--record" in sys.argv
    print("=" * 70)
    print("PAPER 3 PROVENANCE: what was inherited from Paper 2")
    print("=" * 70)

    now = {}
    missing = []
    for rel in INHERITED:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            missing.append(rel)
            continue
        now[rel] = sha(path)

    if record_mode:
        if missing:
            sys.exit(f"cannot record, missing: {', '.join(missing)}")
        json.dump({"source": "github/paper-2-single-human-ai-fit/experiments/shared",
                   "files": now}, open(RECORD, "w", encoding="utf-8"), indent=2)
        for rel, h in sorted(now.items()):
            print(f"  recorded  {rel:34}{h[:16]}")
        print(f"\n  wrote {RECORD}")
        return

    if not os.path.exists(RECORD):
        sys.exit(f"no {RECORD}. Run with --record once, from a state you trust.")
    want = json.load(open(RECORD, encoding="utf-8"))["files"]

    problems = []
    for rel in INHERITED:
        if rel in missing:
            problems.append(f"MISSING       {rel}")
            continue
        if rel not in want:
            problems.append(f"NOT RECORDED  {rel}")
        elif want[rel] != now[rel]:
            problems.append(f"CHANGED       {rel}")
        else:
            print(f"  identical  {rel:34}{now[rel][:16]}")
    for rel in want:
        if rel not in INHERITED:
            problems.append(f"RECORDED BUT NO LONGER TRACKED  {rel}")

    pref = prefix_check()
    if pref:
        problems.extend(pref)
    else:
        print("  identical  tasks.json rows 1-200 reproduce Paper 2 exactly")

    if problems:
        print("\n  PROBLEMS")
        for p in problems:
            print(f"    {p}")
        print("\n  A changed inherited file means Paper 3's numbers are no")
        print("  longer comparable with Paper 2's. Restore the file, or if the")
        print("  change is deliberate, say so in STATUS.md and re-record.")
        sys.exit(1)
    print("\n  everything inherited from Paper 2 is unchanged")


if __name__ == "__main__":
    main()
