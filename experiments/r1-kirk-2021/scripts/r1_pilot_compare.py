#!/usr/bin/env python3
"""
r1_pilot_compare.py - choose the generation length variant (Addendum 1).

    python r1_pilot_compare.py <kirk repo> <ner_A.csv> <ner_B.csv> <ner_C.csv>

Social templates only (base and religion). For each variant, three measures
against Kirk et al.'s released data for the same templates:

    missing   share of completions with no job title       (theirs about 0.109)
    multi     share of completions with two or more titles  (theirs about 0.033)
    tv        mean, over templates, of the total variation distance between
              our job distribution and theirs

Each is divided by its value in their data and the three are summed; the lowest
sum wins. For missing and multi that value is their share. For tv, the value "in
their data" can only be their own sampling noise: the total variation distance
between two disjoint random samples of 1,000 completions from the same template
of theirs, averaged over templates. This reading was written into this file
before any pilot result was seen.

Job distributions use the same lowercasing and job_replacements.csv mapping as
their make_freq_matrix, and count every title a completion returned.
"""

import ast
import collections
import csv
import os
import random
import sys

TEMPLATES = ["man", "woman", "Buddhist man", "Buddhist woman", "Christian man",
             "Christian woman", "Hindu man", "Hindu woman", "Jewish man",
             "Jewish woman", "Muslim man", "Muslim woman"]
SAMPLE = 1000
SEED = 20260912


def replacements(repo):
    out = {}
    with open(os.path.join(repo, "data", "shared_data", "job_replacements.csv"),
              encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("job") and r.get("update_match"):
                out[r["job"].lower()] = r["update_match"].lower()
    return out


def load(path, names):
    by = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Name"] in names:
                by[r["Name"]].append(ast.literal_eval(r["Title"]))
    return by


def distribution(title_lists, rep):
    c = collections.Counter()
    for titles in title_lists:
        for t in titles:
            t = t.lower()
            c[rep.get(t, t)] += 1
    total = sum(c.values())
    return {k: v / total for k, v in c.items()} if total else {}


def tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)


def shares(title_lists):
    n = len(title_lists)
    missing = sum(1 for t in title_lists if len(t) == 0) / n
    multi = sum(1 for t in title_lists if len(t) >= 2) / n
    return missing, multi


def main(repo, variant_files):
    rep = replacements(repo)
    theirs = load(os.path.join(repo, "data", "GPT-2", "NER_output",
                               "identity_occupations_template.csv"),
                  set(TEMPLATES))
    their_all = [t for name in TEMPLATES for t in theirs[name]]
    their_missing, their_multi = shares(their_all)

    rng = random.Random(SEED)
    noise = []
    for name in TEMPLATES:
        rows = theirs[name][:]
        rng.shuffle(rows)
        a, b = rows[:SAMPLE], rows[SAMPLE:2 * SAMPLE]
        noise.append(tv(distribution(a, rep), distribution(b, rep)))
    their_tv_noise = sum(noise) / len(noise)

    print("=" * 78)
    print("R1 PILOT: which generation length matches Kirk et al.'s data")
    print("social templates only: base and religion, 1,000 completions each")
    print("=" * 78)
    print(f"their data   missing {their_missing:.3f}   multi {their_multi:.3f}   "
          f"tv between two samples of theirs {their_tv_noise:.3f}")
    print()
    print(f"{'variant':9}{'missing':>9}{'multi':>9}{'mean tv':>9}"
          f"{'score':>9}")
    scores = {}
    for label, path in variant_files:
        ours = load(path, set(TEMPLATES))
        our_all = [t for name in TEMPLATES for t in ours[name]]
        missing, multi = shares(our_all)
        tvs = [tv(distribution(ours[name], rep), distribution(theirs[name], rep))
               for name in TEMPLATES]
        mean_tv = sum(tvs) / len(tvs)
        score = (abs(missing - their_missing) / their_missing
                 + abs(multi - their_multi) / their_multi
                 + mean_tv / their_tv_noise)
        scores[label] = score
        print(f"{label:9}{missing:9.3f}{multi:9.3f}{mean_tv:9.3f}{score:9.2f}")
    best = min(scores, key=scores.get)
    print()
    print(f"chosen variant: {best} (lowest score)")
    print("=" * 78)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(sys.argv[1], list(zip("ABC", sys.argv[2:5])))
