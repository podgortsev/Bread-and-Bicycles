#!/usr/bin/env python3
"""
build_e1_floor_conditions.py - two more neutral pairs for the floor check.
Writes e1_floor_conditions.json and e1_floor_conditions.csv next to itself.

WHY
---
Every interaction claim in Paper 3 is read against a floor: two socially neutral
but distinctive details stacked exactly the way two disclosures are. Until now
that floor was ONE pair, cycling to work and baking bread. A reviewer will ask
whether that pair is special, and Paper 2 already met a single control clause
that was not neutral at all: method 6b's CONTROL_ALT, which on Qwen won from the
disfavoured slot more often than any disclosure did.

This adds two more pairs, so each channel's floor verdict rests on three
independent neutral pairs rather than one.

WHAT IS HELD IDENTICAL TO e1
----------------------------
The same two-slot shape, the same neutral filler clauses n1 and n2, imported from
build_e1_conditions.py rather than retyped, and a NONE cell byte-identical to
e1's. NONE is collected again in every extension run, so that each new floor is
computed entirely inside one run, and so the two runs' NONE cells can be compared
as a check that collection is repeatable.

WHAT MAKES A CLAUSE NEUTRAL
---------------------------
Distinctive, so it is a real piece of information about the person, but carrying
no cue about age, disability, health, language, nationality, religion, family,
gender, class or education. Checked against a word list at build time.

A first draft used "crime novels". It was replaced by "history books": the genre
is harmless, the word "crime" is not, and two of the three instruments this floor
protects are lexical, so they read words rather than genres.
"""

import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_e1_conditions as B1      # noqa: E402

NEW_CLAUSES = {
    "alt3": "I walk our dog around the park every morning.",
    "alt4": "I have been learning to play the guitar lately.",
    "alt5": "I usually make a pot of tea around noon.",
    "alt6": "I have been reading a lot of history books.",
}

BANNED = {
    "age", "aged", "old", "older", "young", "younger", "retired", "retire",
    "pension", "seventy", "teen", "screen", "reader", "blind", "deaf", "adhd",
    "autistic", "autism", "dyslexic", "dyslexia", "disabled", "disability",
    "wheelchair", "impairment", "chronic", "illness", "ill", "sick", "pain",
    "doctor", "hospital", "therapy", "medication", "english", "language",
    "native", "foreign", "immigrant", "accent", "church", "mosque", "temple",
    "pray", "prayer", "wife", "husband", "mother", "father", "mum", "dad",
    "son", "daughter", "children", "kids", "baby", "pregnant", "rich", "poor",
    "cheap", "expensive", "salary", "degree", "university", "school", "crime",
    "criminal", "prison", "police",
}

# (id, slot1, slot2, label)
CONDITIONS = [
    ("NONE",        "n1",   "n2",   "no signal"),
    ("ALT3",        "alt3", "n2",   "neutral detail: dog walking"),
    ("ALT4_ONLY",   "n1",   "alt4", "neutral detail: guitar"),
    ("ALT_BOTH_34", "alt3", "alt4", "two neutral details: dog and guitar"),
    ("ALT5",        "alt5", "n2",   "neutral detail: tea"),
    ("ALT6_ONLY",   "n1",   "alt6", "neutral detail: history books"),
    ("ALT_BOTH_56", "alt5", "alt6", "two neutral details: tea and books"),
]

# (name, type, (both, single_a, single_b, none)), slot matched like e1's floor
CONTRASTS = [
    ("FLOOR2_ALTxALT", "floor", ("ALT_BOTH_34", "ALT3", "ALT4_ONLY", "NONE")),
    ("FLOOR3_ALTxALT", "floor", ("ALT_BOTH_56", "ALT5", "ALT6_ONLY", "NONE")),
]


def clause_text(name):
    if name in NEW_CLAUSES:
        return NEW_CLAUSES[name]
    return B1.CLAUSES[name]["std"]


def build():
    rows = []
    for cid, s1, s2, label in CONDITIONS:
        text = clause_text(s1) + " " + clause_text(s2)
        rows.append({
            "id": cid,
            "label": label,
            "lead_in": text,
            "slot1": s1,
            "slot2": s2,
            "style": "std",
            "stated_signals": [],
            "form_signals": [],
            "alt_details": [n for n in (s1, s2) if n in NEW_CLAUSES],
            "n_social_signals": 0,
            "n_words": len(text.split()),
        })
    return rows


def check(rows):
    problems = []
    e1 = {r["id"]: r for r in B1.build()}
    by_id = {r["id"]: r for r in rows}

    if len(by_id) != len(rows):
        problems.append("duplicate condition id")
    for r in rows:
        if r["id"] != "NONE" and r["id"] in e1:
            problems.append(f"{r['id']} collides with an e1 condition id")

    none = by_id.get("NONE")
    if none is None or none["lead_in"] != e1["NONE"]["lead_in"]:
        problems.append("NONE is not byte-identical to e1's NONE, so the new "
                        "floors would not share e1's reference")

    lens = [r["n_words"] for r in rows] + [r["n_words"] for r in e1.values()]
    if max(lens) - min(lens) > B1.MAX_WORD_SPREAD:
        problems.append(f"lead-in length spread across e1 and the floor set is "
                        f"{max(lens) - min(lens)} words, limit "
                        f"{B1.MAX_WORD_SPREAD}")

    for name, text in NEW_CLAUSES.items():
        words = set(re.findall(r"[a-z]+", text.lower()))
        hit = sorted(words & BANNED)
        if hit:
            problems.append(f"{name} carries a social cue: {hit}")
        if re.search(r"[0-9]", text):
            problems.append(f"{name} contains a number")

    texts = list(NEW_CLAUSES.values()) + [c["std"] for c in B1.CLAUSES.values()]
    if len(set(texts)) != len(texts):
        problems.append("a new neutral clause repeats an existing clause")

    for name, _ctype, (both, a, b, none_id) in CONTRASTS:
        cells = (both, a, b, none_id)
        if any(c not in by_id for c in cells):
            problems.append(f"contrast {name} names an unknown condition")
            continue
        if (set(by_id[both]["alt_details"])
                != set(by_id[a]["alt_details"]) | set(by_id[b]["alt_details"])):
            problems.append(f"contrast {name}: both-cell does not carry exactly "
                            f"the two singles' details")
        if by_id[none_id]["alt_details"]:
            problems.append(f"contrast {name}: none-cell carries a detail")
        if (by_id[a]["slot1"] != by_id[both]["slot1"]
                or by_id[b]["slot2"] != by_id[both]["slot2"]):
            problems.append(f"contrast {name}: not slot matched")
    return problems


def main():
    rows = build()
    problems = check(rows)
    print("=" * 78)
    print("e1 FLOOR EXTENSION: two more neutral pairs")
    print("=" * 78)
    for r in rows:
        print(f"  {r['id']:12}{r['n_words']:3}w  {r['lead_in']}")
    print()
    for name, _t, (both, a, b, none_id) in CONTRASTS:
        print(f"  {name:16}({both} - {a}) - ({b} - {none_id})")
    if problems:
        print()
        print("  PROBLEMS")
        for p in problems:
            print(f"    {p}")
        sys.exit(1)
    print()
    print("  all structural and neutrality checks pass")

    with open(os.path.join(HERE, "e1_floor_conditions.json"), "w",
              encoding="utf-8") as f:
        json.dump({"conditions": rows,
                   "contrasts": [{"name": n, "type": t, "both": c[0],
                                  "single_a": c[1], "single_b": c[2],
                                  "none": c[3]} for n, t, c in CONTRASTS]},
                  f, indent=2, ensure_ascii=False)
    with open(os.path.join(HERE, "e1_floor_conditions.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "slot1", "slot2", "n_words", "lead_in"])
        for r in rows:
            w.writerow([r["id"], r["label"], r["slot1"], r["slot2"],
                        r["n_words"], r["lead_in"]])
    print(f"  wrote e1_floor_conditions.json and .csv in {HERE}")


if __name__ == "__main__":
    main()
