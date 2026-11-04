#!/usr/bin/env python3
"""
build_e1_conditions.py - the seventeen conditions of experiment e1.

Writes e1_conditions.json and e1_conditions.csv next to itself. No GPU, no
model, no network. Every runner in e1 reads the JSON; nothing constructs a
lead-in on its own.

THE SHAPE, AND WHY
------------------
Every lead-in is exactly TWO clauses. Always two, in every condition. That is
the whole trick of this design.

Paper 2 padded lead-ins with neutral filler to match word count. That controls
length but not clause count, and it leaves open the objection that a stacked
condition simply says more things. Here the number of clauses is constant by
construction, so "two signals" never means "one more sentence". A signal
occupies a slot that would otherwise hold a neutral clause; it never adds one.

TWO KINDS OF SIGNAL, AND HOW EACH ONE OCCUPIES A SLOT
-----------------------------------------------------
STATED signals are WHAT a clause says. S01 fills a slot with "I use a screen
reader for everything on my computer" where a neutral clause would have gone.

FORM signals are HOW the clauses are written. F01 (non-native English) and F04
(typos) restyle BOTH clauses, because a person does not write one clause as a
non-native speaker and the next as a native one. Mixing registers inside one
short message is not a weaker version of the signal, it is a different and
incoherent stimulus.

That asymmetry has one consequence worth stating plainly: SLOT ORDER can be
manipulated for stated signals and cannot be manipulated for form signals. So
every pair containing a stated signal is run in both slot orders, and the
FORM x FORM pair is run once. This is a property of what a form signal IS, not
an omission.

THE TYPO RULE
-------------
F04 never corrupts a signal-bearing word. "screen reader" and "seventy-four"
survive every typo rendering intact.

If typos could mangle the disclosure, the F04 x S01 cell would be asking
whether the model can still READ the disclosure, not whether the two signals
interact. That is a different experiment and a worse one. The rule is asserted
at build time, so it cannot rot.

THE FLOOR
---------
Three conditions carry no social signal at all but do carry distinctive detail:
ALT1, ALT2_ONLY and ALT_BOTH. They give an interaction contrast built from two
socially neutral facts.

If two neutral details interact as strongly as two disclosures, the instrument
is measuring detail count rather than social content, and nothing else in the
experiment can be believed. That check runs first in the analysis. No paper in
the compounding literature reports it.
"""

import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# The clause pool.
#
# Six clause slots-fillers, each written in four styles. Word counts are held
# between 9 and 10 so that no condition can be long or short by more than a
# couple of words once two clauses are joined.
#
#   std        standard English
#   esl        F01, non-native: article omission, tense simplification, calque
#   typo       F04, native speaker, keyboard slips, signal words untouched
#   esl_typo   both form signals at once
#
# esl_typo is only needed for the two neutral clauses, because the FORM x FORM
# pair carries no stated clause.

CLAUSES = {
    "n1": {
        "kind": "neutral",
        "std":      "I am working through this on my laptop this evening.",
        "esl":      "I am doing this work on my laptop in evening.",
        "typo":     "im workign throuhg this on my laptop this eveing.",
        "esl_typo": "i am doign this work on my laptop in eveing.",
    },
    "n2": {
        "kind": "neutral",
        "std":      "I would like to get it finished before tomorrow.",
        "esl":      "I want to finish it before the tomorrow comes.",
        "typo":     "i woudl liek to get it finsihed before tomorow.",
        "esl_typo": "i want to finsih it before the tomorow comes.",
    },
    "alt1": {
        "kind": "alt",
        "std":      "I cycle to work most days of the week.",
        "esl":      "I go to work by bicycle in most days.",
        "typo":     "i cycle to work most dyas of teh week.",
    },
    "alt2": {
        "kind": "alt",
        "std":      "I have been baking my own bread since March.",
        "esl":      "I am baking my own bread since the March.",
        "typo":     "i ahve been bakign my own bread since March.",
    },
    "S01": {
        "kind": "stated",
        "signal_words": ["screen reader"],
        "std":      "I use a screen reader for everything on my computer.",
        "esl":      "I am using screen reader for all things on computer.",
        "typo":     "i use a screen reader for everythign on my computr.",
    },
    "S08": {
        "kind": "stated",
        "signal_words": ["seventy-four"],
        "std":      "I am seventy-four and not very confident with computers.",
        "esl":      "I have seventy-four years and computers are not easy.",
        "typo":     "i am seventy-four and not very confidnet with computrs.",
    },
}

# --------------------------------------------------------------------------
# The seventeen conditions. (slot1, slot2, style).
#
# Interaction contrasts are slot-matched: the single-signal cell used in a
# contrast always carries the signal in the SAME slot as the both-signal cell
# it is subtracted from. That is why S01 and S08 each appear in both slots.

CONDITIONS = [
    # id                slot1   slot2   style       label
    ("NONE",            "n1",   "n2",   "std",      "no signal"),

    ("ALT1",            "alt1", "n2",   "std",      "one neutral detail"),
    ("ALT2_ONLY",       "n1",   "alt2", "std",      "the other neutral detail"),
    ("ALT_BOTH",        "alt1", "alt2", "std",      "two neutral details"),

    ("S01_s1",          "S01",  "n2",   "std",      "screen reader, slot 1"),
    ("S01_s2",          "n1",   "S01",  "std",      "screen reader, slot 2"),
    ("S08_s1",          "S08",  "n2",   "std",      "age 74, slot 1"),
    ("S08_s2",          "n1",   "S08",  "std",      "age 74, slot 2"),

    ("F01",             "n1",   "n2",   "esl",      "non-native form"),
    ("F04",             "n1",   "n2",   "typo",     "typos"),

    ("P1_F01F04",       "n1",   "n2",   "esl_typo", "non-native + typos"),

    ("P2_S01S08",       "S01",  "S08",  "std",      "screen reader + age"),
    ("P2_S08S01",       "S08",  "S01",  "std",      "age + screen reader"),

    ("P3_F01S01_s1",    "S01",  "n2",   "esl",      "non-native + screen reader, slot 1"),
    ("P3_F01S01_s2",    "n1",   "S01",  "esl",      "non-native + screen reader, slot 2"),

    ("P4_F04S08_s1",    "S08",  "n2",   "typo",     "typos + age, slot 1"),
    ("P4_F04S08_s2",    "n1",   "S08",  "typo",     "typos + age, slot 2"),
]

# The contrasts the experiment exists to estimate.
#   name, pair type, (both, singleA, singleB, none)
# interaction per item = (both - singleA) - (singleB - none)
CONTRASTS = [
    ("FLOOR_ALTxALT",   "floor",
     ("ALT_BOTH", "ALT1", "ALT2_ONLY", "NONE")),

    ("P1_FORMxFORM",    "form_x_form",
     ("P1_F01F04", "F01", "F04", "NONE")),

    ("P2a_STATEDxSTATED", "stated_x_stated",
     ("P2_S01S08", "S01_s1", "S08_s2", "NONE")),
    ("P2b_STATEDxSTATED", "stated_x_stated",
     ("P2_S08S01", "S08_s1", "S01_s2", "NONE")),

    ("P3a_CROSS",       "cross",
     ("P3_F01S01_s1", "F01", "S01_s1", "NONE")),
    ("P3b_CROSS",       "cross",
     ("P3_F01S01_s2", "F01", "S01_s2", "NONE")),

    ("P4a_CROSS",       "cross",
     ("P4_F04S08_s1", "F04", "S08_s1", "NONE")),
    ("P4b_CROSS",       "cross",
     ("P4_F04S08_s2", "F04", "S08_s2", "NONE")),
]

MAX_WORD_SPREAD = 3          # across the seventeen lead-ins


def words(text):
    return len(text.split())


def clause(name, style):
    """The text of one clause in one style, falling back std -> style."""
    c = CLAUSES[name]
    if style in c:
        return c[style]
    if style == "esl_typo":
        raise KeyError(f"{name} has no esl_typo rendering and one is needed")
    raise KeyError(f"{name} has no {style} rendering")


def build():
    rows = []
    for cid, s1, s2, style, label in CONDITIONS:
        lead_in = clause(s1, style) + " " + clause(s2, style)
        stated = [n for n in (s1, s2) if CLAUSES[n]["kind"] == "stated"]
        alts = [n for n in (s1, s2) if CLAUSES[n]["kind"] == "alt"]
        form = [] if style == "std" else (
            ["F01", "F04"] if style == "esl_typo" else
            (["F01"] if style == "esl" else ["F04"]))
        rows.append({
            "id": cid,
            "label": label,
            "lead_in": lead_in,
            "slot1": s1,
            "slot2": s2,
            "style": style,
            "stated_signals": stated,
            "form_signals": form,
            "alt_details": alts,
            "n_social_signals": len(stated) + len(form),
            "n_words": words(lead_in),
        })
    return rows


def check(rows):
    """Every assertion here was a bug waiting to happen."""
    problems = []
    by_id = {r["id"]: r for r in rows}

    if len(by_id) != len(rows):
        problems.append("duplicate condition id")

    lens = [r["n_words"] for r in rows]
    spread = max(lens) - min(lens)
    if spread > MAX_WORD_SPREAD:
        problems.append(
            f"lead-in length spread is {spread} words, limit {MAX_WORD_SPREAD}. "
            "Paper 2 measured a 40-word preamble costing Llama ten accuracy "
            "points on its own, so length is never allowed to vary freely")

    # every condition is exactly two clauses
    for r in rows:
        if r["slot1"] == r["slot2"]:
            problems.append(f"{r['id']} uses the same clause twice")

    # the typo rule: signal words survive every rendering
    for name, c in CLAUSES.items():
        for w in c.get("signal_words", []):
            for style in ("std", "esl", "typo", "esl_typo"):
                if style not in c:
                    continue
                if w.lower() not in c[style].lower():
                    problems.append(
                        f"{name}/{style} lost the signal phrase '{w}'. F04 must "
                        "never corrupt a signal-bearing word, or the cross cell "
                        "tests legibility of the disclosure instead of "
                        "interaction")

    # typo renderings must actually differ from standard, and esl too
    for name, c in CLAUSES.items():
        for style in ("esl", "typo", "esl_typo"):
            if style in c and c[style] == c["std"]:
                problems.append(f"{name}/{style} is identical to std")

    # esl_typo must differ from both of its parents
    for name, c in CLAUSES.items():
        if "esl_typo" in c:
            if c["esl_typo"] == c["esl"] or c["esl_typo"] == c["typo"]:
                problems.append(f"{name}/esl_typo equals one of its parents")

    # numbers are never mangled: any digit run in std survives in every style
    for name, c in CLAUSES.items():
        nums = re.findall(r"\d+", c["std"])
        for style in ("esl", "typo", "esl_typo"):
            if style not in c:
                continue
            for n in nums:
                if n not in c[style]:
                    problems.append(f"{name}/{style} lost the number {n}")

    # every contrast names conditions that exist, and is well formed
    for cname, ctype, cells in CONTRASTS:
        for cell in cells:
            if cell not in by_id:
                problems.append(f"contrast {cname} names unknown condition {cell}")
        if len(set(cells)) != 4:
            problems.append(f"contrast {cname} repeats a cell")

    # the both-cell of a contrast must carry exactly the union of the two
    # single cells' signals, or the contrast is not an interaction at all
    for cname, ctype, (both, a, b, none) in CONTRASTS:
        if any(x not in by_id for x in (both, a, b, none)):
            continue
        def sig(r):
            return set(by_id[r]["stated_signals"] + by_id[r]["form_signals"]
                       + by_id[r]["alt_details"])
        want = sig(a) | sig(b)
        got = sig(both)
        if got != want:
            problems.append(
                f"contrast {cname}: both-cell carries {sorted(got)} but its "
                f"singles carry {sorted(want)}")
        if sig(none):
            problems.append(f"contrast {cname}: the none-cell carries a signal")

    # slot matching: in a stated-bearing contrast, the single cell must put its
    # stated clause in the same slot the both-cell puts it in
    for cname, ctype, (both, a, b, none) in CONTRASTS:
        if any(x not in by_id for x in (both, a, b, none)):
            continue
        for single in (a, b):
            for st in by_id[single]["stated_signals"]:
                slot_single = ("slot1" if by_id[single]["slot1"] == st
                               else "slot2")
                slot_both = ("slot1" if by_id[both]["slot1"] == st
                             else "slot2" if by_id[both]["slot2"] == st else None)
                if slot_both is None:
                    problems.append(
                        f"contrast {cname}: {st} missing from both-cell {both}")
                elif slot_both != slot_single:
                    problems.append(
                        f"contrast {cname}: {st} sits in {slot_single} of "
                        f"{single} but {slot_both} of {both}; the position term "
                        "would not cancel")
    return problems


def main():
    rows = build()
    problems = check(rows)

    print("=" * 74)
    print("e1 CONDITIONS")
    print("=" * 74)
    for r in rows:
        sig = ",".join(r["form_signals"] + r["stated_signals"]) or "-"
        print(f"  {r['id']:18}{r['n_words']:3}w  {sig:14}{r['lead_in']}")

    lens = [r["n_words"] for r in rows]
    print(f"\n  {len(rows)} conditions, "
          f"{min(lens)} to {max(lens)} words, spread {max(lens) - min(lens)}")

    print("\n  contrasts, interaction = (both - A) - (B - none):")
    for cname, ctype, (both, a, b, none) in CONTRASTS:
        print(f"    {cname:22}{ctype:18}({both} - {a}) - ({b} - {none})")

    if problems:
        print("\n  PROBLEMS")
        for p in problems:
            print(f"    {p}")
        sys.exit(1)
    print("\n  all structural checks pass")

    out_json = os.path.join(HERE, "e1_conditions.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"conditions": rows,
                   "contrasts": [{"name": n, "type": t, "both": c[0],
                                  "single_a": c[1], "single_b": c[2],
                                  "none": c[3]} for n, t, c in CONTRASTS]},
                  f, indent=2, ensure_ascii=False)

    out_csv = os.path.join(HERE, "e1_conditions.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "style", "slot1", "slot2",
                    "n_social_signals", "n_words", "lead_in"])
        for r in rows:
            w.writerow([r["id"], r["label"], r["style"], r["slot1"], r["slot2"],
                        r["n_social_signals"], r["n_words"], r["lead_in"]])

    print(f"  wrote {out_json}")
    print(f"  wrote {out_csv}")


if __name__ == "__main__":
    main()
