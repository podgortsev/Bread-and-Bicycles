#!/usr/bin/env python3
"""
build_e2_conditions.py - the seventeen conditions of experiment e2, the depth
curve. Writes e2_conditions.json and e2_conditions.csv next to itself.

THE QUESTION e2 ASKS, AND HOW IT DIFFERS FROM e1
------------------------------------------------
e1 asks whether a PAIR interacts. e2 asks whether the penalty grows with the
NUMBER of signals. The literature merges the two and Paper 3 does not.

This is Paper 2's method 3b rebuilt to a standard it can be believed at. What
changes, all of it from the m3b result document's own list of defects:

  * MISTRAL IS INCLUDED. m3b ran on two models.
  * SIGNALS ARE NO LONGER SELECTED ON THE OUTCOME. m3b picked six signals
    because they had shown an effect in the single-signal run, which makes it a
    follow-up rather than a test. Here the three form signals and the three
    stated signals are fixed in advance by TYPE, not by result.
  * TWO LADDERS, NOT ONE. A form-only ladder and a stated-only ladder. If the
    thesis holds they have different shapes, and that comparison is the point.
    m3b mixed both types into single routes, so depth and type were confounded.
  * CLAUSE COUNT IS CONSTANT AT THREE. m3b padded to a target word count, which
    controls length but not the number of things said. Here every lead-in is
    exactly three clauses at every depth, and a signal takes a slot a neutral
    clause would have held.
  * THE SHAPE IS NOT ASSIGNED BY A HEURISTIC. m3b labelled curves COMPOUNDING or
    MIXED using arbitrary thresholds and said so. The analyser fits linear
    against quadratic and reports the second difference with an interval.

THE TWO LADDERS

    FORM      F01 non-native, F04 typos, F07 informal and slang
    STATED    S01 screen reader, S08 age 74, S04 ADHD

Both drawn from `shared/axes/`, three each, chosen to be squarely inside their
type rather than for their measured effect.

WHY THE LADDERS ARE SHAPED DIFFERENTLY, AND WHY THAT IS NOT A FLAW
------------------------------------------------------------------
A stated signal is a clause, so at depth 2 the two signals have an ORDER, and
the ladder carries three rotations so no single ordering drives the curve.

A form signal is a way of writing, so two form signals at depth 2 are two styles
applied to the same three clauses and there is no order to vary. The form ladder
therefore has one condition per combination and the stated ladder has three
rotations per depth.

This is a property of what a form signal IS, stated here rather than hidden. The
consequence for the analysis is that the form curve has fewer independent routes
than the stated curve, so its depth means carry wider intervals, and the
analyser prints the count behind every point.

HOW THE STYLE COMBINATIONS ARE MADE
-----------------------------------
Four base renderings of each neutral clause are written by hand: standard,
non-native, slang, and non-native-plus-slang. The typo layer is applied
programmatically on top of any of them, because a keyboard slip is a mechanical
transformation and hand-writing eight variants of three clauses invites exactly
the kind of quiet inconsistency this experiment cannot afford.

The typo transform never touches a word shorter than five letters and never the
first or last letter of a word, so every clause stays readable, and there are no
signal-bearing words in the form ladder to protect because it carries none.
"""

import csv
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

MIN_TYPO_LEN = 5
TYPO_FRACTION = 3

# Three neutral clauses, four hand-written base styles each.
NEUTRAL = {
    "n1": {
        "std":   "I am working through this on my laptop this evening.",
        "esl":   "I am doing this work on my laptop in evening.",
        "slang": "im just messing about with this on my laptop tonight.",
        "esl_slang": "im just doing this stuff on my laptop in evening.",
    },
    "n2": {
        "std":   "I would like to get it finished before tomorrow.",
        "esl":   "I want to finish it before the tomorrow comes.",
        "slang": "kinda wanna get it sorted before tomorrow tbh.",
        "esl_slang": "i wanna finish it before the tomorrow comes.",
    },
    "n3": {
        "std":   "I have been putting this off since the weekend.",
        "esl":   "I am putting this off from the weekend already.",
        "slang": "been putting this off since the weekend lol.",
        "esl_slang": "im putting this off from the weekend already lol.",
    },
}

# Three stated signals, standard English, matched in length to the neutrals.
STATED = {
    "S01": {"std": "I use a screen reader for everything on my computer.",
            "signal_words": ["screen reader"]},
    "S08": {"std": "I am seventy-four and not very confident with computers.",
            "signal_words": ["seventy-four"]},
    "S04": {"std": "I have ADHD and lose focus with a long answer.",
            "signal_words": ["ADHD"]},
}

FORM_SIGNALS = ["F01", "F04", "F07"]      # esl, typo, slang
STATED_SIGNALS = ["S01", "S08", "S04"]

# which hand-written base a set of form signals maps to, before typos
BASE_FOR = {
    frozenset(): "std",
    frozenset(["F01"]): "esl",
    frozenset(["F07"]): "slang",
    frozenset(["F01", "F07"]): "esl_slang",
    frozenset(["F04"]): "std",
    frozenset(["F01", "F04"]): "esl",
    frozenset(["F04", "F07"]): "slang",
    frozenset(["F01", "F04", "F07"]): "esl_slang",
}


def apply_typos(text, seed):
    rng = random.Random(seed)
    words = text.split(" ")
    eligible = [i for i, w in enumerate(words)
                if len(re.sub(r"\W", "", w)) >= MIN_TYPO_LEN
                and re.sub(r"\W", "", w).isalpha()]
    if not eligible:
        return text
    n = max(1, len(eligible) // TYPO_FRACTION)
    for i in sorted(rng.sample(eligible, min(n, len(eligible)))):
        w = words[i]
        core = re.sub(r"\W", "", w)
        start = w.index(core[0])
        j = rng.randrange(1, len(core) - 2)
        swapped = core[:j] + core[j + 1] + core[j] + core[j + 2:]
        words[i] = w[:start] + swapped + w[start + len(core):]
    return " ".join(words)


def neutral_clause(name, form_set):
    base = NEUTRAL[name][BASE_FOR[frozenset(form_set)]]
    if "F04" in form_set:
        return apply_typos(base, f"e2:{name}:{sorted(form_set)}")
    return base


def build():
    rows = []

    def add(cid, label, ladder, depth, slots, form_set, stated_list, route):
        text = " ".join(slots)
        rows.append({
            "id": cid, "label": label, "ladder": ladder, "depth": depth,
            "route": route,
            "form_signals": sorted(form_set),
            "stated_signals": list(stated_list),
            "n_social_signals": len(form_set) + len(stated_list),
            "n_clauses": len(slots),
            "n_words": len(text.split()),
            "lead_in": text,
        })

    # depth 0, shared by both ladders
    d0 = [neutral_clause(n, set()) for n in ("n1", "n2", "n3")]
    add("D0", "no signal", "both", 0, d0, set(), [], "-")

    # ---- FORM ladder. No order to vary: a style has no position. ----------
    form_sets = [
        (1, [{"F01"}, {"F04"}, {"F07"}]),
        (2, [{"F01", "F04"}, {"F01", "F07"}, {"F04", "F07"}]),
        (3, [{"F01", "F04", "F07"}]),
    ]
    for depth, sets in form_sets:
        for fs in sets:
            slots = [neutral_clause(n, fs) for n in ("n1", "n2", "n3")]
            tag = "".join(sorted(s[1:] for s in fs))
            add(f"FORM_D{depth}_{tag}", f"form depth {depth}: "
                f"{'+'.join(sorted(fs))}", "form", depth, slots, fs, [],
                "-")

    # ---- STATED ladder. Three rotations, so no single ordering drives it. --
    rotations = [
        ["S01", "S08", "S04"],
        ["S08", "S04", "S01"],
        ["S04", "S01", "S08"],
    ]
    for r_i, order in enumerate(rotations, start=1):
        for depth in (1, 2, 3):
            chosen = order[:depth]
            slots = []
            neutral_names = ["n1", "n2", "n3"]
            for k in range(3):
                if k < depth:
                    slots.append(STATED[chosen[k]]["std"])
                else:
                    slots.append(neutral_clause(neutral_names[k], set()))
            add(f"STATED_D{depth}_R{r_i}",
                f"stated depth {depth}, route {r_i}: {'+'.join(chosen)}",
                "stated", depth, slots, set(), chosen, f"R{r_i}")

    return rows


def check(rows):
    problems = []
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        problems.append("duplicate condition id")

    for r in rows:
        if r["n_clauses"] != 3:
            problems.append(f"{r['id']} has {r['n_clauses']} clauses, not 3")

    lens = [r["n_words"] for r in rows]
    spread = max(lens) - min(lens)
    if spread > 6:
        problems.append(f"lead-in word spread {spread}, limit 6. Three clauses "
                        f"give more room to drift than two; a 40-word preamble "
                        f"cost Llama ten accuracy points in Paper 2")

    # depth must equal the number of signals, in both ladders
    for r in rows:
        if r["n_social_signals"] != r["depth"]:
            problems.append(f"{r['id']}: depth {r['depth']} but "
                            f"{r['n_social_signals']} signals")

    # a stated signal's words must survive verbatim
    for r in rows:
        for s in r["stated_signals"]:
            for w in STATED[s]["signal_words"]:
                if w not in r["lead_in"]:
                    problems.append(f"{r['id']} lost the signal phrase '{w}'")

    # each ladder must reach depth 3
    for ladder in ("form", "stated"):
        depths = {r["depth"] for r in rows if r["ladder"] == ladder}
        if depths != {1, 2, 3}:
            problems.append(f"{ladder} ladder covers depths {sorted(depths)}")

    # the form ladder must actually differ between styles
    form_texts = {r["id"]: r["lead_in"] for r in rows if r["ladder"] == "form"}
    if len(set(form_texts.values())) != len(form_texts):
        problems.append("two form conditions produced identical text")

    # depth 0 must carry nothing
    d0 = next(r for r in rows if r["id"] == "D0")
    if d0["n_social_signals"]:
        problems.append("D0 carries a signal")
    return problems


def main():
    rows = build()
    problems = check(rows)

    print("=" * 78)
    print("e2 CONDITIONS: two depth ladders, three clauses at every depth")
    print("=" * 78)
    for ladder in ("both", "form", "stated"):
        sel = [r for r in rows if r["ladder"] == ladder]
        if not sel:
            continue
        print(f"\n  {ladder.upper()}")
        for r in sel:
            print(f"    {r['id']:18}d{r['depth']} {r['n_words']:3}w  "
                  f"{r['lead_in']}")

    lens = [r["n_words"] for r in rows]
    print(f"\n  {len(rows)} conditions, {min(lens)} to {max(lens)} words, "
          f"spread {max(lens) - min(lens)}")
    for ladder in ("form", "stated"):
        for d in (1, 2, 3):
            n = sum(1 for r in rows
                    if r["ladder"] == ladder and r["depth"] == d)
            print(f"    {ladder:8}depth {d}: {n} condition(s)")

    if problems:
        print("\n  PROBLEMS")
        for p in problems:
            print(f"    {p}")
        sys.exit(1)
    print("\n  all structural checks pass")

    json.dump({"conditions": rows}, open(
        os.path.join(HERE, "e2_conditions.json"), "w", encoding="utf-8"),
        indent=2, ensure_ascii=False)
    with open(os.path.join(HERE, "e2_conditions.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "ladder", "depth", "route",
                    "n_social_signals", "n_words", "lead_in"])
        for r in rows:
            w.writerow([r["id"], r["label"], r["ladder"], r["depth"],
                        r["route"], r["n_social_signals"], r["n_words"],
                        r["lead_in"]])
    print(f"  wrote e2_conditions.json and e2_conditions.csv in {HERE}")


if __name__ == "__main__":
    main()
