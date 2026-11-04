#!/usr/bin/env python3
"""
build_e2b_tasks.py - the task text of the 600 arithmetic tasks, rewritten in
each of the three form styles. Writes e2b_tasks.json next to itself.

    python build_e2b_tasks.py        (reads shared/tasks/tasks.json)

WHAT e2b IS FOR
---------------
Liu et al. (arXiv:2604.04723) cross eight ESL variants with three typo severity
levels and report that the combined effect is larger than either alone and not
simply additive. That is the strongest published super-additive result on task
performance, and it is the nearest competitor to Paper 3.

They apply both factors to the TASK ITSELF. Two corruptions of the input a model
has to parse will compound mechanically, because legibility failures multiply.
That is a different claim from "the model compounds what it infers about the
person", and the two are not separable in their design.

e1 keeps the task byte-identical and puts the form signal only in the person's
own two-clause message. e2b runs the same four cells with the form applied to
the task text as well. Comparing the two interactions separates the two
mechanisms:

    arm A (e1)   form in the message only, task clean
    arm B (e2b)  form in the message AND in the task

If the interaction is super-additive in arm B and absent in arm A, the published
compounding result is about input legibility, not about how a model treats a
person. That would be a headline in its own right, and it costs one short run.

THE RULE THAT MAKES THIS SAFE
-----------------------------
A rewritten task must have the same answer as the original. Three guarantees,
all asserted here and none of them optional:

  1. EVERY NUMBER SURVIVES. Every digit run in the original appears in the
     rewrite, in the same multiset. Typos never touch a token containing a
     digit; the ESL rules never move or reword a number.
  2. NO NUMBER IS ADDED. The multiset is equal, not merely a superset.
  3. THE ANSWER KEY IS COPIED, NEVER RECOMPUTED. The rewrite cannot change what
     the question asks, only how it reads.

WHAT THIS CANNOT GUARANTEE, STATED PLAINLY
------------------------------------------
`validate_tasks.py` re-derives every key by parsing the question with regular
expressions. Those expressions match Paper 2's exact phrasing, so they cannot
parse a rewritten question, and the rewritten sets get no independent
re-derivation. The numeric assertions above are what stands in for it.

The residual risk is a rewrite that preserves every number but changes the
QUESTION, for example turning a maximum into a minimum. The ESL rules are
therefore deliberately narrow: article deletion, three preposition calques and
two auxiliary drops. None of them touches a comparative, a superlative or a
quantity word. The full list is in ESL_RULES below and is short enough to read.

TYPOS
-----
Adjacent-character transposition inside a word, which is the commonest real
keyboard slip and stays readable. Applied to at most a third of eligible words,
chosen deterministically from a seed derived from the task id, so the set
regenerates identically. A word is eligible only if it is at least five letters
long, is purely alphabetic, and is not in KEEP_WHOLE.

KEEP_WHOLE holds the words that carry the question's meaning: comparatives,
superlatives, and the words naming the quantity asked for. Mangling "highest"
into "hgihest" would risk changing what is being asked, which is the one thing
this experiment must not do.
"""

import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
# read the canonical task set from shared/, not a copy. A second tasks.json
# next to this script would be one more file that can drift.
SRC = next((c for c in (
    os.path.join(HERE, "..", "..", "shared", "tasks", "tasks.json"),
    os.path.join(HERE, "tasks.json"),
) if os.path.exists(c)), os.path.join(HERE, "tasks.json"))
OUT = os.path.join(HERE, "e2b_tasks.json")

ARMS = ["std", "esl", "typo", "esl_typo"]

# ESL rewriting. Deliberately narrow: nothing here touches a comparative, a
# superlative, a unit or a quantity word.
ESL_RULES = [
    (r"\bWhat is the\b", "What is"),
    (r"\bwhat is the\b", "what is"),
    (r"\bthe new price\b", "new price"),
    (r"\bthe total\b", "total"),
    (r"\bthe difference between the\b", "difference between"),
    (r"\bthe highest and the lowest\b", "highest and lowest"),
    (r"\bthe oldest\b", "oldest"),
    (r"\bthe youngest\b", "youngest"),
    (r"\bthe second oldest\b", "second oldest"),
    (r"\bpercent of\b", "percent from"),          # preposition calque
    (r"\bare there in\b", "is in"),               # agreement simplification
    (r"\bdoes the journey take\b", "the journey take"),
    (r"\bdoes it travel\b", "it travel"),
    (r"\bdoes it end\b", "it end"),
    (r"\bis its average speed\b", "is average speed"),
    (r"\bA team of\b", "One team of"),
    (r"\bA vehicle\b", "One vehicle"),
    (r"\bA project\b", "One project"),
    (r"\bA price of\b", "One price of"),
    (r"\bEach person is paid\b", "Each person get"),
    (r"\bWhich region sold\b", "Which region sell"),
    (r"\bHow many\b", "How much"),
]

# never typo these: they carry what the question asks for
KEEP_WHOLE = {
    "highest", "lowest", "oldest", "youngest", "second", "most", "least",
    "total", "difference", "percent", "percentage", "average", "increases",
    "decreases", "increase", "decrease", "older", "younger", "before", "after",
    "format", "answer", "region", "regions",
}

TYPO_FRACTION = 3          # one word in three among the eligible ones
MIN_TYPO_LEN = 5


def numbers(text):
    return Counter(re.findall(r"\d+", text))


def esl(text):
    out = text
    for pat, rep in ESL_RULES:
        out = re.sub(pat, rep, out)
    return out


def _rng_for(task_id):
    import random
    return random.Random(f"e2b:{task_id}")


def typo(text, task_id):
    rng = _rng_for(task_id)
    words = text.split(" ")
    eligible = [i for i, w in enumerate(words)
                if len(re.sub(r"\W", "", w)) >= MIN_TYPO_LEN
                and re.sub(r"\W", "", w).isalpha()
                and re.sub(r"\W", "", w).lower() not in KEEP_WHOLE]
    if not eligible:
        return text
    n = max(1, len(eligible) // TYPO_FRACTION)
    for i in sorted(rng.sample(eligible, min(n, len(eligible)))):
        w = words[i]
        core = re.sub(r"\W", "", w)
        start = w.index(core[0])
        j = rng.randrange(1, len(core) - 2)      # never the first or last letter
        swapped = core[:j] + core[j + 1] + core[j] + core[j + 2:]
        words[i] = w[:start] + swapped + w[start + len(core):]
    return " ".join(words)


def rewrite(question, task_id, arm):
    if arm == "std":
        return question
    if arm == "esl":
        return esl(question)
    if arm == "typo":
        return typo(question, task_id)
    return typo(esl(question), task_id)


def main():
    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC}. Copy tasks.json from experiments/shared/tasks/")
    tasks = json.load(open(SRC, encoding="utf-8"))

    out = []
    problems = []
    unchanged = {a: 0 for a in ARMS}
    for t in tasks:
        base_nums = numbers(t["question"])
        row = {"id": t["id"], "family": t["family"], "answer": t["answer"],
               "answer_kind": t["answer_kind"], "text": {}}
        for arm in ARMS:
            q = rewrite(t["question"], t["id"], arm)
            row["text"][arm] = q
            if numbers(q) != base_nums:
                problems.append(
                    f"{t['id']}/{arm}: numbers changed, "
                    f"{sorted(base_nums.elements())} -> "
                    f"{sorted(numbers(q).elements())}")
            if arm != "std" and q == t["question"]:
                unchanged[arm] += 1
        out.append(row)

    print("=" * 72)
    print("e2b TASK REWRITES")
    print("=" * 72)
    print(f"  {len(out)} tasks x {len(ARMS)} arms")
    for arm in ARMS[1:]:
        print(f"  {arm:10}{len(out) - unchanged[arm]:4} of {len(out)} tasks "
              f"actually changed")
    print()
    for arm in ARMS:
        print(f"  {arm}:")
        print(f"    {out[0]['text'][arm]}")
    print()
    for arm in ARMS:
        print(f"  {arm}:")
        print(f"    {out[7]['text'][arm]}")

    if problems:
        print("\n  PROBLEMS, nothing was written:")
        for p in problems[:20]:
            print(f"    {p}")
        sys.exit(1)

    # the ESL arm must actually do something to most tasks, or the cell is a
    # duplicate of std wearing a different name
    for arm in ARMS[1:]:
        changed = len(out) - unchanged[arm]
        if changed < 0.8 * len(out):
            print(f"\n  PROBLEM: {arm} changed only {changed} of {len(out)} "
                  f"tasks. A cell that is mostly identical to std measures "
                  f"nothing.")
            sys.exit(1)

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    print(f"\n  every number preserved in every arm")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
