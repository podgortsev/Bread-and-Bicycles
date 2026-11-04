#!/usr/bin/env python3
"""
validate_tasks.py, an independent check of the shipped tasks.json.

No GPU. `python validate_tasks.py` reads tasks.json and re-derives the answer
from the QUESTION TEXT, not from the generator's variables.

WHY THIS EXISTS
---------------
build_tasks.py has a verify() function, and for a while its docstring described
that function as "an independent solver". It is not one: it checks that a
numeric answer parses as a float, that the question ends in punctuation, and
that it is not absurdly short. Those are format checks. A generator that
computed the wrong key would pass all three.

This file is the independent solver the docstring claimed. It re-parses each
question out of the JSON and recomputes the answer with arithmetic written
separately from the generator, so a mistake would have to be made twice, in the
same direction, to survive.

It also checks the properties the generator does not guarantee:

  * every question is unique
  * no table task ties at the extreme it asks about
  * the answer key is reachable by exactly one reading of the question

WHAT IT FINDS ON THE SHIPPED DATASET
------------------------------------
Two things, both documented in the README and neither of which changes a
published number:

1. 200 rows contain 191 distinct questions. Nine rows are repeats of a question
   that already appears. They carry consistent answers, so no key is wrong, but
   "200 tasks" means 200 rows over 191 distinct items.

2. The date family is ambiguous between an inclusive and an exclusive reading
   of "runs for N days". The key uses the exclusive one. This shifts baseline
   accuracy and cancels in the paired design; see the note in the README.

The dataset is NOT regenerated to fix either. Every accuracy number in methods
3a, 3b and 3c was measured on these exact 200 rows.
"""

import collections
import json
import os
import re
import sys
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def close(a, b, tol=0.01):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# independent re-derivation, one family at a time, from the question text
# ---------------------------------------------------------------------------

def solve_percentage(q):
    m = re.match(r"What is ([\d.]+) percent of ([\d.]+)\?$", q)
    if m:
        return float(m.group(1)) / 100.0 * float(m.group(2))
    m = re.match(r"A price of ([\d.]+) dollars (increases|decreases) by "
                 r"([\d.]+) percent\. What is the new price\?$", q)
    if m:
        base, pct = float(m.group(1)), float(m.group(3))
        step = base * pct / 100.0
        return base + step if m.group(2) == "increases" else base - step
    m = re.match(r"([\d.]+) is what percent of ([\d.]+)\?$", q)
    if m:
        return float(m.group(1)) / float(m.group(2)) * 100.0
    return None


def solve_conversion(q):
    m = re.match(r"How many minutes are there in ([\d.]+) hours\?$", q)
    if m:
        return float(m.group(1)) * 60
    m = re.match(r"How many seconds are there in ([\d.]+) minutes\?$", q)
    if m:
        return float(m.group(1)) * 60
    m = re.match(r"How many grams are there in ([\d.]+) kilograms\?$", q)
    if m:
        return float(m.group(1)) * 1000
    m = re.match(r"How many centimetres are there in ([\d.]+) metres\?$", q)
    if m:
        return float(m.group(1)) * 100
    m = re.match(r"How many millilitres are there in ([\d.]+) litres\?$", q)
    if m:
        return float(m.group(1)) * 1000
    m = re.match(r"How many hours are there in ([\d.]+) days\?$", q)
    if m:
        return float(m.group(1)) * 24
    m = re.match(r"How many days are there in ([\d.]+) weeks\?$", q)
    if m:
        return float(m.group(1)) * 7
    m = re.match(r"How many kilograms are there in ([\d.]+) tonnes\?$", q)
    if m:
        return float(m.group(1)) * 1000
    m = re.match(r"How many metres are there in ([\d.]+) kilometres\?$", q)
    if m:
        return float(m.group(1)) * 1000
    return None


def solve_word_problem(q):
    m = re.match(r"A team of (\d+) people works (\d+) hours a day for (\d+) "
                 r"days\. Each person is paid (\d+) dollars an hour\. What is "
                 r"the total wage bill\?$", q)
    if m:
        n, h, d, pay = (int(g) for g in m.groups())
        return n * h * d * pay
    return None


def solve_rate(q):
    m = re.match(r"A vehicle travels ([\d.]+) kilometres at a steady ([\d.]+) "
                 r"kilometres per hour\. How many hours does the journey take\?",
                 q)
    if m:
        return float(m.group(1)) / float(m.group(2))
    m = re.match(r"A vehicle covers ([\d.]+) kilometres in ([\d.]+) hours\. "
                 r"What is its average speed", q)
    if m:
        return float(m.group(1)) / float(m.group(2))
    m = re.match(r"A vehicle travels for ([\d.]+) hours at a steady ([\d.]+) "
                 r"kilometres per hour\. How far does it travel\?", q)
    if m:
        return float(m.group(1)) * float(m.group(2))
    return None


def solve_date(q):
    m = re.match(r"A project starts on (\d{2}) (\w+) (\d{4}) and runs for "
                 r"(\d+) days\.", q)
    if not m:
        return None
    d, mon, y, n = m.groups()
    start = date(int(y), MONTHS.index(mon) + 1, int(d))
    # The exclusive reading: the key is start + N days. The inclusive reading
    # would be start + (N - 1). Both are returned so the ambiguity is visible.
    return ((start + timedelta(days=int(n))).strftime("%d %B %Y"),
            (start + timedelta(days=int(n) - 1)).strftime("%d %B %Y"))


def table_values(q):
    return {m.group(1): int(m.group(2)) for m in
            re.finditer(r"(North|South|East|West|Central): (\d+)", q)}


def solve_table(q):
    v = table_values(q)
    if not v:
        return None
    if "sold the most" in q:
        return max(v, key=v.get)
    if "sold the least" in q:
        return min(v, key=v.get)
    if "total across all regions" in q:
        return sum(v.values())
    if "difference between" in q:
        return max(v.values()) - min(v.values())
    return None


def solve_logic(q):
    """Rebuild the order from the pairwise facts by topological sort.

    "X is older than Y" is an edge X -> Y. A unique ordering exists only if at
    every step exactly one person has nobody left who is stated to be older
    than them. If two ever qualify, the question has more than one consistent
    answer and is reported as ambiguous rather than silently graded.
    """
    facts = re.findall(r"(\w+) is older than (\w+)", q)
    if not facts:
        return None
    people = {p for f in facts for p in f}
    older_than = {p: set() for p in people}     # p -> those p outranks
    indeg = collections.Counter({p: 0 for p in people})
    for a, b in facts:
        if b not in older_than[a]:
            older_than[a].add(b)
            indeg[b] += 1

    order, remaining = [], dict(indeg)
    while remaining:
        ready = [p for p, d in remaining.items() if d == 0]
        if len(ready) != 1:
            return "AMBIGUOUS"          # 0 = contradiction, >1 = several orders
        p = ready[0]
        order.append(p)
        del remaining[p]
        for c in older_than[p]:
            if c in remaining:
                remaining[c] -= 1

    # "second oldest" contains "oldest", so it must be tested first.
    if "second oldest" in q:
        return order[1]
    if "oldest" in q:
        return order[0]
    if "youngest" in q:
        return order[-1]
    return None


SOLVERS = {"percentage": solve_percentage, "conversion": solve_conversion,
           "rate": solve_rate, "table": solve_table, "logic": solve_logic,
           "word_problem": solve_word_problem}


def main():
    path = os.path.join(HERE, "tasks.json")
    if not os.path.exists(path):
        print(f"tasks.json not found next to this script ({HERE}).")
        return 1
    tasks = json.load(open(path, encoding="utf-8"))

    print("=" * 74)
    print("INDEPENDENT CHECK OF tasks.json")
    print("=" * 74)
    print(f"{len(tasks)} rows, families: "
          f"{dict(collections.Counter(t['family'] for t in tasks))}\n")

    fails, checked, skipped = [], 0, collections.Counter()

    # --- the keys ---------------------------------------------------------
    for t in tasks:
        fam, q, key = t["family"], t["question"], t["answer"]
        if fam == "date":
            got = solve_date(q)
            if got is None:
                skipped[fam] += 1
                continue
            checked += 1
            if key != got[0]:
                fails.append((t["id"], fam, key, got[0]))
            continue
        fn = SOLVERS.get(fam)
        if fn is None:
            skipped[fam] += 1
            continue
        got = fn(q)
        if got is None or got == "AMBIGUOUS":
            skipped[fam] += 1
            if got == "AMBIGUOUS":
                fails.append((t["id"], fam, key, "chain is not total"))
            continue
        checked += 1
        ok = close(key, got) if isinstance(got, (int, float)) else str(key) == str(got)
        if not ok:
            fails.append((t["id"], fam, key, got))

    print(f"keys re-derived from the question text: {checked}")
    print(f"not machine-readable, skipped: {sum(skipped.values())} "
          f"{dict(skipped)}")
    if fails:
        print(f"\n  KEY MISMATCHES: {len(fails)}")
        for i, fam, key, got in fails[:20]:
            print(f"    {i} {fam}: key={key!r} independent={got!r}")
    else:
        print("  every re-derived key matches. No wrong answers.")

    # --- uniqueness -------------------------------------------------------
    c = collections.Counter(t["question"] for t in tasks)
    dups = {q: n for q, n in c.items() if n > 1}
    print(f"\ndistinct questions: {len(c)} of {len(tasks)} rows")
    if dups:
        extra = sum(n - 1 for n in dups.values())
        print(f"  {len(dups)} questions repeat, {extra} extra rows")
        byfam = collections.Counter()
        conflict = 0
        for q, n in dups.items():
            rows = [t for t in tasks if t["question"] == q]
            byfam[rows[0]["family"]] += n - 1
            if len({r["answer"] for r in rows}) > 1:
                conflict += 1
                print(f"    CONFLICTING KEYS: {q[:70]}")
        print(f"  extra rows by family: {dict(byfam)}")
        print(f"  repeats with conflicting answers: {conflict}"
              + ("  (none, so no key is wrong)" if not conflict else ""))

    # --- table ties -------------------------------------------------------
    ties = 0
    for t in tasks:
        if t["family"] != "table":
            continue
        v = table_values(t["question"])
        if not v:
            continue
        vals = list(v.values())
        if ("sold the most" in t["question"] and vals.count(max(vals)) > 1) or \
           ("sold the least" in t["question"] and vals.count(min(vals)) > 1):
            ties += 1
            print(f"    AMBIGUOUS TABLE TASK: {t['id']}")
    print(f"\ntable tasks tied at the extreme they ask about: {ties}")
    if not ties:
        print("  none. The generator does not exclude a tie, so this is luck")
        print("  rather than design; it is asserted here so a regenerated")
        print("  dataset cannot ship one silently.")

    # --- the date convention ----------------------------------------------
    print("\ndate family convention")
    print("  the key uses the EXCLUSIVE reading: start + N days.")
    print("  the inclusive reading, start + (N-1), is also defensible English.")
    print("  A model using it is marked wrong on all 25 date tasks, in EVERY")
    print("  condition alike, so it moves baseline accuracy and cancels in the")
    print("  paired within-task comparison every method 3 result rests on.")

    print("\n" + "=" * 74)
    hard = [f for f in fails]
    print("RESULT: PASS" if not hard and not ties else
          f"RESULT: FAIL ({len(hard)} key problems, {ties} ambiguous tables)")
    print("=" * 74)
    return 1 if (hard or ties) else 0


if __name__ == "__main__":
    sys.exit(main())
