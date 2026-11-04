#!/usr/bin/env python3
"""
build_tasks_p3.py - extends the Paper 2 task set from 200 to 600.

    python build_tasks_p3.py            reads tasks_p2_200.json, writes tasks.json

WHY THIS EXISTS
---------------
Paper 3 asks about INTERACTIONS, not main effects. An interaction is a
difference of differences, so it carries about twice the variance of a single
difference at the same n.

The e1 validator measured what that costs. On the binary accuracy channel with
the 200 Paper 2 tasks, the power to detect a form-by-form interaction of ten
accuracy points is 8 percent, and of fifteen points, 22 percent. At that power
a null means nothing at all. Measured, not assumed:

    n tasks    -0.05   -0.10   -0.15   -0.20    MDE
      200         2%      8%     22%     58%   0.119
      400         3%     20%     50%     88%   0.097
      600         5%     45%     78%     93%   0.075
      800         5%     47%     95%    100%   0.067

600 is the chosen point: 78 percent power at fifteen points and a minimum
detectable interaction of about seven and a half points, for 10,200 generations
per model, which is roughly two Colab sessions. 800 buys 95 percent at fifteen
points for a third more compute, and is the obvious upgrade if sessions turn
out cheaper than expected.

Nothing affordable reaches useful power at five points. That is a property of a
binary outcome, and it is why e1 also runs two continuous channels. The number
belongs in the paper next to every null on this channel.

THE FIRST 200 ROWS ARE NOT REGENERATED
--------------------------------------
Rows T001 to T200 are copied byte for byte from the Paper 2 set. Paper 2's
methods 3a, 3b and 3c were all measured on those exact rows, and changing the
random stream would silently invalidate every accuracy number in the published
paper. The Paper 2 build script says so explicitly and it is right.

So this script only ADDS. T201 to T600 are new, drawn from the same seven
families with a different seed.

THREE THINGS PAPER 2 COULD NOT FIX, FIXED HERE FOR THE NEW ROWS ONLY
--------------------------------------------------------------------
Paper 2's build script documents three known defects and deliberately does not
repair them, because repairing them would have meant regenerating. The new rows
are not bound by that, so:

  * UNIQUENESS IS ENFORCED. Paper 2's 200 rows hold 191 distinct questions, so
    nine items are weighted twice. Every new question is checked against the
    original 200 and against the other new ones.
  * TABLE TIES ARE EXCLUDED. The table family asks which region sold most or
    least; a tie would give two correct answers. New rows reject any draw with
    a tie at the extreme being asked about.
  * THE DATE FAMILY KEEPS THE EXCLUSIVE READING (start + N days), matching
    Paper 2 so the two halves of the set are scored the same way. This is the
    one known ambiguity that is deliberately preserved rather than fixed,
    because consistency across the 600 matters more than being right about a
    convention that cancels in every paired comparison.

Independent re-derivation is `validate_tasks.py`, next to this file, which
re-parses each question and recomputes the key with arithmetic written
separately from this generator. Run it after this.
"""

import csv
import json
import os
import random
import sys
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "tasks_p2_200.json")
OUT_JSON = os.path.join(HERE, "tasks.json")
OUT_CSV = os.path.join(HERE, "tasks.csv")

N_NEW = 400
SEED = 20260908

# how many of each family among the new rows, in the Paper 2 proportions
FAMILY_PLAN = [
    ("percentage", 70),
    ("word_problem", 60),
    ("conversion", 50),
    ("date", 50),
    ("table", 60),
    ("logic", 50),
    ("rate", 60),
]

CONV = [
    ("kilometres", "metres", 1000), ("hours", "minutes", 60),
    ("kilograms", "grams", 1000), ("litres", "millilitres", 1000),
    ("days", "hours", 24), ("weeks", "days", 7),
    ("minutes", "seconds", 60), ("tonnes", "kilograms", 1000),
]
REGIONS = ["North", "South", "East", "West", "Central"]
NAMES = ["Alex", "Bailey", "Casey", "Devon", "Ellis", "Frankie", "Gray"]

# widened draw pools, so 400 new rows can be distinct without straining
PCT_BASES = [110, 120, 135, 150, 180, 210, 240, 260, 320, 340, 380, 420,
             480, 520, 560, 640, 720, 780, 850, 940, 1100, 1250, 1420,
             1600, 1850, 2100, 2400, 2750, 3200, 3600]
PCT_RATES = [5, 8, 12, 15, 16, 18, 20, 22, 24, 25, 30, 35, 40, 45, 50,
             55, 60, 65, 70, 75, 80]
CONV_NS = [3, 5, 7, 9, 11, 12, 15, 18, 21, 24, 28, 30, 36, 42, 48, 54,
           60, 72, 84, 90, 108, 125, 140, 160]
DISTS = [60, 90, 120, 135, 150, 180, 210, 240, 270, 300, 330, 360, 420, 480]
SPEEDS = [25, 30, 36, 40, 45, 50, 60, 72, 75, 80, 90, 100]


def gen_percentage(rng):
    base = rng.choice(PCT_BASES)
    pct = rng.choice(PCT_RATES)
    mode = rng.choice(["of", "increase", "decrease"])
    if mode == "of":
        return (f"What is {pct} percent of {base}?",
                f"{base * pct / 100:g}", "number")
    if mode == "increase":
        return (f"A price of {base} dollars increases by {pct} percent. "
                f"What is the new price?",
                f"{base * (1 + pct / 100):g}", "number")
    return (f"A price of {base} dollars decreases by {pct} percent. "
            f"What is the new price?",
            f"{base * (1 - pct / 100):g}", "number")


def gen_word_problem(rng):
    workers = rng.randint(3, 12)
    hours = rng.randint(4, 12)
    rate = rng.choice([15, 18, 20, 22, 25, 28, 30, 35, 38, 42, 45])
    days = rng.randint(2, 9)
    return (f"A team of {workers} people works {hours} hours a day for "
            f"{days} days. Each person is paid {rate} dollars an hour. "
            f"What is the total wage bill?",
            str(workers * hours * rate * days), "number")


def gen_conversion(rng):
    a, b, f = rng.choice(CONV)
    n = rng.choice(CONV_NS)
    return (f"How many {b} are there in {n} {a}?", str(n * f), "number")


def gen_date(rng):
    start = date(rng.choice([2026, 2027]), rng.randint(1, 12),
                 rng.randint(1, 28))
    delta = rng.choice([9, 11, 14, 18, 21, 25, 30, 40, 45, 50, 60, 70,
                        75, 80, 90, 100])
    end = start + timedelta(days=delta)
    return (f"A project starts on {start.strftime('%d %B %Y')} and runs for "
            f"{delta} days. On what date does it end? Answer in the format "
            f"DD Month YYYY.",
            end.strftime("%d %B %Y"), "text")


def gen_table(rng):
    """Rejects any draw with a tie at the extreme the question asks about."""
    for _ in range(200):
        vals = {r: rng.randint(100, 900) for r in rng.sample(REGIONS, 4)}
        v = list(vals.values())
        mode = rng.choice(["max", "min", "sum", "diff"])
        if mode == "max" and v.count(max(v)) > 1:
            continue
        if mode == "min" and v.count(min(v)) > 1:
            continue
        rows = "; ".join(f"{k}: {x}" for k, x in vals.items())
        if mode == "max":
            return (f"Quarterly units sold. {rows}. Which region sold the "
                    f"most?", max(vals, key=vals.get), "text")
        if mode == "min":
            return (f"Quarterly units sold. {rows}. Which region sold the "
                    f"least?", min(vals, key=vals.get), "text")
        if mode == "sum":
            return (f"Quarterly units sold. {rows}. What is the total across "
                    f"all regions?", str(sum(v)), "number")
        return (f"Quarterly units sold. {rows}. What is the difference "
                f"between the highest and the lowest region?",
                str(max(v) - min(v)), "number")
    raise RuntimeError("table family could not find a tie-free draw")


def gen_logic(rng):
    people = rng.sample(NAMES, 4)
    facts = [f"{people[i]} is older than {people[i + 1]}" for i in range(3)]
    rng.shuffle(facts)
    mode = rng.choice(["oldest", "youngest", "second"])
    q = ". ".join(facts)
    if mode == "oldest":
        return (f"{q}. Who is the oldest?", people[0], "text")
    if mode == "youngest":
        return (f"{q}. Who is the youngest?", people[3], "text")
    return (f"{q}. Who is the second oldest?", people[1], "text")


def gen_rate(rng):
    speed = rng.choice(SPEEDS)
    mode = rng.choice(["time", "distance", "speed"])
    if mode == "time":
        for _ in range(50):
            dist = rng.choice(DISTS)
            if dist % speed == 0 or (dist / speed) == round(dist / speed, 2):
                break
        return (f"A vehicle travels {dist} kilometres at a steady {speed} "
                f"kilometres per hour. How many hours does the journey take?",
                f"{dist / speed:g}", "number")
    hrs = rng.choice([2, 3, 4, 5, 6, 7, 8])
    if mode == "distance":
        return (f"A vehicle travels for {hrs} hours at a steady {speed} "
                f"kilometres per hour. How far does it travel?",
                f"{hrs * speed:g}", "number")
    return (f"A vehicle covers {speed * hrs} kilometres in {hrs} hours. "
            f"What is its average speed in kilometres per hour?",
            f"{speed:g}", "number")


GENERATORS = {
    "percentage": gen_percentage,
    "word_problem": gen_word_problem,
    "conversion": gen_conversion,
    "date": gen_date,
    "table": gen_table,
    "logic": gen_logic,
    "rate": gen_rate,
}


def main():
    if not os.path.exists(SOURCE):
        sys.exit(f"missing {SOURCE}. Copy Paper 2's tasks.json here under that "
                 f"name so the original 200 rows can be carried through "
                 f"unchanged.")
    original = json.load(open(SOURCE, encoding="utf-8"))
    if len(original) != 200:
        sys.exit(f"expected 200 original tasks, found {len(original)}")

    rng = random.Random(SEED)
    seen = {t["question"].strip() for t in original}
    out = list(original)
    per_family = {}
    rejected_dup = 0

    for family, want in FAMILY_PLAN:
        made = 0
        guard = 0
        while made < want:
            guard += 1
            if guard > want * 400:
                sys.exit(f"family {family} could not produce {want} distinct "
                         f"questions; widen its draw pool")
            q, a, kind = GENERATORS[family](rng)
            if q.strip() in seen:
                rejected_dup += 1
                continue
            seen.add(q.strip())
            out.append({"id": f"T{len(out) + 1:03d}", "family": family,
                        "question": q, "answer": a, "answer_kind": kind})
            made += 1
        per_family[family] = made

    assert len(out) == 200 + N_NEW, f"built {len(out)} rows"

    # the original rows must be untouched
    for a, b in zip(original, out[:200]):
        if a != b:
            sys.exit(f"original row {a['id']} was modified. Abort.")

    # every question distinct across the whole set
    qs = [t["question"].strip() for t in out]
    if len(set(qs)) != len(qs):
        # the original 200 hold 9 known repeats; new rows must add none
        new_qs = [t["question"].strip() for t in out[200:]]
        if len(set(new_qs)) != len(new_qs):
            sys.exit("a new row duplicates another new row")

    print("=" * 70)
    print("PAPER 3 TASK SET")
    print("=" * 70)
    print(f"  {len(original)} carried over from Paper 2, byte identical")
    print(f"  {N_NEW} new, {rejected_dup} draws rejected as duplicates")
    print(f"  {len(out)} rows total, "
          f"{len(set(qs))} distinct questions "
          f"({len(qs) - len(set(qs))} repeats, all inherited from Paper 2)")
    print()
    for family, want in FAMILY_PLAN:
        n_old = sum(1 for t in original if t["family"] == family)
        print(f"  {family:14}{n_old:4} old + {per_family[family]:4} new "
              f"= {n_old + per_family[family]:4}")

    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "family", "question",
                                          "answer", "answer_kind"])
        w.writeheader()
        w.writerows(out)
    print(f"\n  wrote {OUT_JSON}")
    print(f"  wrote {OUT_CSV}")
    print("\n  NOW RUN: python validate_tasks.py")
    print("  It re-derives every key with arithmetic written separately from")
    print("  this generator. Nothing ships until it reports zero mismatches.")


if __name__ == "__main__":
    main()
