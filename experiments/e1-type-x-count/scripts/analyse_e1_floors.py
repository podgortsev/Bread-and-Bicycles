#!/usr/bin/env python3
"""
analyse_e1_floors.py - all three floors, on every channel, side by side.

    python analyse_e1_floors.py                 reads ../outputs/<model>/
    python analyse_e1_floors.py path/to/outputs

This is the table the paper's methodological claim stands on. For each model and
each channel it reports the interaction of three socially neutral pairs, each
stacked exactly the way two disclosures are:

    floor 1   cycling to work    x baking bread       from the main run
    floor 2   walking the dog    x learning guitar    from the floor extension
    floor 3   making tea         x history books      from the floor extension

A channel is CLEAN on a model only if none of its floors is significant after
family-wise correction within that model and channel, on EITHER of two tests:
signed-rank, and a sign-flip permutation test on the mean. One family covers
every floor overall and within every stratum: Holm per test, x2 across tests. The verdict
does not get to choose its preferred floor, or its preferred test. See p3_stats
for why there are two.

Each model and channel is its own family. An earlier draft ran one BH correction
over every floor on every channel, which let the badly contaminated channels
make the clean channel's floors easier to flag.

On the two task channels, A and A2, the floors are also tested WITHIN three
difficulty strata (tasks near the ceiling, between, far from it), Holm over
floors x strata. A channel quiet in aggregate but moving within a stratum is
CLEAN IN AGGREGATE ONLY: on qwen's A2, floor 1 was +0.51 far from the ceiling
and -0.03 near it, two movements that cancel in the average. The guard is
applied to both task channels, not only the one where it was found.

Floor 1 comes from the main run and is compared against the main run's NONE;
floors 2 and 3 come from the extension and use the extension's own NONE, so each
floor is computed inside a single run.

TWO REPEATABILITY CHECKS COME FOR FREE
--------------------------------------
NONE was collected in both runs. Collection is greedy and scoring deterministic,
so the two copies should agree:

    channel A   the share of tasks with the same correct or wrong outcome
    channel B   the largest per-profile difference in trait log probability

A large disagreement would mean results depend on the session they were
collected in, which would undermine every comparison across runs.
"""

import csv
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "shared")))

import p3_stats as S                          # noqa: E402
import analyse_e1_answer_logprob as A2A       # noqa: E402

MODELS = ("qwen", "llama", "mistral")
MASS_GATE = 0.5


def _conditions_path(name):
    for cand in (name, os.path.join(HERE, name),
                 os.path.join(HERE, "..", "..", "shared", "conditions", name)):
        if os.path.exists(cand):
            return cand
    raise SystemExit(f"{name} not found")


def floors_in(name):
    with open(_conditions_path(name), encoding="utf-8") as f:
        return [c for c in json.load(f)["contrasts"] if c["type"] == "floor"]


def csv_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def both_orders(d):
    return {k: sum(v) / len(v) for k, v in d.items() if len(v) >= 2}


def load_acc(path):
    cells = defaultdict(dict)
    for r in csv_rows(path):
        cells[r["condition"]][r["task_id"]] = int(r["correct"])
    return cells


def load_lp(path):
    cells = defaultdict(dict)
    for r in csv_rows(path):
        v = float(r["logprob_sum"])
        if v == v:
            cells[r["condition"]][r["task_id"]] = v
    return cells


def load_traits(path):
    raw = defaultdict(lambda: defaultdict(list))
    for r in csv_rows(path):
        if r["valence"] == "positive":
            raw[r["condition"]][r["profile"]].append(float(r["logprob"]))
    return {c: {p: sum(v) / len(v) for p, v in d.items()}
            for c, d in raw.items()}


def load_choice_base(path):
    raw = defaultdict(lambda: defaultdict(list))
    for r in csv_rows(path):
        if float(r["mass"]) >= MASS_GATE:
            raw[r["condition"]][f"{r['profile']}:{r['question']}"].append(
                float(r["margin_signal"]))
    return {c: both_orders(d) for c, d in raw.items()}


def load_pairs(path):
    raw = defaultdict(lambda: defaultdict(list))
    for r in csv_rows(path):
        if float(r["mass"]) >= MASS_GATE:
            raw[(r["contrast"], r["side"])][
                f"{r['profile']}:{r['question']}"].append(
                float(r["margin_left"]))
    return {k: both_orders(d) for k, d in raw.items()}


def interaction(cells, ct):
    need = [ct["both"], ct["single_a"], ct["single_b"], ct["none"]]
    if any(c not in cells for c in need):
        return []
    ids = sorted(set.intersection(*[set(cells[c]) for c in need]))
    return [(cells[ct["both"]][i] - cells[ct["single_a"]][i])
            - (cells[ct["single_b"]][i] - cells[ct["none"]][i]) for i in ids]


def pairs_interaction(pairs, base, ct, side):
    other = ct["single_b"] if side == "vs_a" else ct["single_a"]
    left = pairs.get((ct["name"], side), {})
    right = base.get(other, {})
    ids = sorted(set(left) & set(right))
    return [left[i] - right[i] for i in ids]


def agreement(a, b):
    ids = set(a) & set(b)
    if not ids:
        return float("nan")
    return sum(1 for i in ids if a[i] == b[i]) / len(ids)


def max_abs_diff(a, b):
    ids = set(a) & set(b)
    if not ids:
        return float("nan")
    return max(abs(a[i] - b[i]) for i in ids)


def main(outputs, csv_out=None):
    f1 = floors_in("e1_conditions.json")
    f23 = floors_in("e1_floor_conditions.json")
    rows, checks, missing = [], [], []
    strat = defaultdict(list)     # (model, channel) -> stratified floor rows

    def add(model, channel, name, side, d):
        if len(d) < 10:
            missing.append(f"{model} {channel} {name} {side}: {len(d)} items")
            return
        mean, lo, hi = S.boot_ci(d)
        p_rank, _ = S.wilcoxon(d)
        rows.append({"model": model, "channel": channel, "floor": name,
                     "side": side, "n": len(d), "mean": mean, "lo": lo,
                     "hi": hi, "p_rank": p_rank, "p_mean": S.sign_flip(d),
                     "mde": S.mde(d)})

    for m in MODELS:
        def path(stem):
            return os.path.join(outputs, m, f"{stem}_{m}.csv")

        def have(*stems):
            gone = [s for s in stems if not os.path.exists(path(s))]
            for s in gone:
                missing.append(f"{m}: no {s}_{m}.csv")
            return not gone

        if have("e1_accuracy", "e1_accuracy_floor"):
            old, new = load_acc(path("e1_accuracy")), load_acc(
                path("e1_accuracy_floor"))
            for ct in f1:
                add(m, "A accuracy", ct["name"], "-", interaction(old, ct))
            for ct in f23:
                add(m, "A accuracy", ct["name"], "-", interaction(new, ct))
            for cells, fl in ((old, f1), (new, f23)):
                strat[(m, "A accuracy")] += A2A.stratified_rows(
                    cells, fl, m, A2A.ACCURACY_CUTS)
            if "NONE" in old and "NONE" in new:
                checks.append((m, "A accuracy: NONE gives the same outcome in "
                               "both runs on",
                               f"{agreement(old['NONE'], new['NONE']):.1%} "
                               f"of tasks"))

        if have("e1_answer_logprob"):
            lp = load_lp(path("e1_answer_logprob"))
            for ct in f1 + f23:
                add(m, "A2 answer logprob", ct["name"], "-",
                    interaction(lp, ct))
            strat[(m, "A2 answer logprob")] += A2A.stratified_rows(
                lp, f1 + f23, m, A2A.LOGPROB_CUTS)

        if have("e1_traits", "e1_traits_floor"):
            old, new = load_traits(path("e1_traits")), load_traits(
                path("e1_traits_floor"))
            for ct in f1:
                add(m, "B traits", ct["name"], "-", interaction(old, ct))
            for ct in f23:
                add(m, "B traits", ct["name"], "-", interaction(new, ct))
            if "NONE" in old and "NONE" in new:
                checks.append((m, "B traits: largest NONE difference between "
                               "runs",
                               f"{max_abs_diff(old['NONE'], new['NONE']):.5f} "
                               f"log prob"))

        if have("e1_choice_pairs", "e1_choice"):
            pr, bs = load_pairs(path("e1_choice_pairs")), load_choice_base(
                path("e1_choice"))
            for ct in f1:
                for side in ("vs_a", "vs_b"):
                    add(m, "C2 choice pairs", ct["name"], side,
                        pairs_interaction(pr, bs, ct, side))
        if have("e1_choice_pairs_floor", "e1_choice_floor"):
            pr, bs = load_pairs(path("e1_choice_pairs_floor")), \
                load_choice_base(path("e1_choice_floor"))
            for ct in f23:
                for side in ("vs_a", "vs_b"):
                    add(m, "C2 choice pairs", ct["name"], side,
                        pairs_interaction(pr, bs, ct, side))

    # one family per model and channel: every floor overall and within every
    # stratum, Holm per test, x2 across the two tests
    families = defaultdict(list)
    for r in rows:
        families[(r["model"], r["channel"])].append(r)
    for key, fam in families.items():
        A2A.holm_either(fam + strat.get(key, []))

    print("=" * 84)
    print("THREE FLOORS, EVERY CHANNEL. Two socially neutral details, stacked the")
    print("same way two disclosures are. A floor that moves means the instrument")
    print("responds to combining ANY two clauses and cannot identify a social")
    print("interaction.")
    print("=" * 84)
    verdicts = []
    for m in MODELS:
        for ch in ("A accuracy", "A2 answer logprob", "B traits",
                   "C2 choice pairs"):
            sel = [r for r in rows if r["model"] == m and r["channel"] == ch]
            if not sel:
                continue
            print()
            print(f"  {m}   {ch}")
            for r in sel:
                star = " *" if r["holm"] < 0.05 else ""
                print(f"    {r['floor']:16}{r['side']:6}n={r['n']:4}  "
                      f"{r['mean']:+9.4f} [{r['lo']:+8.4f},{r['hi']:+8.4f}]  "
                      f"MDE {r['mde']:.4f}  Holm rank {r['holm_rank']:<9.3g}"
                      f" mean {r['holm_mean']:<9.3g}{star}")
            bad = sum(1 for r in sel if r["holm"] < 0.05)
            verdict = "CLEAN" if bad == 0 else "CONTAMINATED"
            srows = strat.get((m, ch), [])
            if srows:
                flagged = [r for r in srows if r["holm"] < 0.05]
                print(f"    within difficulty strata, same family, "
                      f"{len(srows)} floor x stratum tests: "
                      + (", ".join(f"{r['name']} {r['stratum']} "
                                   f"{r['mean']:+.3f} (Holm {r['holm']:.3g})"
                                   for r in flagged) if flagged else "none move"))
                if verdict == "CLEAN" and flagged:
                    verdict = "CLEAN IN AGGREGATE ONLY"
            verdicts.append((m, ch, verdict, bad, len(sel)))

    print()
    print("=" * 84)
    print("VERDICT PER MODEL AND CHANNEL")
    print("=" * 84)
    for m, ch, verdict, bad, n in verdicts:
        print(f"  {m:9}{ch:20}{verdict:25}{bad} of {n} floor estimates "
              f"significant in aggregate")

    if csv_out:
        # Appendix A. These are the analyser's own rows, written out after the
        # same correction that decided the table above, so the appendix cannot
        # drift from it.
        import csv as _csv
        with open(csv_out, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["model", "instrument", "floor pair", "slot order",
                        "scope", "n", "interaction", "raw p signed-rank",
                        "raw p sign-flip", "Holm p signed-rank",
                        "Holm p sign-flip", "decision p", "family"])
            for key in sorted(families):
                fam = f"{key[0]} x {key[1]}"
                for r in families[key]:
                    w.writerow([r["model"], r["channel"], r["floor"],
                                r["side"], "aggregate", r["n"],
                                f"{r['mean']:+.4f}", f"{r['p_rank']:.4g}",
                                f"{r['p_mean']:.4g}", f"{r['holm_rank']:.4g}",
                                f"{r['holm_mean']:.4g}", f"{r['holm']:.4g}",
                                fam])
                for r in strat.get(key, []):
                    w.writerow([key[0], key[1], r["name"], "-",
                                r["stratum"], r.get("n", ""),
                                f"{r['mean']:+.4f}",
                                f"{r.get('p_rank', float('nan')):.4g}",
                                f"{r.get('p_mean', float('nan')):.4g}",
                                f"{r['holm_rank']:.4g}",
                                f"{r['holm_mean']:.4g}", f"{r['holm']:.4g}",
                                fam])
        print()
        print(f"appendix rows written to {csv_out}")

    if checks:
        print()
        print("-" * 84)
        print("REPEATABILITY: NONE collected in two separate runs")
        print("-" * 84)
        for m, what, value in checks:
            print(f"  {m:9}{what} {value}")

    if missing:
        print()
        print("not computed:")
        for x in missing:
            print(f"  {x}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--csv=")]
    csv_out = next((a.split("=", 1)[1] for a in sys.argv[1:]
                    if a.startswith("--csv=")), None)
    target = args[0] if args else os.path.join(HERE, "..", "outputs")
    main(os.path.abspath(target), csv_out)
