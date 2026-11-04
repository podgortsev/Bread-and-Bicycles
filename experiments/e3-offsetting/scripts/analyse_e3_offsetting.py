#!/usr/bin/env python3
"""
analyse_e3_offsetting.py - the read to trust for e3. No GPU.

    python analyse_e3_offsetting.py e3_numeric_qwen.csv [...]
    python analyse_e3_offsetting.py e3_choice_qwen.csv [...]

Needs e3_conditions.json in the working directory. The channel is detected from
the CSV header. Do not mix the two in one call: dollars and logits are not the
same unit.

TWO QUESTIONS, AND THE SECOND ONE IS THE READER'S
-------------------------------------------------
1. INTERACTION. Is credential-plus-disclosure worth more than the sum of its
   parts. This is the same statistic as everywhere else in Paper 3.

2. RESTORATION. Is the person BACK AT THE CONTROL. A credential can fully
   restore someone with no interaction at all, simply by being large enough,
   and that is the outcome a hiring manager cares about.

   restoration gap = (combined cell) - (NONE)
   recovered fraction = 1 - gap / (disclosure alone - NONE)

   A recovered fraction near 1 means the credential bought the person back.
   Near 0 means it did nothing. Above 1 means it overshot.

   The fraction is only computed when the disclosure had a main effect to
   recover from. Dividing by a denominator that is itself noise produces
   spectacular ratios that mean nothing, and the analysis refuses to print one.

3. SYMMETRY. The same credential against two different disclosures. If it buys
   back one and not the other, the exchange rate is a property of what is being
   traded against, not of the credential.

BEFORE ANY OF THAT, TWO GATES
-----------------------------
The numeric channel prints the refusal rate and the prompt-copy rate per
condition FIRST. Refusal is never a low number, and an answer copied from the
prompt is not an estimate. The choice channel prints the letter-mass gate.

COHERENCE
---------
SUCCESS and CHURN are matched opposites. A condition that raises the chance the
person succeeds must not also raise the chance they leave. That check caught a
broken instrument twice in Paper 2 and runs here on every condition.
"""

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared")))

import p3_stats as S      # noqa: E402


def _conditions_path(name):
    """Find a condition file whether the analyser is run from its own folder,
    from the repository root, or from a directory of downloaded CSVs."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (name,
                 os.path.join(here, name),
                 os.path.join(here, "..", "..", "shared", "conditions", name),
                 os.path.join(here, "..", "shared", "conditions", name)):
        if os.path.exists(cand):
            return cand
    raise SystemExit(
        f"{name} not found. Build it with the matching builder in "
        f"shared/conditions/, or run this from a directory that has it.")

MASS_GATE = 0.5
HIGHER_IS_BETTER = {"SALARY": True, "RAISE": True, "SUCCESS": True,
                    "CHURN": False, "SCORE": True, "RATING": True}

# a measure dies if it is refused away, answered with one constant, or copied
REFUSED_MAX = 0.50
CONSTANT_MAX = 0.70
ANCHORED_MAX = 0.30

# filled by per_measure_gates(), read by every table below
DEAD_MEASURES = set()


def person_numbers(paths):
    """What numbers the PERSON's own text contains, per profile and condition.

    The recorded `copied_from_prompt` column in the first run scanned the whole
    prompt, which includes the measure's own question and therefore its range,
    so every answer of 0 to a "from 0 to 100" question was flagged. On Llama
    that marked 99 percent of CHURN as anchoring when the real problem was that
    Llama answers 0. The flag is recomputed here from the profile and the
    lead-in only; the runner has been corrected for future runs.
    """
    import re
    here = os.path.dirname(os.path.abspath(__file__))
    prof_path = next((p for p in (
        "profiles.json",
        os.path.join(here, "..", "..", "shared", "profiles", "profiles.json"))
        if os.path.exists(p)), None)
    cond_path = next((p for p in (
        "e3_conditions.json",
        os.path.join(here, "..", "..", "shared", "conditions",
                     "e3_conditions.json")) if os.path.exists(p)), None)
    if not (prof_path and cond_path):
        return None
    profiles = [p["text"] for p in
                json.load(open(prof_path, encoding="utf-8"))]
    conds = {c["id"]: c["lead_in"] for c in
             json.load(open(cond_path, encoding="utf-8"))["conditions"]}
    nums = {}
    for i, text in enumerate(profiles):
        for cid, lead in conds.items():
            nums[(f"P{i + 1:03d}", cid)] = set(
                re.findall(r"\d+", text + " " + lead))
    return nums


def per_measure_gates(paths):
    """usable[model][measure] = counts; dead[model] = measures to exclude."""
    from collections import Counter
    nums = person_numbers(paths)
    rows_by = defaultdict(lambda: defaultdict(list))
    ref_by = defaultdict(lambda: defaultdict(list))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                mk, meas = r["model_key"], r["measure"]
                ref_by[mk][meas].append(int(r["refused"]))
                if r["refused"] == "1" or not r["value"]:
                    continue
                rows_by[mk][meas].append(r)

    usable, dead = {}, {}
    for mk in ref_by:
        usable[mk], dead[mk] = {}, set()
        for meas, refs in ref_by[mk].items():
            rows = rows_by[mk].get(meas, [])
            vals = [float(r["value"]) for r in rows]
            top, share = ("-", 0.0)
            if vals:
                c = Counter(vals)
                k, n = c.most_common(1)[0]
                top, share = f"{k:g}", n / len(vals)
            if nums is not None and rows:
                anch = sum(
                    1 for r in rows
                    if f"{float(r['value']):g}"
                    in nums.get((r["profile"], r["condition"]), set())
                ) / len(rows)
            else:
                anch = sum(int(r["copied_from_prompt"]) for r in rows) / \
                    max(1, len(rows))
            g = {"n": len(vals), "refused": sum(refs) / len(refs),
                 "top": top, "share": share, "anchored": anch}
            usable[mk][meas] = g
            if (g["refused"] > REFUSED_MAX or g["share"] > CONSTANT_MAX
                    or g["anchored"] > ANCHORED_MAX):
                dead[mk].add(meas)
    return usable, dead


def detect(path):
    with open(path, encoding="utf-8") as f:
        header = next(csv.reader(f))
    if "measure" in header:
        return "D"
    if "margin_signal" in header:
        return "C"
    raise SystemExit(f"{path}: not an e3 numeric or choice file")


def load_numeric(paths):
    val = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    ref = defaultdict(lambda: defaultdict(list))
    cop = defaultdict(lambda: defaultdict(list))
    oor = defaultdict(int)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                mk = r["model_key"]
                ref[mk][r["condition"]].append(int(r["refused"]))
                cop[mk][r["condition"]].append(int(r["copied_from_prompt"]))
                oor[mk] += int(r["out_of_range"])
                if r["refused"] == "1" or not r["value"]:
                    continue
                val[mk][r["measure"]][r["condition"]][r["profile"]] = \
                    float(r["value"])
    return val, ref, cop, oor


def load_choice(paths):
    """val[model]["CHOICE"][condition][item] = margin, both orders averaged."""
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    mass = defaultdict(list)
    gated = defaultdict(int)
    total = defaultdict(int)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                mk = r["model_key"]
                total[mk] += 1
                mass[mk].append(float(r["mass"]))
                if float(r["mass"]) < MASS_GATE:
                    gated[mk] += 1
                    continue
                item = f"{r['profile']}:{r['question']}"
                raw[mk][r["condition"]][item].append(float(r["margin_signal"]))
    out = {}
    for mk in raw:
        out[mk] = {"CHOICE": {}}
        for cid, items in raw[mk].items():
            cell = {}
            for item, vs in items.items():
                # both slot orders must survive the gate, or the slot term
                # does not cancel and the number is not comparable
                if len(vs) >= 2:
                    cell[item] = sum(vs) / len(vs)
            if cell:
                out[mk]["CHOICE"][cid] = cell
    return out, mass, gated, total


def paired(cells, a, b):
    if a not in cells or b not in cells:
        return []
    ids = sorted(set(cells[a]) & set(cells[b]))
    return [cells[a][i] - cells[b][i] for i in ids]


def main(paths):
    channels = {detect(p) for p in paths}
    if len(channels) > 1:
        sys.exit("numeric and choice files have different units; "
                 "analyse them in separate calls")
    channel = channels.pop()

    with open(_conditions_path("e3_conditions.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    contrasts, restoration = cfg["contrasts"], cfg["restoration"]

    if channel == "D":
        val, ref, cop, oor = load_numeric(paths)
        unit = "the measure's own units"
    else:
        val, mass, gated, total = load_choice(paths)
        unit = "logits of choice margin against NONE"
    models = sorted(val)

    print("=" * 78)
    print(f"e3 CHANNEL {channel}: does a credential buy back a disclosure")
    print(f"models: {', '.join(models)}   unit: {unit}")
    print("=" * 78)

    # ---- gates -------------------------------------------------------------
    print("\n" + "-" * 78)
    print("0. GATES. Nothing below is read until these are.")
    print("-" * 78)
    if channel == "D":
        for m in models:
            allr = [x for v in ref[m].values() for x in v]
            allc = [x for v in cop[m].values() for x in v]
            print(f"  {m:9}refusal {sum(allr) / len(allr):6.2%}   "
                  f"answer copied from the prompt {sum(allc) / len(allc):6.2%}"
                  f"   out of range {oor[m]}")
            worst = sorted(ref[m], key=lambda c: -sum(ref[m][c]) / len(ref[m][c]))[:3]
            for c in worst:
                v = ref[m][c]
                print(f"  {'':9}  most refused: {c:12}{sum(v) / len(v):6.2%}")
            if sum(allr) / len(allr) > 0.25:
                print(f"  {'':9}  MORE THAN A QUARTER REFUSED. Paper 2 saw 19 "
                      f"to 28 percent on Mistral. Report it as a finding.")

        print("\n" + "-" * 78)
        print("0b. PER MEASURE: is this measure usable on this model at all.")
        print("    Three ways a measure dies, and all three have happened:")
        print("    refused away, answered with one constant, or copied.")
        print("-" * 78)
        usable, dead = per_measure_gates(paths)
        for m in models:
            print(f"\n  {m}")
            for meas in sorted(usable.get(m, {})):
                g = usable[m][meas]
                verdict = "usable" if meas not in dead.get(m, set()) else "DEAD"
                print(f"    {meas:9}n={g['n']:5}  refused {g['refused']:6.1%}"
                      f"  top value {g['top']:>9} in {g['share']:5.1%}"
                      f"  anchored {g['anchored']:5.1%}   {verdict}")
            if dead.get(m):
                print(f"    excluded on this model: "
                      f"{', '.join(sorted(dead[m]))}")
        print("\n  A measure marked DEAD is not evidence of anything and is")
        print("  dropped from every table below. Method 7 taught this: Llama's")
        print("  constant '6' was read as 'the model does not admit it' when")
        print("  the instrument had simply said nothing.")
        DEAD_MEASURES.update({(m, x) for m in dead for x in dead[m]})
    else:
        for m in models:
            mm = mass[m]
            print(f"  {m:9}mean letter mass {sum(mm) / len(mm):.3f}   "
                  f"gated out {gated[m]} of {total[m]} "
                  f"({gated[m] / total[m]:.1%})")
            if gated[m] / total[m] > 0.5:
                print(f"  {'':9}  MOST READS UNUSABLE. Report as refusal, "
                      f"not as a result.")

    measures = sorted({k for m in models for k in val[m]})

    # ---- main effects ------------------------------------------------------
    print("\n" + "-" * 78)
    print("1. MAIN EFFECTS against NONE, paired within profile.")
    print("-" * 78)
    for m in models:
        for meas in measures:
            cells = val[m].get(meas, {})
            if (m, meas) in DEAD_MEASURES:
                continue
            if not cells:
                continue
            print(f"\n  {m}  {meas}")
            for cid in sorted(cells):
                if cid == "NONE":
                    continue
                d = (paired(cells, cid, "NONE") if channel == "D"
                     else list(cells[cid].values()))
                if not d:
                    continue
                mean, lo, hi = S.boot_ci(d)
                t = S.two_tests(d)
                print(f"    {cid:12}n={len(d):4}  {mean:+12.4f} "
                      f"[{lo:+11.4f},{hi:+11.4f}]  p rank {t['p_rank']:.4g} "
                      f"mean {t['p_mean']:.4g}")

    # ---- interaction -------------------------------------------------------
    print("\n" + "-" * 78)
    print("2. INTERACTION. Is the combination worth more than its parts.")
    print("-" * 78)
    rows = []
    for m in models:
        for meas in measures:
            cells = val[m].get(meas, {})
            if (m, meas) in DEAD_MEASURES:
                continue
            for ct in contrasts:
                need = [ct["both"], ct["single_a"], ct["single_b"]]
                if channel == "D":
                    need.append(ct["none"])
                if any(c not in cells for c in need):
                    continue
                ids = sorted(set.intersection(*[set(cells[c]) for c in need]))
                if not ids:
                    continue
                if channel == "D":
                    d = [(cells[ct["both"]][i] - cells[ct["single_a"]][i])
                         - (cells[ct["single_b"]][i] - cells[ct["none"]][i])
                         for i in ids]
                else:
                    d = [cells[ct["both"]][i] - cells[ct["single_a"]][i]
                         - cells[ct["single_b"]][i] for i in ids]
                mean, lo, hi = S.boot_ci(d)
                row = {"model": m, "measure": meas, "name": ct["name"],
                       "type": ct.get("type", "credential"), "n": len(d),
                       "mean": mean, "lo": lo, "hi": hi, "mde": S.mde(d)}
                row.update(S.two_tests(d))
                rows.append(row)
    # BH over max(p rank, p mean): both tests, and the direction check; any
    # floor rows are corrected family-wise on their own, per model and measure
    S.correct(rows, group=("model", "measure"))
    print(f"\n  {'model':9}{'measure':9}{'contrast':16}{'n':>5}"
          f"{'interaction':>14}{'MDE':>12}{'BH':>10}")
    for r in rows:
        star = " *" if r["claimed"] else ""
        print(f"  {r['model']:9}{r['measure']:9}{r['name']:16}{r['n']:5}"
              f"{r['mean']:+14.4f}{r['mde']:12.4f}{r['bh']:10.4f}{star}")

    # ---- restoration, the reader's question --------------------------------
    print("\n" + "-" * 78)
    print("3. RESTORATION. Is the person back at the control.")
    print("   recovered = 1 - (combined - NONE) / (disclosure alone - NONE)")
    print("   Only computed where the disclosure had an effect to recover from.")
    print("-" * 78)
    for m in models:
        for meas in measures:
            cells = val[m].get(meas, {})
            if (m, meas) in DEAD_MEASURES:
                continue
            if not cells:
                continue
            print(f"\n  {m}  {meas}")
            for rs in restoration:
                for combined, alone in zip(rs["combined"], rs["alone"]):
                    if combined not in cells or alone not in cells:
                        continue
                    if channel == "D":
                        gap = paired(cells, combined, "NONE")
                        harm = paired(cells, alone, "NONE")
                    else:
                        gap = list(cells[combined].values())
                        harm = list(cells[alone].values())
                    if not gap or not harm:
                        continue
                    g, glo, ghi = S.boot_ci(gap)
                    h, _, _ = S.boot_ci(harm)
                    gt = S.two_tests(gap)
                    denom_ok = abs(h) > S.mde(harm)
                    frac = (f"{1 - g / h:6.2f}" if denom_ok
                            else "  n/a")
                    note = ("" if denom_ok else
                            "   (the disclosure alone had no effect to recover)")
                    print(f"    {rs['signal']:5}{combined:12}gap to NONE "
                          f"{g:+10.4f} [{glo:+9.4f},{ghi:+9.4f}] "
                          f"p rank {gt['p_rank']:.3g} mean {gt['p_mean']:.3g}"
                          f"   alone {h:+9.4f}   recovered {frac}{note}")

    # ---- symmetry ----------------------------------------------------------
    print("\n" + "-" * 78)
    print("4. SYMMETRY. The same credential against two disclosures.")
    print("-" * 78)
    for m in models:
        for meas in measures:
            cells = val[m].get(meas, {})
            if (m, meas) in DEAD_MEASURES:
                continue
            per = {}
            for rs in restoration:
                gaps = []
                for combined in rs["combined"]:
                    if combined not in cells:
                        continue
                    d = (paired(cells, combined, "NONE") if channel == "D"
                         else list(cells[combined].values()))
                    if d:
                        gaps.append(sum(d) / len(d))
                if gaps:
                    per[rs["signal"]] = sum(gaps) / len(gaps)
            if len(per) == 2:
                keys = sorted(per)
                diff = per[keys[0]] - per[keys[1]]
                print(f"  {m:9}{meas:9}"
                      + "  ".join(f"{k} gap {per[k]:+.4f}" for k in keys)
                      + f"   difference {diff:+.4f}")

    # ---- coherence ---------------------------------------------------------
    if channel == "D" and "SUCCESS" in measures and "CHURN" in measures:
        print("\n" + "-" * 78)
        print("5. COHERENCE. More likely to succeed must not mean more likely")
        print("   to leave. This caught a broken instrument twice in Paper 2.")
        print("-" * 78)
        for m in models:
            if (m, "SUCCESS") in DEAD_MEASURES or (m, "CHURN") in DEAD_MEASURES:
                print(f"  {m:9}skipped: one half of the pair is not usable")
                continue
            s_cells = val[m].get("SUCCESS", {})
            c_cells = val[m].get("CHURN", {})
            if "NONE" not in s_cells or "NONE" not in c_cells:
                continue
            print(f"\n  {m}")
            for cid in sorted(s_cells):
                if cid == "NONE" or cid not in c_cells:
                    continue
                ds = paired(s_cells, cid, "NONE")
                dc = paired(c_cells, cid, "NONE")
                if not ds or not dc:
                    continue
                ms, mc = sum(ds) / len(ds), sum(dc) / len(dc)
                verdict = ("coherent" if ms * mc <= 0
                           else "INCOHERENT, both move the same way")
                print(f"    {cid:12}success {ms:+9.3f}  churn {mc:+9.3f}"
                      f"   {verdict}")

    print("\n" + "=" * 78)
    print("Read section 3 before section 2. The interaction says whether the")
    print("combination beat the sum of its parts; restoration says whether the")
    print("person got their treatment back, which is the question that gets")
    print("repeated back outside this field.")
    print("=" * 78)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        sys.exit("usage: analyse_e3_offsetting.py <csv> [<csv> ...]")
    missing = [a for a in args if not os.path.exists(a)]
    if missing:
        sys.exit(f"missing: {', '.join(missing)}")
    main(args)
