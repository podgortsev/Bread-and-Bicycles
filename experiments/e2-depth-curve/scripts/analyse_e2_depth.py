#!/usr/bin/env python3
"""
analyse_e2_depth.py - the read to trust for e2. No GPU.

    python analyse_e2_depth.py e2_depth_qwen.csv e2_depth_llama.csv ...

Needs e2_conditions.json in the working directory.

WHAT IT PRODUCES

1. the depth curve per ladder, paired within task against depth 0
2. THE SECOND DIFFERENCE with a bootstrap interval, which is the only honest
   statement about shape. A curve bends upward if
   (D3 - D2) - (D2 - D1) is reliably negative
3. linear against quadratic fit on the same points, reported as the gain in
   fit rather than as a verdict
4. the paired depth-3 against depth-1 comparison over ROUTES, which is the
   comparison method 3b got wrong
5. the two ladders side by side, which is what the thesis predicts differ

WHY THE SECOND DIFFERENCE AND NOT A LABEL
-----------------------------------------
Method 3b assigned COMPOUNDING or MIXED with thresholds chosen for convenience
and said in its own result document that small changes in the increments flip
the label. A shape claim needs a second difference with an interval or a
comparison of fitted models. This does both and reports the interval, so a
reader can see when the answer is "the curve is too noisy to have a shape",
which at these sample sizes it often will be.

WHAT THE ROUTE COUNT MEANS
--------------------------
The form ladder has one condition at depth 3 and the stated ladder has three.
The number of independent routes behind every point is printed, because a mean
over one route has no spread to report and must not be read as if it did.
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


def load(paths):
    acc = defaultdict(lambda: defaultdict(dict))
    ref = defaultdict(lambda: defaultdict(list))
    meta = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                mk = r["model_key"]
                acc[mk][r["condition"]][r["task_id"]] = int(r["correct"])
                ref[mk][r["condition"]].append(int(r["refused"]))
                meta[r["condition"]] = (r["ladder"], int(r["depth"]),
                                        r["route"])
    return acc, ref, meta


def net_vs_base(cells, cid, base):
    ids = sorted(set(cells[cid]) & set(base))
    return [cells[cid][i] - base[i] for i in ids]


def fit_quadratic(xs, ys):
    """Least squares on 4 points. Returns (linear SSE, quadratic SSE, curve)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    a = my - b * mx
    sse_lin = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    # quadratic via normal equations on a 3-term basis
    import itertools
    basis = [[1.0, x, x * x] for x in xs]
    ata = [[sum(basis[k][i] * basis[k][j] for k in range(n))
            for j in range(3)] for i in range(3)]
    atb = [sum(basis[k][i] * ys[k] for k in range(n)) for i in range(3)]
    # 3x3 solve, Gaussian elimination with partial pivoting
    m = [row[:] + [atb[i]] for i, row in enumerate(ata)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return sse_lin, sse_lin, 0.0
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, 4):
                m[r][c] -= f * m[col][c]
    coef = [m[i][3] / m[i][i] for i in range(3)]
    sse_quad = sum((y - (coef[0] + coef[1] * x + coef[2] * x * x)) ** 2
                   for x, y in zip(xs, ys))
    return sse_lin, sse_quad, coef[2]


def main(paths):
    with open(_conditions_path("e2_conditions.json"), encoding="utf-8") as f:
        conditions = json.load(f)["conditions"]
    by_id = {c["id"]: c for c in conditions}
    acc, ref, meta = load(paths)
    models = sorted(acc)

    print("=" * 78)
    print("e2: does the penalty grow with the NUMBER of signals")
    print(f"models: {', '.join(models)}")
    print("=" * 78)

    all_rows = []
    for m in models:
        base = acc[m].get("D0", {})
        if not base:
            print(f"\n  {m}: no depth-0 cell, nothing can be referenced")
            continue
        print(f"\n{'-' * 78}\n{m}   depth 0 baseline "
              f"{sum(base.values()) / len(base):.1%} on {len(base)} tasks"
              f"\n{'-' * 78}")

        for ladder in ("form", "stated"):
            print(f"\n  {ladder.upper()} LADDER")
            depth_item_means = {}
            route_means = defaultdict(list)
            for depth in (1, 2, 3):
                cids = [c for c in acc[m]
                        if meta.get(c, ("", 0, ""))[0] == ladder
                        and meta[c][1] == depth]
                if not cids:
                    continue
                # per-condition (route) net, and the pooled per-item net
                pooled = []
                for cid in cids:
                    d = net_vs_base(acc[m], cid, base)
                    route_means[depth].append(sum(d) / len(d))
                    pooled.extend(d)
                depth_item_means[depth] = pooled
                mean, lo, hi = S.boot_ci(route_means[depth]) if len(
                    route_means[depth]) > 1 else (route_means[depth][0],
                                                  float("nan"), float("nan"))
                pm, plo, phi = S.boot_ci(pooled)
                tt = S.two_tests(pooled)
                ci = (f"[{lo:+.4f}, {hi:+.4f}]" if len(route_means[depth]) > 1
                      else "one route, no spread")
                print(f"    depth {depth}  routes={len(cids)}  "
                      f"route mean {mean:+.4f} {ci}")
                print(f"             pooled per task {pm:+.4f} "
                      f"[{plo:+.4f}, {phi:+.4f}]  p rank {tt['p_rank']:.4g} "
                      f"mean {tt['p_mean']:.4g}  MDE +/-{S.mde(pooled):.4f}")
                row = {"model": m, "ladder": ladder, "depth": depth,
                       "mean": pm}
                row.update(tt)
                all_rows.append(row)

            # ---- shape --------------------------------------------------
            if len(depth_item_means) == 3:
                xs = [0, 1, 2, 3]
                ys = [0.0] + [sum(depth_item_means[d]) / len(depth_item_means[d])
                              for d in (1, 2, 3)]
                second = (ys[3] - ys[2]) - (ys[2] - ys[1])

                # bootstrap the second difference over tasks
                ids = sorted(set(base))
                per_task = {}
                for d in (1, 2, 3):
                    cids = [c for c in acc[m]
                            if meta.get(c, ("", 0, ""))[0] == ladder
                            and meta[c][1] == d]
                    per_task[d] = {}
                    for i in ids:
                        vals = [acc[m][c][i] - base[i] for c in cids
                                if i in acc[m][c]]
                        if vals:
                            per_task[d][i] = sum(vals) / len(vals)
                common = sorted(set(per_task[1]) & set(per_task[2])
                                & set(per_task[3]))
                sd = [(per_task[3][i] - per_task[2][i])
                      - (per_task[2][i] - per_task[1][i]) for i in common]
                sm, slo, shi = S.boot_ci(sd)
                st = S.two_tests(sd)

                sse_l, sse_q, curve = fit_quadratic(xs, ys)
                print(f"\n    SHAPE")
                print(f"      depth means: 0.0000  " +
                      "  ".join(f"{y:+.4f}" for y in ys[1:]))
                print(f"      second difference (D3-D2)-(D2-D1): {sm:+.4f} "
                      f"[{slo:+.4f}, {shi:+.4f}]  p rank {st['p_rank']:.4g} "
                      f"mean {st['p_mean']:.4g}")
                bends = "bends downward (compounding)" if shi < 0 else (
                    "bends upward (saturating)" if slo > 0 else
                    "interval covers zero: NO SHAPE CAN BE CLAIMED")
                print(f"      {bends}")
                print(f"      quadratic term {curve:+.5f}, "
                      f"SSE linear {sse_l:.5f} -> quadratic {sse_q:.5f}")
                print(f"      Four points and three free parameters, so the")
                print(f"      quadratic fit cannot fail to improve. Read the")
                print(f"      second difference, not the SSE.")

                # ---- the comparison m3b got wrong ------------------------
                if len(route_means[1]) > 1 and len(route_means[3]) > 1:
                    k = min(len(route_means[1]), len(route_means[3]))
                    diff = [route_means[3][j] - route_means[1][j]
                            for j in range(k)]
                    wt = S.two_tests(diff)
                    print(f"\n      depth 3 against depth 1, PAIRED over "
                          f"{k} routes: p rank {wt['p_rank']:.4g}, "
                          f"p mean {wt['p_mean']:.4g}")
                    print(f"      Method 3b used an unpaired Mann-Whitney here")
                    print(f"      and its headline p moved from 0.031 to 0.0625")
                    print(f"      once corrected.")
                else:
                    print(f"\n      depth 3 against depth 1 not tested: "
                          f"{len(route_means[3])} route(s) at depth 3. "
                          f"The form ladder has one by construction.")

    # ---- both ladders together --------------------------------------------
    if all_rows:
        # BH over max(p rank, p mean): both tests, and the direction check
        S.correct(all_rows)
        print("\n" + "-" * 78)
        print("MULTIPLICITY. Every depth against depth 0, corrected together.")
        print("Method 3b applied no correction at depth and said so.")
        print("-" * 78)
        print(f"  {'model':9}{'ladder':9}{'depth':>6}{'net':>10}{'BH':>10}")
        for r in all_rows:
            star = " *" if r["claimed"] else ""
            print(f"  {r['model']:9}{r['ladder']:9}{r['depth']:6}"
                  f"{r['mean']:+10.4f}{r['bh']:10.4f}{star}")

    # ---- refusal -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("REFUSAL against depth. Counted separately, never scored as wrong.")
    print("-" * 78)
    for m in models:
        by_depth = defaultdict(list)
        for cid, vals in ref[m].items():
            if cid in by_id:
                by_depth[(by_id[cid]["ladder"], by_id[cid]["depth"])].extend(vals)
        print(f"\n  {m}")
        for k in sorted(by_depth):
            v = by_depth[k]
            print(f"    {k[0]:8}depth {k[1]}   {sum(v) / len(v):6.2%} "
                  f"({sum(v)} of {len(v)})")

    print("\n" + "=" * 78)
    print("The thesis predicts the two ladders have DIFFERENT shapes: the form")
    print("ladder bending downward, the stated ladder flat or saturating. Read")
    print("the second differences side by side, and read their intervals.")
    print("=" * 78)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        sys.exit("usage: analyse_e2_depth.py <csv> [<csv> ...]")
    missing = [a for a in args if not os.path.exists(a)]
    if missing:
        sys.exit(f"missing: {', '.join(missing)}")
    main(args)
