#!/usr/bin/env python3
"""
r5_preprocess.py - Step 0a (Addendum 8): SOC coding, filtering, and the
comparison with Wilson and Caliskan's Table 1.

    python r5_preprocess.py --resumes Resume.csv --jobs JD_data.csv
                            --cache nioccs_cache.jsonl --out <dir>

Readings A, B, C in the registered order; the first that passes is used and its
filtered resume and job files are written to --out.
"""

import argparse
import json
import os

import pandas as pd

TABLE1 = {  # broad SOC: (title, resumes, descriptions)
    "11-101": ("Chief Executives", 49, 33),
    "11-202": ("Marketing and Sales Managers", 51, 81),
    "11-919": ("Misc. Managers", 112, 92),
    "13-107": ("Human Resources Workers", 22, 27),
    "13-201": ("Accountants and Auditors", 133, 99),
    "17-219": ("Misc. Engineers", 28, 112),
    "25-203": ("Secondary School Teachers", 20, 52),
    "27-102": ("Designers", 87, 55),
    "41-909": ("Misc. Sales and Related Workers", 52, 20),
}
THRESHOLD = 0.60


def first_line(text):
    for line in str(text).splitlines():
        if line.strip():
            return " ".join(line.split())
    return ""


def second_line(text):
    lines = [l for l in str(text).splitlines() if l.strip()]
    return " ".join(lines[1].split()) if len(lines) > 1 else ""


def load_cache(path):
    codes = {}
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        occ = rec["response"].get("Occupation") if isinstance(rec["response"], dict) else None
        if occ:
            codes[rec["text"]] = (occ[0]["SOCCode"], float(occ[0]["SOCProbability"]))
    return codes


def broad(code_prob):
    if not code_prob:
        return None
    code, prob = code_prob
    if prob < THRESHOLD or code.startswith("00-99"):
        return None
    return code[:6]


def reading(r, j, codes, resume_text, job_text):
    r = r.copy()
    j = j.copy()
    r["broad_occupation"] = [broad(codes.get(resume_text(x))) for x in r.to_dict("records")]
    j["broad_occupation"] = [broad(codes.get(job_text(x))) for x in j.to_dict("records")]
    r = r.dropna(subset=["broad_occupation"]).drop_duplicates(subset=["Resume_str"])
    j = j.dropna(subset=["broad_occupation"]).drop_duplicates(subset=["description"])
    rc = r["broad_occupation"].value_counts()
    jc = j["broad_occupation"].value_counts()
    keep = sorted(o for o in set(rc.index) & set(jc.index) if rc[o] >= 20 and jc[o] >= 20)
    return r[r["broad_occupation"].isin(keep)], j[j["broad_occupation"].isin(keep)], keep, rc, jc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resumes", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    r = pd.read_csv(args.resumes)
    j = pd.read_csv(args.jobs)
    codes = load_cache(args.cache)
    readings = [
        ("A", lambda x: first_line(x["Resume_str"])[:300], lambda x: str(x["position"])),
        ("B", lambda x: x["Category"].replace("-", " ").lower(), lambda x: str(x["position"])),
        ("C", lambda x: first_line(x["Resume_str"])[:300], lambda x: str(x["job"])),
        # Addendum 9
        ("D", lambda x: second_line(x["Resume_str"])[:300], lambda x: str(x["position"])),
        ("E", lambda x: " ".join(str(x["Resume_str"]).split())[:300], lambda x: str(x["position"])),
        ("F", lambda x: " ".join(str(x["Resume_str"]).split())[:1000], lambda x: str(x["position"])),
        ("G", lambda x: first_line(x["Resume_str"])[:300] + " "
                        + x["Category"].replace("-", " ").lower(), lambda x: str(x["position"])),
    ]
    print("=" * 78)
    print("R5 STEP 0a: preprocessing against their Table 1 (Addendum 8)")
    print("=" * 78)
    chosen = None
    for name, rt, jt in readings:
        rf, jf, keep, rc, jc = reading(r, j, codes, rt, jt)
        same = set(keep) == set(TABLE1)
        within = 0
        print(f"\nreading {name}: occupations kept {len(keep)}; same nine as Table 1: {same}")
        for occ, (title, tr, tj) in TABLE1.items():
            orr, oj = int(rc.get(occ, 0)), int(jc.get(occ, 0))
            ok_r = abs(orr - tr) <= 0.2 * tr
            ok_j = abs(oj - tj) <= 0.2 * tj
            within += ok_r + ok_j
            print(f"  {occ} {title:32} resumes ours {orr:4} theirs {tr:4} {'ok' if ok_r else '  '}"
                  f"   descriptions ours {oj:4} theirs {tj:4} {'ok' if ok_j else '  '}")
        extra = [o for o in keep if o not in TABLE1]
        if extra:
            print(f"  kept but not in Table 1: {extra}")
        passed = same and within >= 15
        print(f"  counts within 20 percent: {within} of 18   {'PASS' if passed else 'fail'}")
        if passed and chosen is None:
            chosen = name
            os.makedirs(args.out, exist_ok=True)
            rf.to_csv(os.path.join(args.out, "resumes_filtered.csv"), index=False)
            jf.to_csv(os.path.join(args.out, "jobs_filtered.csv"), index=False)
            break
    print(f"\nVERDICT 0a: {'PASS with reading ' + chosen if chosen else 'FAIL: reading A used, differences stated'}")
    if not chosen:
        rf, jf, keep, _, _ = reading(r, j, codes, readings[0][1], readings[0][2])
        os.makedirs(args.out, exist_ok=True)
        rf.to_csv(os.path.join(args.out, "resumes_filtered.csv"), index=False)
        jf.to_csv(os.path.join(args.out, "jobs_filtered.csv"), index=False)
    print("=" * 78)


if __name__ == "__main__":
    main()
