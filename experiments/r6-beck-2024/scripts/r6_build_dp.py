#!/usr/bin/env python3
"""
r6_build_dp.py - their Diverse Perspectives sample, rebuilt with the code of
their Dataset_Preparation_Sampling.ipynb (cells 6, 7 and 19), unchanged except
for file paths.

    python r6_build_dp.py --ratings toxicity_ratings.json --out <jsonl>

Input: toxicity_ratings.json from Kumar et al. (SOUPS 2021), the encrypted
archive at scans.io/data/toxicity_perspectives. Not redistributable: the file
stays outside the repository and nothing derived from it is committed except
counts.

Their three filters drop a comment if ANY of its ratings is incomplete:
'Prefer not to say' on one of the five attributes, more than one race, or a
value they excluded for rarity. Then sample(n=1000, random_state=24).

Note their spelling 'political_affilation', which is the key in the data.
"""

import argparse

import pandas as pd


# cell 6, verbatim
def filter_PreferNotToSay(row):
    ratings_specified = []
    for r in row.ratings:
        try:
            if any(['Prefer not to say' in r['gender'], 'Prefer not to say' in r['race'],
                           'Prefer not to say' in r['education'], 'Prefer not to say' in r['age_range'],
                           'Prefer not to say' in r['political_affilation']]):
                ratings_specified.append(False)
            else:
                ratings_specified.append(True)
        except:
            ratings_specified.append(False)
    return all(ratings_specified)


def filter_multiRace(row):
    ratings_specified = []
    for r in row.ratings:
        try:
            if ',' in r['race']:
                ratings_specified.append(False)
            else:
                ratings_specified.append(True)
        except:
            ratings_specified.append(False)
    return all(ratings_specified)


def filter_UnknownValues(row):
    ratings_specified = []
    for r in row.ratings:
        try:
            if 'Other' in r['race'] or 'Unknown' in r['gender'] or 'Other' in r['political_affilation'] or 'Other' in r['education']:
                ratings_specified.append(False)
            else:
                ratings_specified.append(True)
        except:
            ratings_specified.append(False)
    return all(ratings_specified)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=24)
    args = ap.parse_args()

    # cell 7
    df = pd.read_json(args.ratings, lines=True, orient='records')
    print('Initial dataset size: {}'.format(len(df)), flush=True)
    df = df[df.apply(lambda row: filter_PreferNotToSay(row), axis=1)]
    print('After filtering empty attributes: {}'.format(len(df)), flush=True)
    df = df[df.apply(lambda row: filter_multiRace(row), axis=1)]
    print('After filtering multi-race attributes: {}'.format(len(df)), flush=True)
    df = df[df.apply(lambda row: filter_UnknownValues(row), axis=1)]
    print('After filtering less common attribute values: {}'.format(len(df)), flush=True)

    # cell 19
    sample = df.sample(n=1000, random_state=args.seed)
    sample.to_json(args.out, lines=True, orient='records')

    counts = {}
    for row in sample.to_dict(orient='records'):
        for r in row['ratings']:
            for att in ['gender', 'race', 'education', 'age_range', 'political_affilation']:
                counts.setdefault(att, {})
                counts[att][r[att]] = counts[att].get(r[att], 0) + 1
    print("=" * 78)
    print(f"R6 Diverse Perspectives sample: {len(sample)} comments, "
          f"{sum(len(r['ratings']) for r in sample.to_dict(orient='records'))} ratings")
    print("=" * 78)
    for att, vals in counts.items():
        total = sum(vals.values())
        print(f"  {att}:")
        for value, n in sorted(vals.items(), key=lambda kv: -kv[1]):
            print(f"    {value:70} {n:6} {n / total:6.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
