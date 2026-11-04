#!/usr/bin/env python3
"""
r6_build_diaz.py - their Diaz sample, rebuilt with the code of their
Dataset_Preparation_Sampling.ipynb (cells 27 to 35 and 46), unchanged except for
file paths.

    python r6_build_diaz.py --diaz <dir with the two Dataverse zips unpacked> --out <jsonl>

Addendum 11. Input files from Harvard Dataverse doi:10.7910/DVN/F6EMTS
(train_older_adult_annotations.csv, test_annotations.csv) and
doi:10.7910/DVN/GXS7DI (demographics_responses.csv).
"""

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diaz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # cell 27
    sd = pd.read_csv(os.path.join(args.diaz, "demographics_responses.csv"))
    sd['Please indicate your race - Selected Choice'] = sd.apply(lambda x: 'Hispanic/Latino' if x['Are you Hispanic or Latino?']=='Yes' else x['Please indicate your race - Selected Choice'], axis=1)
    column_map = {
        'Please indicate your age': 'age_range',
        'Please indicate your race - Selected Choice': 'race',
        'Education': 'education',
        'Please indicate your political identification': 'political_affiliation',
        'Please indicate your gender': 'gender',
        'How would you describe the area where you grew up?': 'area_origin',
        'How would you describe the area in which you currently live?': 'area_living',
        'In which region of the United States do you currently live?': 'us_region',
        'Please indicate your annual household income': 'annual_household_income',
        'Which employment status best describes you?': 'employment_status',
        'How would you describe your current living situation (check all that apply) - Selected Choice': 'living_situation',
    }
    sd.rename(columns=column_map, inplace=True)
    sd.drop(columns=['Please indicate your race - Other - Text',
                     'Are you Hispanic or Latino?',
                     'How would you describe your current living situation (check all that apply) - Other - Text'],
            inplace=True)
    political_affil_map = {
        'Somewhat conservative': 'Conservative',
        'Very conservative': 'Conservative',
        'Moderate': 'Moderate',
        'Somewhat liberal': 'Liberal',
        'Very liberal': 'Liberal'
    }
    sd['political_affiliation'] = sd['political_affiliation'].apply(lambda x: political_affil_map[x])

    # cells 28 to 30
    df_train = pd.read_csv(os.path.join(args.diaz, "train_older_adult_annotations.csv"), encoding='latin-1')
    df_train = pd.merge(df_train, sd, on='respondent_id', how='inner')
    df_train['split'] = 'train'
    df_test = pd.read_csv(os.path.join(args.diaz, "test_annotations.csv"), encoding='latin-1')
    df_test = pd.merge(df_test, sd, on='respondent_id', how='inner')
    df_test['split'] = 'test'
    df = pd.concat([df_train, df_test])

    # cells 32 and 33
    def filter_UnknownValues(row):
        ratings_specified = []
        try:
            if 'Other' in row['race']:
                ratings_specified.append(False)
            else:
                ratings_specified.append(True)
        except:
            ratings_specified.append(False)
        return all(ratings_specified)

    print('Initial dataset size: {}'.format(len(df)))
    df = df[df.apply(lambda row: filter_UnknownValues(row), axis=1)]
    print('After filtering "Other" from race: {}'.format(len(df)))

    # cell 35
    df_agg = []
    for unit_id in df['unit_id'].unique():
        n = {'unit_id': unit_id, 'ratings': []}
        for row in df[df['unit_id']==unit_id].to_dict(orient='records'):
            if 'unit_text' not in n:
                n['unit_text'] = row['unit_text']
            rating = {**row}
            del rating['unit_id']
            del rating['unit_text']
            n['ratings'].append(rating)
        df_agg.append(n)
    df_agg = pd.DataFrame(df_agg)
    print('Unique text instances:', len(df_agg))

    # cell 46
    sample = df_agg.sample(n=1000, random_state=24)
    sample.to_json(args.out, lines=True, orient='records')
    n_ratings = sum(len(r) for r in sample['ratings'])
    print(f'sample: {len(sample)} instances, {n_ratings} annotator profiles, written to {args.out}')


if __name__ == "__main__":
    main()
