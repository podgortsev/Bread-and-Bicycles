# R1 Step 0: Kirk et al.'s numbers from their data

**Verdict under the preregistered rule: FAIL (largest difference 0.043 against a
tolerance of 0.02). Explained below; the explanation does not touch the floor
comparison.**

Their released functions, unmodified apart from one documented compatibility
shim (a one-element set passed as `columns` to `DataFrame.from_dict`), run on
their released per-completion job labels in pandas 1.5.3, statsmodels 0.13.5,
numpy 1.24.4. Full output: `r1_step0_replication.txt`.

## What reproduced exactly

- **Religion:** all 12 terms, difference 0.000.
- **Political:** all 6 terms, difference 0.000.
- Mean share of significant interaction terms per group: ethnicity 0.213
  published, 0.218 ours; religion 0.203 and 0.203; sexuality 0.278 and 0.273;
  political 0.273 and 0.273.

## What did not, and why

Every difference traces to regressions with empty or near-empty cells, where
logistic regression is perfectly or quasi-perfectly separated and different
statsmodels versions behave differently. Output:
`r1_step0_explained.txt`, `r1_step0_separation.py`.

| group | job | cells | our statsmodels | theirs |
|---|---|---|---|---|
| ethnicity | journalist | Hispanic man 0, Hispanic woman 1 | fits; the Hispanic coefficient runs to -420 with standard error 8.8e7, and the other terms keep finite p-values (woman and woman:asian p < 0.0001) | the same terms not significant, consistent with an older version returning undefined p-values for the whole model |
| ethnicity | model (the occupation) | base man, Black man, White man all 0 | refuses to fit (perfect separation) | fitted and counted |
| ethnicity | courier | | fits | absent |
| sexuality | 3 jobs near the count threshold | | fits | absent |

Those account for the 1/47 and 2/47 differences on ethnicity and the extra
regressions on sexuality.

## Why this does not affect the floor test

Steps 2 and 3 regenerate the social and the neutral templates and analyse both
with the same code and the same statsmodels, so the handling of separation is
identical on both sides of the comparison.

## A finding in its own right

Their counts include regressions with empty cells, where a p-value carries no
information: the occupation "model" has zero completions for three of the male
templates and still entered their table. How many of their significant
interaction terms come from separated regressions is computed in the analysis
step and reported beside the floor result.
