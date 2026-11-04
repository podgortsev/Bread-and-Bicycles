# R1 result: Kirk et al. (NeurIPS 2021) against the floor

GPT-2 124M, variant A (Addendum 1), 38 templates x 7,000 completions, CoreNLP
4.2.0 job titles, their regression functions unchanged. Analysed 2026-09-16 by
`scripts/r1_analyse.py` with Addendum 4 (centred intervals, randomisation over
descriptor pools), 100 resamples. Outputs: `outputs/completions_A.csv`,
`outputs/ner_A.csv`, `results/r1_analysis_console.txt`.

## Uninterpretability check: passed

No template has a no-title share above twice theirs (limit 0.212); the highest
is thirsty man at 0.191.

## Step 2, fidelity: PASS

Mean share of significant woman-by-religion terms: ours 0.242, theirs 0.203,
their centred interval [0.167, 0.245]. Ours lies inside it, near the top.

## Step 3, the floor: PARTIAL

The statistic is theirs: the share of per-job logistic regressions in which an
interaction term has p < 0.05, averaged over a group's terms.

| group | size | social | neutral | neutral interval (centred) | verdict |
|---|---|---|---|---|---|
| religion | 5 | 0.242 | 0.247 | [0.199, 0.290] | inside the neutral range |
| ethnicity | 4 | 0.224 | 0.221 | [0.178, 0.270] | inside the neutral range |
| sexuality | 2 | 0.289 | 0.216 | [0.147, 0.287] | above the neutral range |
| political | 2 | 0.291 | 0.216 | [0.147, 0.287] | above the neutral range |

**Secondary (Addendum 4), randomisation over descriptor pools, deciding nothing:**
religion p = 0.615 over 252 groups, ethnicity p = 0.643 over 70, sexuality
p = 0.333 over 15, political p = 0.267 over 15. No social group stands out from
the groups that can be formed from the same pool with neutral descriptors.

## Reading

- Their sentence "Interactions are significant in approximately one third of
  regressions" rests on a statistic whose neutral value is 0.22 to 0.25. Nominal
  chance is 0.05. Most of the "one third" is what this regression pipeline
  produces for adjectives such as hungry and sleepy.
- For religion and ethnicity, the social interaction share is no higher than the
  neutral one.
- Sexuality and political affiliation clear the neutral interval by 0.002 and
  0.004. They survive under the preregistered rule, against a single neutral
  pair (hungry, sleepy) shared by both. The randomisation test, which uses every
  size-2 group of the pool, does not separate them (p = 0.33 and 0.27, at a
  resolution of 0.067).
- **Mechanism, visible in the secondary table:** many regressions have an empty
  category-by-gender cell (41 of 64 for the neutral religion group, 41 of 67 for
  the social one, 38 of 63 in their released data). Logistic regression with an
  empty cell separates, and separated terms return extreme p-values. This inflates
  the share for any descriptor.

## What cannot be claimed

- GPT-2 124M only, as in their released data. Their larger models are not tested.
- Our social data are regenerated (Addendum 2); fidelity passed on religion only,
  the group their released regressions reproduce exactly.
- The two surviving groups share one neutral comparison group; a second neutral
  pair of size 2 was not registered.
