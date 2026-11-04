# R2 result: WinoIdentity (COLM 2025) against the floor

Mistral-7B-Instruct-v0.2, 4-bit, referent augmentation, both WinoBias types.
Scored 2026-09-13 to 2026-09-16 on the local RTX A3000 by `scripts/r2_score.py`
(single passes, eager attention; see the commits of 2026-09-13 for why).
Analysed 2026-09-16 by `scripts/r2_analyse.py`, written before any score existed.
Outputs: `outputs/`, `results/r2_score_console.txt`,
`results/r2_analysis_console.txt`.

## Step 0: PASS

Raw prompt format, chosen by the preregistered rule on demographic rows only:
Spearman 0.905 against their 22 Table 2 values, mean absolute difference 0.029
(rule: at least 0.6 and at most 0.06). The chat format also passed (0.726,
0.058) and was not used.

## Step 1, the floor: 11 of 20 cells above it, 9 not distinguishable

A disparity is their Equation 2, maximum minus minimum mean coreference
confidence over gender by marker. Neutral markers sit in the same slot.

| attribute | type 1 | type 2 |
|---|---|---|
| age | not distinguishable (0.095 vs neutral max 0.136) | not distinguishable (0.195 vs 0.350) |
| body type | above (0.247 vs 0.136) | above (0.418 vs 0.350) |
| disability | above (0.193 vs 0.163) | above (0.422 vs 0.235) |
| gender identity | above (0.249 vs 0.136) | not distinguishable (0.073 vs 0.350) |
| language | above (0.181 vs 0.136) | above (0.357 vs 0.350) |
| nationality | not distinguishable (0.090 vs 0.136) | not distinguishable (0.103 vs 0.350) |
| sexual orientation | above (0.257 vs 0.136) | not distinguishable (0.245 vs 0.350) |
| socio-economic status | above (0.405 vs 0.136) | above (0.424 vs 0.350) |
| race | above (0.179 vs 0.163) | not distinguishable (0.170 vs 0.235) |
| religion | not distinguishable (0.081 vs 0.192) | not distinguishable (0.124 vs 0.340) |

What this does to their headline numbers:

- Their paper counts disparities "higher than 20%" as findings. A pair of
  neutral markers, early-rising and late-rising, reaches **0.350** on type 2
  sentences, and the three-marker neutral set reaches 0.340. On type 2, a
  disparity above 20 percent is what a neutral marker produces.
- The largest disparities (socio-economic status, body type, disability type 2)
  stay above every neutral set. Of the 11 cells above the floor, four clear it by
  less than the width of their own bootstrap interval: language type 2 (0.357
  against 0.350), body type type 2 (0.418 against 0.350), disability type 1
  (0.193 against 0.163) and race type 1 (0.179 against 0.163).

## Step 1, the doubly-disadvantaged pattern: not attributable to identity

D is the mean over sentence pairs of feminine minus masculine confidence on
male-dominated referent occupations. More negative means the feminine version
loses more.

- **With no marker at all, D = -0.721.** That is the most negative value in the
  whole table. The neutral markers give -0.489 to -0.676.
- The feminine disadvantage exists in the base sentences, and adding a marker,
  social or neutral, mostly **reduces** it: transgender -0.179, neurodivergent
  -0.245, Asian -0.241, white -0.264.
- By the preregistered rule, 8 of 10 attributes are "gender bias carried through
  any marker". The two exceptions are attributed to identity only formally: the
  most negative marker for disability is **able-bodied** (-0.677), the dominant
  group, and for language non-English-speaking (-0.718) is still no more
  negative than no marker.

So the pattern the paper reads as "exacerbated bias towards doubly-disadvantaged
identities" is, on this model and instrument, the gender bias of the sentences
themselves, present at its strongest before any identity is added.

## What cannot be claimed

- One of their five models, at 4-bit. The other four are not tested here.
- The preregistration asked for a paired bootstrap interval on D. The analyser
  prints the D values without it; the intervals are still to be added, and they
  decide nothing under the rule.
- Four "above floor" cells clear the floor by less than the bootstrap intervals'
  width. The rule decides on point values, as registered.
- The 4-bit scores depended on batch size on this GPU (0.1 to 0.25 nats), so the
  run used single passes. Step 0 passing at 0.905 is the check that this
  instrument matches theirs.
