# RESULT: channel A, task accuracy under stacked signals

The behavioural channel: does the model solve the person's own arithmetic task,
when the person's message carries one signal or two. Binary outcome, 600 tasks,
paired within task.

Source: `analysis_e1_accuracy.txt`. Floors: `RESULT_e1_floors.md`.

---

## Why this channel carries the paper

It is the only instrument here whose floor is clean on all three models, in
aggregate and within difficulty strata. Everything else in Paper 3 that reports
an interaction reports it from an instrument that also reports one for two
neutral details.

It is not the sensitive one. At n=600 it detects an interaction of 15 accuracy
points with 90 percent power and one of 5 points with 13 percent. Every estimate
below is printed with its own minimum detectable effect, and where the estimate
is smaller than the MDE the answer is "not visible from here".

## Main effects, exact McNemar, paired within task

Only llama moves at all.

| signal | llama | mistral | qwen |
|---|---|---|---|
| S01 screen reader, slot 1 | -0.050 (p=0.0001) | +0.008 | +0.003 |
| S01 screen reader, slot 2 | -0.063 (p<0.0001) | +0.025 | +0.005 |
| S08 age 74, slot 1 | -0.065 (p<0.0001) | +0.002 | -0.007 |
| S08 age 74, slot 2 | -0.033 (p=0.0151) | +0.010 | -0.015 |
| F01 non-native form | +0.015 | +0.008 | -0.010 |
| F04 typos | -0.027 (p=0.0166) | +0.003 | +0.008 |
| neutral details (ALT) | -0.017 to +0.008 | -0.032 to -0.017 | -0.012 to 0.000 |

On llama a stated disclosure costs 3 to 6.5 accuracy points on the person's own
task. On qwen and mistral nothing moves, so there is no main effect for an
interaction to modify, and the interaction nulls below say something about the
instrument on those models rather than about harm.

Paper 2's method 3c did find screen-reader costs on qwen with single-clause
lead-ins across six wrappers. The two-clause presentation used here may be what
removed it. That is a possibility, not a finding.

## Interactions

Negative is super-additive (the pair costs more than its parts), positive is
sub-additive. A star needs both tests to reject after Benjamini-Hochberg and the
mean to lean with the signed ranks.

| model | contrast | type | interaction | MDE | adjusted p |
|---|---|---|---|---|---|
| llama | P2b | stated x stated | **+0.0717** | 0.0364 | 0.0042 * |
| llama | P2a | stated x stated | +0.0383 | 0.0314 | 0.1736 |
| llama | P1 | form x form | -0.0100 | 0.0249 | 0.7771 |
| llama | P3a, P3b, P4a, P4b | cross | -0.022 to +0.028 | 0.024 to 0.030 | 0.42 to 0.78 |
| qwen | P2b | stated x stated | -0.0367 | 0.0305 | 0.1736 |
| qwen | P1 | form x form | -0.0183 | 0.0246 | 0.5219 |
| qwen | all others | | -0.013 to +0.013 | 0.024 to 0.030 | 0.52 to 1.00 |
| mistral | all | | -0.008 to +0.022 | 0.025 to 0.035 | 0.44 to 0.97 |

One interaction survives on one model: on llama, two stated disclosures together
cost 7.2 accuracy points LESS than the sum of their separate costs. It is
sub-additive, the opposite of compounding, and it agrees in sign with the other
slot order (+0.038, not claimed on its own).

Form by form is not distinguishable from zero on any model, with the estimate
below the MDE in every case.

## Order

| model | pairs that agree across slot orders | slot-dependent |
|---|---|---|
| llama | P2, P3, P4 | none |
| qwen | P2, P3 | P4 |
| mistral | P4 | P2, P3 |

## Refusal

Between 0 and 2 refusals per 10,200 generations, on every model and at every
signal count. Refusal does not stack here. Mistral's refusals, which dominated
Paper 2's numeric channel, do not appear when the task has one right answer.

## Against the registered expectations

Written before the run. Confirmation required form by form super-additive AND
stated by stated indistinguishable from zero, in two of three models on two of
three channels.

- Form by form: not super-additive anywhere. Half the thesis is refuted.
- Stated by stated: sub-additive and significant on llama, so not null there
  either, and in the direction opposite to compounding.
- Floor: clean on all three, which is what makes both statements readable.

This channel does not support the signal-type thesis. It supports the
sub-additive camp on one model and says nothing on two.
