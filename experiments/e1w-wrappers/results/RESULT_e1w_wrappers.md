# RESULT: e1w, the same interactions under six neutral wrappers

The robustness experiment. Every e1 interaction, measured six times, changing
nothing but the neutral opening sentence around the signal.

Source: `analysis_e1w.txt`. 5,400 generations per model.

---

## The question

Paper 2 found that six arbitrary but defensible openings moved the baseline by 2
to 5 accuracy points, and Tonneau et al. (2026) report the same instability for
demographic cues at a scale of 14.8 million prompts: the magnitude and even the
direction of a group difference depend on how the group is cued.

If an interaction is smaller than the spread it shows across neutral wrappers,
it is a fact about the wrapper, not about the person.

## The result: nothing clears its own wrapper spread

| model | contrast | mean across wrappers | spread | wrappers agreeing in sign |
|---|---|---|---|---|
| llama | form x form | -0.012 | 0.050 | 4 of 6 |
| llama | stated x stated | +0.025 | 0.080 | 5 of 6 |
| llama | cross | -0.003 | 0.070 | 4 of 6 |
| mistral | form x form | +0.002 | 0.080 | 3 of 6 |
| mistral | stated x stated | +0.012 | 0.080 | 5 of 6 |
| mistral | cross | -0.002 | 0.080 | 4 of 6 |
| qwen | form x form | +0.000 | 0.090 | 4 of 6 |
| qwen | stated x stated | -0.012 | 0.160 | 5 of 6 |
| qwen | cross | +0.032 | 0.070 | 6 of 6 |

The spread is 3 to 15 times the mean. Eight of nine interactions change sign
between one neutral opening and another. The ninth, qwen cross, keeps its sign
in all six wrappers but its mean (+0.032) sits inside the spread (0.070).

Pooled across wrappers, treating wrapper as a replicate: no contrast survives
correction (adjusted p 0.29 to 1.00). With six observations that test is weak by
construction, and it is reported for completeness only. The sign agreement and
the spread carry the argument.

## What this means for the rest of the paper

Any single-wrapper interaction estimate in this literature carries this spread
as unreported uncertainty. That includes ours: every e1 number is a
single-wrapper number, and this experiment is the error bar around it.

It also sets the bar that the one surviving e1 interaction has to clear. On
llama, stated by stated on channel A is +0.072 in the main run; the wrapper
spread for that contrast is 0.080. The honest statement is that the effect is
the size of the wrapper spread, so it is reported as a direction with a stated
limit, not as a settled quantity.

## What cannot be claimed

- Not that the interactions are zero. A wrapper-dependent result is not a null;
  it is an effect that belongs to a particular presentation.
- Not that six wrappers bound the instability. Six is what was affordable, and
  the spread is an estimate from six points.
