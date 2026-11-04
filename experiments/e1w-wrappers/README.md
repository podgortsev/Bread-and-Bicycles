# e1w: the same interactions under six neutral wrappers

**Question.** Does e1's interaction survive a change of the neutral text around
the signal.

This is the card no competitor in this literature holds. Every stacking result
in the compounding camp is a single-wrapper result.

## Why it exists

Paper 2's method 3b ran the same 24 conditions twice, changing only the neutral
padding. Correlation between the runs: r=0.13 on Llama, r=0.22 on Qwen.
Individual conditions changed sign. The shape of Llama's curve inverted.

Tonneau et al. (arXiv:2601.18486, EMNLP 2026) report the same from a different
angle across 14.8 million prompts: both the magnitude and the DIRECTION of a
group difference vary with how the group is cued.

Method 3c's answer was to measure it deliberately across six wrappers and report
the spread as the error bar. That is what this does for the stacked case.

## Design

The six wrappers are Paper 2's method 3c wrappers, unchanged, so the two papers
measure wrapper sensitivity on the same axis. In Paper 2 they moved baseline
accuracy by 2 to 5 points on their own, before any signal.

Nine of e1's seventeen conditions: NONE, the four singles, and the four
both-cells in one slot order. Enough to recompute every interaction contrast
once per wrapper. 100 tasks rather than 600, because what is estimated here is a
SPREAD over six numbers, and precision within each buys little.

The consequence is stated rather than hidden: each per-wrapper interaction is
noisy, and only the spread across wrappers is interpreted.

## Scripts

    run_e1w_wrappers.py       9 x 100 x 6 = 5,400 generations per model
    analyse_e1w_wrappers.py   the read to trust, no GPU

Inputs: `e1_conditions.json`, `tasks.json`.

## The rule this experiment adds

An interaction is reported as real only if it clears the across-wrapper spread.
That is stricter than significance, and it is the rule none of the compounding
literature applies. An effect whose sign flips across wrappers is a fact about
the wrapper, and gets reported as one rather than averaged away.
