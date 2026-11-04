# RESULT: channel A2, the probability of the correct answer

Added after the first analysis to close a gap: channel A is behavioural but
binary, and single signals move it only on llama. A2 reads the same prompts and
scores log P(correct answer), a continuous quantity, on 23 conditions x 600
tasks per model.

Source: `analysis_e1_answer_logprob.txt`. Floors: `RESULT_e1_floors.md`.

---

## The instrument is valid, and it still cannot answer the question

**Valid.** On the signal-free condition a higher answer probability means the
task was actually solved: AUC 0.914 on llama, 0.896 on mistral, 0.995 on qwen.
This reads behaviour, not style.

**Unusable for interactions.** Floors contaminated on llama and mistral, clean
in aggregate only on qwen. No interaction is identified on any model.

## What it does show, and it is one pattern on every model

The interaction takes the sign OPPOSITE to the two single effects. A second
clause adds less than the first, whichever way the clauses push. On qwen, far
from the ceiling:

| pair | first alone | second alone | both | interaction | reading |
|---|---|---|---|---|---|
| floor 1, neutral | -1.11 | -0.38 | -0.98 | +0.51 | cost does not add |
| floor 2, neutral | -1.09 | -0.90 | -1.60 | +0.40 | cost does not add |
| stated P2a | +0.48 | +0.14 | +0.04 | -0.57 | benefit does not add |
| stated P2b | +0.39 | +0.50 | -0.33 | -1.22 | benefit does not add |

Across all three models, of the 12 interactions that pass both tests:

- 7 are "benefit does not add"
- 2 are "cost does not add"
- 3 have singles pointing opposite ways, so no direction can be labelled
- 0 are compounding harm

The analyser labels every interaction by the direction of its two single
effects, because "more negative than the sum" means compounding only when the
singles are costs. On qwen and mistral the stated signals RAISE the probability
of the right answer, and two of them raise it less than the sum. That is
saturation of a benefit, not compounding of a harm.

## Two rival explanations, both ruled out

**Ceiling compression.** qwen answers 46 percent of tasks with P(correct) above
0.95, and a benefit that cannot exceed the ceiling would manufacture a negative
interaction by arithmetic alone. It does not explain this one: on the 250 tasks
far from the ceiling the stated-pair interactions are LARGER (-0.57 and -1.22),
not smaller.

**A batching artefact.** The first collection run stopped itself before writing
a row: its GPU self-check found that scoring 16 answers in one padded batch
moved them by up to 0.56 nats against a plain forward pass, while scoring each
alone matched the reference to four decimals. Every answer is now scored in a
forward pass of its own. The check is `P3Model.check_answer_scoring`; the token
arithmetic is proven offline against a tiny real model by
`shared/validate_score_answers.py`.

## Main effects

Single signals move this channel on all three models, which channel A could not
show. The directions differ:

- **llama:** age costs (-0.64 in slot 1, -0.24 in slot 2), typos help (+0.21).
- **qwen and mistral:** stated disclosures HELP (+0.15 to +0.30 overall, +0.39
  to +0.50 on hard tasks).
- **Neutral clauses move it as much as social ones, and on qwen more:** ALT5
  -0.93, ALT3 -0.50, ALT1 -0.47, against +0.04 to +0.20 for the four social
  signals.

Which text sits in the slot matters more than what that text says about a
person. That is the floor result again, measured on main effects instead of
interactions.

## What cannot be claimed

- Not that disclosure helps: the instrument that shows it responds to neutral
  clauses of the same size.
- Not any interaction from this channel, on any model.
- Not that qwen's floor is genuinely clean: it is quiet only because two
  significant movements in opposite strata cancel in the average.
