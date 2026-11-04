# RESULT: channel B, trait log probability. A demonstrated instrument failure

Kept in the paper deliberately, as a failure rather than as a result.

The channel: the model is shown a person's message and the readiness to apply
six favourable words to that person is measured in log probability, averaged
over five phrasings, on 100 profiles. This is Paper 2's method 1 scorer, and a
common shape of bias measurement in the literature.

Source: `analysis_e1_traits.txt`. Floors: `RESULT_e1_floors.md`.

---

## The floor kills it on all three models

Two socially neutral details, stacked exactly as two disclosures are:

| model | floor 1 | floor 2 | floor 3 | largest social interaction |
|---|---|---|---|---|
| qwen | +0.553 | +0.379 | +0.464 | +1.231 |
| llama | -0.091 | -0.254 | +0.208 | +0.558 |
| mistral | +0.428 | +0.586 | +0.252 | +0.658 |

All nine significant on both tests. On llama the three floors do not share a
sign, so this is not an offset that a correction could remove.

Every social interaction on this channel is significant too, on every model, in
both directions: qwen form by form +0.548, mistral form by form -0.414, llama
cross +0.558. An instrument that finds a significant interaction for every pair
it is given, including pairs that mean nothing, is not measuring the pairs.

## The coherence check fails as well

A condition that lowers the favourable words should not also lower their
opposites.

| model | cell | positive | negative | verdict |
|---|---|---|---|---|
| llama | P1 form pair | -0.178 | -0.177 | incoherent |
| llama | S01 screen reader | -0.115 | -0.125 | incoherent |
| mistral | F01, P1, S01 | all | all | incoherent |
| qwen | F01, P1, S01 | -0.457, -0.934, -1.170 | +0.299, +0.260, +0.342 | coherent |

Two of three models move the favourable and the unfavourable words in the same
direction. That is a scale shift, not a judgement about a person.

## What survives

**Main effects.** A single clause replacing a neutral clause has no second
clause to interact with, so the floor result does not touch them, and they
replicate Paper 2's method 1 on new data:

| signal | llama | mistral | qwen |
|---|---|---|---|
| S01 screen reader, slot 1 | -0.115 | -0.297 | -1.170 |
| S01 screen reader, slot 2 | -0.668 | -1.054 | -1.985 |
| S08 age 74, slot 1 | -0.915 | -0.314 | (see file) |
| F04 typos | -0.352 | +0.531 | (see file) |

All significant on both tests. Disclosing a screen reader lowers the model's
readiness to apply favourable words to that person, on all three models, by up
to two log probability units.

**Neutral details also move it,** and by as much: qwen ALT1 -0.935, ALT2 -0.858,
mistral ALT1 -0.693. Read against a neutral detail rather than against silence,
the disclosure penalty is smaller than it looks against NONE.

## Why it stays in the paper

Deleting it would leave the paper's methodological argument resting on an
assertion about other people's instruments. Keeping it makes the argument a
measurement: here is a widely used instrument shape, here is what it reports for
two meaningless details, here is its coherence check failing, on real data from
three models.

No interaction from this channel enters any headline.
