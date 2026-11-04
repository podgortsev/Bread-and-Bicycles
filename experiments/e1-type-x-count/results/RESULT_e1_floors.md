# RESULT: the floor check, five instruments, three models

**The central result of Paper 3.** Before any question about stacking can be
answered, the instrument must be shown to measure people rather than words. It
mostly cannot.

Source: `analysis_e1_floors.txt`, produced by `analyse_e1_floors.py`.

---

## What a floor is

Every audited pair puts two socially meaningful signals into the same message
and reads the interaction

    d = (both - A) - (B - none)

A FLOOR pair does exactly the same thing with two socially neutral but
distinctive details, in the same slots, under the same arithmetic:

| floor | the two details | collected in |
|---|---|---|
| floor 1 | cycling to work, baking bread | the main run |
| floor 2 | walking a dog, learning guitar | the extension |
| floor 3 | making tea, reading history books | the extension |

Its interaction should be zero. If it is not, the instrument responds to
combining any two clauses, and a social interaction cannot be told apart from a
lexical one.

Three floors, not one. Until 2026-09-10 every floor verdict in this paper rested
on the cycling-and-bread pair alone, and one neutral pair can itself be unusual.

## The rule

A channel is CLEAN on a model only if no floor moves. "Moves" is decided by two
tests, signed rank and a sign-flip permutation test on the mean, corrected as
ONE family per model and channel: Holm over every floor test, overall and within
three difficulty strata, then Bonferroni across the two tests. Measured false
alarm of that rule on synthetic null data: 2.0 percent.

The strata matter. A floor can be quiet on average because it moves one way on
easy items and the other way on hard ones, which is exactly what channel A2 does
on qwen.

---

## The table

| model | A task accuracy | A2 answer log prob | B trait log prob | C2 head-to-head |
|---|---|---|---|---|
| qwen | CLEAN | clean in aggregate only | CONTAMINATED 3/3 | CONTAMINATED 6/6 |
| llama | CLEAN | CONTAMINATED 2/3 | CONTAMINATED 3/3 | CONTAMINATED 6/6 |
| mistral | CLEAN | CONTAMINATED 2/3 | CONTAMINATED 3/3 | CONTAMINATED 5/5 |

Channel C, the first head-to-head read, is not in the table because its
interaction contrast is invalid by construction (margins against a shared
reference do not difference into an interaction). Its floor is also large:
+0.50 on llama, +1.35 on mistral, +3.56 on qwen.

### Channel A, task accuracy: the one clean zero

Nine floor estimates, none significant, none above its own minimum detectable
effect:

| model | floor 1 | floor 2 | floor 3 | MDE |
|---|---|---|---|---|
| qwen | +0.0000 | +0.0050 | -0.0150 | 0.027 to 0.032 |
| llama | +0.0100 | -0.0133 | +0.0150 | 0.030 to 0.035 |
| mistral | +0.0217 | +0.0033 | -0.0183 | 0.032 to 0.034 |

Within difficulty strata, 9 floor-by-stratum tests per model: none move, on any
model. This is the only instrument in the paper that passes that check.

### Channel A2, answer log probability: fails, and one failure is subtle

- **llama:** floors 2 and 3 move in aggregate (+0.135, +0.214).
- **mistral:** floors 2 and 3 move in aggregate (+0.370, +0.278).
- **qwen:** all three floors are quiet in aggregate (+0.182, +0.133, -0.015,
  MDE about 0.17), and floor 1 moves inside the near-ceiling stratum (-0.028,
  Holm 0.038). Far from the ceiling the same floors sit at +0.51 and +0.40.
  Two movements in opposite directions cancel in the average, so the verdict is
  CLEAN IN AGGREGATE ONLY and no interaction is identified.

### Channel B, trait log probability: fails everywhere, and not by an offset

| model | floor 1 | floor 2 | floor 3 |
|---|---|---|---|
| qwen | +0.553 | +0.379 | +0.464 |
| llama | -0.091 | -0.254 | +0.208 |
| mistral | +0.428 | +0.586 | +0.252 |

All nine significant. On llama the three floors do not even share a sign, so the
contamination is not a constant that a correction could subtract. For scale:
qwen's floor of +0.553 is larger than its stated-by-stated social interaction of
+0.400.

### Channel C2, the rebuilt head-to-head: fails everywhere

Every floor estimate significant on every model: llama +0.35 to +0.80, qwen
+0.65 to +6.44, mistral +0.79 to +4.04 on heavily gated data (32.2 percent of
mistral's reads carry less than half the probability mass on the two letters).

---

## It is not run-to-run noise

The signal-free condition NONE was collected twice, in two separate sessions.

- Channel A: the same right-or-wrong outcome on 100.0 percent of tasks, on all
  three models.
- Channel B: largest per-profile difference between runs 0.0053, 0.0024 and
  0.0020 log probability, against floor movements of 0.09 to 0.59.

## The mechanism, from the choice channel

The effect of a clause depends on the clause next to it. Re-derived
independently in `shared/reverify_findings.py`:

| model | cycling added to a neutral message | cycling added to the bread message |
|---|---|---|
| qwen | -4.39 | +2.05 |
| llama | -0.33 | +0.38 |

The same harmless detail helps or hurts depending on its neighbour. An
instrument with that property reports an interaction for any pair it is given.

## What this licenses, and what it does not

**Licensed.** Of five instruments that read a model's response to two stacked
signals, four cannot identify an interaction on any model, one can. Reporting a
compounding effect from any of the four, at any significance, is not supported.

**Not licensed.** That these instruments are useless. Their main effects, single
signal against none, are unaffected by this result: a single clause replacing a
neutral clause has no second clause to interact with. Channel B's and channel
C's main effects are reported and stand.

**Not licensed.** That disadvantage never compounds. What is shown is that on
these instruments and these models, a compounding claim cannot be separated from
the instrument's response to more text.
