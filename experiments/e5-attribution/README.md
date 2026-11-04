# e5: can the model say WHICH of two signals moved it

**Question.** Method 7 asked, with one signal present, whether a model knows a
disclosure changed its answer. With two present the question sharpens into an
attribution question, and unlike method 7 there is a ground truth: the factorial
measured what each signal did on the same task.

No paper in the Paper 3 survey asks it.

## What method 7 established

- Qwen admits loudly and over-claims: stated 7.4 to 8.1 out of 10 against a
  measured 1.0 to 2.8 logits. The "alignment teaches models to hide it"
  prediction did not hold.
- Llama answered "6" to 57 of 60, standard deviation 0.45. Its self-report was a
  constant. Reading that as "the model does not admit it" was WRONG: the
  instrument said nothing.
- At item level, the correlation between claim and measured change was
  significant in ZERO of nine cells.

The degeneracy check therefore runs first here, before any other read, and a
model whose ratings have no variance is excluded.

## The asymmetry that shapes the prompt

For a stated pair the person said two things. For the form pair the person
mentioned NOTHING: the signals are non-native phrasing and typing mistakes,
which are properties of the writing.

So the second turn never says "you mentioned". It asks about two described
features of the message:

    S01   that I use a screen reader
    S08   that I am seventy-four
    F01   the way my English is written
    F04   the typing mistakes in my message

That buys the more interesting question for free: can a model attribute its
behaviour to a cue it was never told about. If it rates the typing at the floor
while e1 measured that the typing cost the person accuracy, the model is not
merely uncalibrated, it is unaware of the channel.

## Design

Seven both-cells from e1, 60 tasks each, 420 self-reports per model. **Nothing
is regenerated**: turn one is the answer e1 channel A actually collected,
replayed verbatim. A regenerated first turn would be a different exchange, and
the model would be reporting on something that never happened.

Requires e1 channel A to have run for the same model. If it has not, the script
says so and does nothing rather than inventing a first turn.

## What the analysis reports

0. the degeneracy check
1. what the model claims, per signal
2. **the ranking test**: within a pair, does it rate the signal that actually
   moved the answer above the one that did not. This is the attribution
   question, and method 7 could not ask it because with one signal there is
   nothing to rank
3. the item-level correlation, with its limit stated: per-item accuracy change
   is in {-1, 0, +1}, so a null here is weaker evidence than a null on a
   continuous outcome
4. the unstated-cue comparison

## Scripts

    run_e5_attribution.py        420 generations per model, two turns
    analyse_e5_attribution.py    give it BOTH this file and e1's channel A file
