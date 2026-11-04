# e3: does a credential buy back a disclosure

**Question.** Everything else in Paper 3 asks whether two unfavourable signals
compound. This asks the question a hiring manager actually has: if the model
marks a person down for disclosing something, does a strong credential buy it
back, and is the exchange rate symmetric.

## The only prior art

Leyva-Vazquez and Smarandache (arXiv:2608.18107) document a "rescue effect" in
three factorial experiments with bootstrap intervals: a Nature publication
compensates for a low-prestige institution, and compensates MORE for the
lower-prestige candidate (+2.127) than for MIT (+1.745).

That is prestige against prestige. Nobody has asked it for a disclosure, where
the two signals are not on the same dimension at all: one is capability
evidence, the other is about how the person works.

## Two quantities, and the second is the reader's

**Interaction**, the usual (both - A) - (B - none): is the combination worth
more than the sum of its parts.

**Restoration**: is the person BACK AT THE CONTROL.

    recovered = 1 - (combined - NONE) / (disclosure alone - NONE)

A credential can fully restore someone with no interaction at all, simply by
being large enough. The fraction is only computed where the disclosure had a
main effect to recover from; dividing by a denominator that is itself noise
produces spectacular ratios that mean nothing, and the analyser refuses to print
one.

**Symmetry**: the same credential against two disclosures. If it buys back one
and not the other, the exchange rate belongs to what is being traded against,
not to the credential. Paper 2 gives a reason to expect this: age lowered a
numeric score while a screen reader RAISED it on the same instrument.

## Design

Eleven conditions, the same two-slot shape as e1 so the credential never arrives
as an extra sentence. Both credential-plus-disclosure pairs run in both orders.

Two channels, both about evaluating a person rather than arithmetic:

    D numeric   method 4's instrument, 6 measures, 6,600 generations
    C choice    method 6b's instrument, reused unchanged, 4,000 forward passes

## The anchoring exclusion

The credential says "fifteen years". Method 4's EXPERIENCE measure asks how many
years of experience the person appears to have, so it would measure copying, not
estimation. Paper 2 met exactly this: told a person is 74, models answer 74.

EXPERIENCE is excluded entirely, and every row records whether the answer equals
a number printed in the prompt, so the same failure cannot hide in the measures
that remain.

Refusal is counted separately and never scored as a low number. SUCCESS and
CHURN are kept as a matched opposite-valence pair so the coherence check still
works.

## Scripts

    ../shared/conditions/build_e3_conditions.py
    run_e3_numeric.py            channel D
    run_e3_choice.py             channel C, a thin wrapper on e1's instrument
    analyse_e3_offsetting.py     the read to trust, one channel per call
