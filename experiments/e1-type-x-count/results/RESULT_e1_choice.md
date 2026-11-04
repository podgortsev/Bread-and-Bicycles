# RESULT: channels C and C2, head-to-head choice. Invalid, then rebuilt, then failed

Two attempts at the same measurement, reported together because the second one
is what the first one's failure demanded.

Sources: `analysis_e1_choice.txt`, `analysis_e1_choice_pairs.txt`.
Floors: `RESULT_e1_floors.md`.

---

## Channel C: the contrast was invalid by construction

The first run compared every condition against the signal-free twin and read the
margin in logits of P(A) against P(B), both slot orders, gated on the two
letters carrying at least half the probability mass.

The interaction was then built by differencing those margins. That is not an
interaction. Each single differs from the reference in one place, the both-cell
differs in two, and subtracting margins measured against a shared reference does
not isolate the extra cost of the second signal. The floor says the same thing
numerically: +0.499 on llama, +1.346 on mistral, +3.564 on qwen, against social
interactions of the same size or smaller.

**Main effects from this channel are valid and are reported.** A margin between
two candidates differing in exactly one clause is a clean comparison: on llama,
age 74 in slot 1 costs -1.305 logits, screen reader in slot 2 gains +0.547; on
qwen every disclosure and every neutral detail loses to the twin.

## Channel C2: the rebuild, and its floor

The fix was to collect the comparison the contrast actually needs: the both-cell
against each single directly, so the two candidates differ in exactly one
clause. 6,400 extra forward passes per model.

It failed the same check:

| model | floor estimates | usable items |
|---|---|---|
| llama | +0.630, +0.701, +0.797, +0.469, +0.351, +0.382 | 200 per side |
| qwen | +0.647, +6.440, +2.673, +5.208, +4.887, +5.490 | 200 per side |
| mistral | +2.642, +3.337, +1.202, +4.043, +0.794 | 12 to 127 per side |

Every floor significant. The social interactions sit in the same range, so
nothing separates them from two meaningless details.

## The mechanism, which is why no arithmetic rescues it

The effect of a clause depends on the clause next to it:

| model | cycling added to a neutral message | cycling added to the bread message |
|---|---|---|
| qwen | -4.39 | +2.05 |
| llama | -0.33 | +0.38 |

A neutral detail flips sign with its neighbour. Correcting by the floor does not
help either: floor-corrected estimates flip sign between the two directions of
the same pair (qwen form pair +4.94 and -2.39, llama form pair -0.09 and +0.69).
One neutral pair is not a stable reference when the instrument is context
dependent.

## The letter-mass gate, and mistral

Every read records P(A)+P(B). On mistral 32.2 percent of the 12,800 pairs reads
carry less than half the mass on the two letters, and after requiring both slot
orders to survive, some contrasts are left with 12 to 15 items. Mistral's
numbers on this channel are reported with their n and are not treated as
evidence. This is the fourth appearance of the same refusal in the programme.

## Correction to an earlier claim

An earlier draft said the validator "proves the rebuilt contrast is robust". It
proved robustness to saturation only. The real failure is context dependence,
which the saturation model did not contain. Section I of
`validate_e1_channels.py` now shows that no pairwise contrast survives context
dependence and that the floor is the only defence.

## Status

Channel C and C2 are retired for interactions on all three models. Their main
effects stand and are reported.
