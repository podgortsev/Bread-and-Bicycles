# Validation

Everything in this paper is checked offline, without a GPU, before and after
collection. The logs in `validation/` are the output of the commands below, run
on the final code and the final data.

    cd experiments/e1-type-x-count/scripts
    python validate_e1_accuracy.py
    python validate_e1_channels.py
    python validate_e1_floor_extension.py
    cd ../../shared
    python validate_p3_experiments.py
    python reverify_findings.py
    python check_provenance.py
    python validate_score_answers.py        # needs CPU torch and transformers

    cd ../../../products/interaction-floor-test
    python test_floor_test.py

Part II's reruns carry their own validators, run from their own directories:

    cd experiments/r1-kirk-2021/scripts
    python r1_validate_addendum4.py
    cd ../../r2-winoidentity-2025/scripts
    python r2_validate.py
    cd ../../r4-neutral-not-unbiased-2025/scripts
    python r4_validate.py

## Summary

| check | what it covers | result |
|---|---|---|
| `validate_e1_accuracy.py` | channel A and the whole statistics module against hand-computed answers | 48 passed |
| `validate_e1_channels.py` | channels B and C, saturation and context dependence | 44 passed |
| `validate_e1_floor_extension.py` | the floor extension, channel A2, the two-test rule, strata, direction labels | 64 passed |
| `validate_p3_experiments.py` | e1w, e2, e2b, e3, e5 | 77 passed |
| `reverify_findings.py` | every headline number re-derived independently of the analysers | 43 passed |
| `check_provenance.py` | SHA-256 of everything inherited from Paper 2, and that tasks T001 to T200 still reproduce Paper 2 exactly | unchanged |
| `validate_score_answers.py` | the channel A2 scorer on a real tiny model on CPU | 14 passed |
| `r1_validate_addendum4.py` | the centred interval and the randomisation over descriptor pools, with an injected effect that must be found and a null that must not | all passed |
| `r2_validate.py` | the disparity statistic, the floor comparison and the scoring loop | 24 passed |
| `r4_validate.py` | the similarity and sentiment steps, and the fidelity rule, against injected answers | all passed |

All of the above were re-run in this repository after Part II's experiments were
merged in, and all still pass. Part II's result documents also state, for each
target, what cannot be claimed from it.
| `products/.../test_floor_test.py` | the released floor-check tool, including false-alarm rates over many seeds | 55 passed |

Total: 345 checks, 0 failures.

## What the checks are built to do

**Exercise the statistics on data with a known answer.** Every test injects an
effect of a stated size and requires the analyser to recover it, then injects a
null and requires the analyser NOT to find it. A test that can only pass is not
a test: `validate_p3_experiments.py` once caught a broken test of its own before
it could pass vacuously.

**Measure false alarms as a rate, not as a single seed.** The floor rule is run
over dozens to hundreds of synthetic null worlds and the alarm rate is reported:
2.0 percent for the full floor rule (aggregate plus strata, two tests), 5.7
percent for the aggregate-only rule that preceded it.

**Reproduce the failure before trusting the fix.** Each guard added late in the
project has a synthetic world built to reproduce the exact problem it was added
for: a ceiling with a heavy tail where the two tests disagree, floors that
cancel across strata, binary outcomes where most differences are exactly zero,
and a shared neutral cell that biases a stratifier.

## What the validation caught

Every item here changed a number or a conclusion.

1. **20 percent power at n=200** for the interaction on channel A, measured
   before any compute was spent. The task set was extended to 600.
2. **A shared module global** through which experiment e3 overwrote experiment
   e1's file prefix, which would have sent e1's rows into e3's file on a rerun.
   Found by the Colab smoke test.
3. **Floors corrected together with the contrasts they protect.** Under one
   joint Benjamini-Hochberg, strong audited effects make the correction more
   lenient for the floors, so the more an audit finds, the easier a quiet floor
   is flagged. Found by the product's unit tests; floors now use Holm as their
   own family everywhere.
4. **Batched answer scoring drifted** by up to 0.56 nats against a plain forward
   pass. The GPU self-check stopped the run before a row was written; scoring is
   now one answer per pass.
5. **A rank test and a mean disagreeing** on a ceiling-heavy channel (mean +0.18
   with an interval excluding zero, signed-rank p 0.87). Both tests are now run,
   and the rule can only remove claims.
6. **A floor quiet in aggregate but moving within strata**, on qwen's A2.
7. **A stratifier biased by a shared neutral cell**, which flagged a floor that
   does not move on real llama data. Found by a null-rate test.
8. **NaN p-values entering a correction** in e5, which turned one significant
   correlation cell into five. `bh` and `holm` now refuse NaN outright.
9. **An anchoring flag that scanned the question text**, which made llama look
   like it copied 40 percent of its answers from the prompt. Recomputed from the
   person's text: 0.0 to 2.3 percent.
10. **A verdict line using a magnitude heuristic** (floor against half the
    biggest effect) instead of the floor test, which called mistral's clean
    channel A floor contaminated.

## Reproducing the analysis

The analysers read the collected CSVs in `experiments/*/outputs/` and print the
files in `experiments/*/results/`. No GPU, no network. Collection itself needs a
GPU and is driven by `experiments/shared/run_p3_all.py` and `run_p3_floor.py`.
