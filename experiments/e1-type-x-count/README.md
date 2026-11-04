# e1: signal type crossed with signal count

The core experiment of Paper 3.

**Question.** Does a second signal cost more when both signals change the text
than when both merely state a fact about the person.

**Working thesis.** Signals that change the text compound; signals that only
state a fact do not. If it holds, it explains why the published literature is
split: the compounding results come from form-like cues, the sub-additive ones
from stated attributes, and nobody has crossed the two in one design.

---

## Design in one screen

Four signals, drawn from `shared/axes/`:

    FORM     F01  non-native English shown by the writing
             F04  typos and keyboard slips
    STATED   S01  "I use a screen reader for everything on my computer."
             S08  "I am seventy-four and not very confident with computers."

Four 2x2 pairs: F01xF04, S01xS08, and two cross pairs (F01xS01, F04xS08).

Seventeen conditions, because the four 2x2s share cells. Every lead-in is
exactly two clauses. Full list and the reasoning in
`shared/conditions/build_e1_conditions.py`.

Interaction, per item:

    d = (both - single_A) - (single_B - none)

Negative is super-additive (the compounding claim). Positive is sub-additive
(Paper 2's method 3b, and Fu et al. 2026).

## Channels

| channel | script | outcome | items | cost per model |
|---|---|---|---|---|
| A task accuracy | `run_e1_accuracy.py` | binary correct | 600 tasks | 10,200 generations |
| B trait log probability | `run_e1_traits.py` | mean log prob of six favourable words | 100 profiles | 35,400 scored continuations |
| C head-to-head choice | `run_e1_choice.py` | choice margin against a signal-free twin | 200 profile-question pairs | 6,400 forward passes |

Channel A is the behavioural evidence and the gap in the literature. It is not
the sensitive one: at n=600 it detects an interaction of fifteen accuracy points
with 90 percent power and one of five points with 13 percent.

Channels B and C carry the interaction question. On the validator's synthetic
data their minimum detectable interaction is a small fraction of the
single-signal effects Paper 2 measured (method 1 ran 0.95 to 2.92 logits, method
6b ran 0.58 to 6.93). Those synthetic numbers assume an item noise that is a
guess until real data arrives, so they show the order of sensitivity, not its
value; the analyser recomputes the MDE from what is actually collected and that
is the number that goes in the paper.

Channel C reads the probability of a letter, which is the instrument Paper 2
saw fail on Mistral three separate times. Every read records P(A)+P(B), the gate
is applied in the analysis rather than at collection, and the analyser reports
how much data it removed. If more than half a model's reads are unusable, that
is reported as a refusal and not as a result.

---

## Run plan: ONE notebook per model

One notebook per model for the WHOLE paper, not just this experiment:
`shared/run_p3_all.py` loads the model once and runs all nine collection steps
of all six experiments against it. See `colab/HOW-TO.md` at the project root.

To run only this experiment: `!python run_p3_all.py qwen --only=e1a,e1b,e1c`.

**1. Offline first, no GPU. All four must pass.**

    cd experiments/shared            && python check_provenance.py
    cd experiments/shared/conditions && python build_e1_conditions.py
    cd experiments/shared/tasks      && python build_tasks_p3.py && python validate_tasks.py
    cd experiments/e1-type-x-count/scripts && python validate_e1_accuracy.py && python validate_e1_channels.py

**2. In Colab.** New notebook, Runtime > Change runtime type > T4 GPU.

    !pip -q install transformers accelerate bitsandbytes
    from google.colab import drive; drive.mount('/content/drive')

Upload one file, `p3_bundle.zip`, built by `colab/build_bundle.py`. The
notebook unzips it. After the first time it lives in Drive and is never
uploaded again.

For Llama and Mistral, log in first:

    from huggingface_hub import notebook_login; notebook_login()

Then one line:

    !python run_p3_all.py qwen        # or llama, or mistral

**3. When the session dies, and it will.** Open a new one, redo step 2, run the
same line. Every channel skips what is already in Drive. Expect two to three
sessions per model; nothing is ever recomputed.

To rerun a single channel: `!python run_p3_all.py qwen --only=e1a`.

**4. What to send back.** The three CSVs from Drive, and the console text from
the first `====` line onwards. Never the model-loading logs.

**5. Analysis, offline.** With `e1_conditions.json` in the same directory:

    python analyse_e1_accuracy.py   e1_accuracy_*.csv
    python analyse_e1_continuous.py e1_traits_*.csv
    python analyse_e1_continuous.py e1_choice_*.csv

Channels B and C are analysed in separate calls because their units differ.

---

## Reading the output

In this order, and the order matters.

1. **The floor check.** Two socially neutral details stacked the same way. If it
   moves like the social contrasts do, stop: the instrument is counting details,
   not reading them, and nothing below can be believed.
2. **Main effects.** A signal with no main effect cannot have its interaction
   interpreted.
3. **Interactions, with the MDE column read BEFORE the p column.** Where the
   estimate is smaller than the MDE, the answer is "not visible from here", not
   "no interaction".
4. **Order.** If the two slot orders of a pair disagree in sign, the result
   belongs to the slot and is reported that way rather than averaged.
5. **Refusal against signal count.**

## Registered expectations

Written before the first run, in `docs/paper-3-experiment-menu.md`.

- **Supports the thesis:** form-by-form super-additive and clearing the wrapper
  spread, stated-by-stated not distinguishable from zero, in at least two of
  three models on at least two of three channels.
- **Refutes it:** stated-by-stated super-additive, or form-by-form absent, in
  two of three models on channels powered for it.
- **A null that is still the paper:** no interaction of either type anywhere,
  with the floor showing that clause count explains what movement there is.
  Then stacking does not happen at this scale on these instruments, the minimum
  detectable effect is stated, and the compounding claim is challenged on
  evidence.

---

## Floor extension and channel A2, 2026-09-10

Added after the first analysis, run from `colab/run_floor_<model>.ipynb`.

| channel | script | outcome |
|---|---|---|
| A2 answer log probability | `run_e1_answer_logprob.py` | log P(correct answer) on channel A's prompts, 23 conditions x 600 tasks |
| floors 2 and 3 | the existing runners with `e1_floor_conditions.json` | dog x guitar and tea x history books, on channels A, B, C and C2 |

Analysis, from `scripts/`:

    python analyse_e1_floors.py
    python analyse_e1_answer_logprob.py ../outputs/*/e1_answer_logprob_*.csv --accuracy ../outputs/*/e1_accuracy_*.csv

Read `analyse_e1_floors.py` first. A channel is clean on a model only if none of
its three floors rejects under Holm within that model and channel, on either of
two tests (signed-rank, and sign-flip on the mean), and on the task channels
also within three difficulty strata.

Result, 2026-09-11: A2 is valid (AUC 0.90 to 0.995) but identifies no
interaction: floors contaminated on llama and mistral, clean in aggregate only on
qwen. Every interaction it does register takes the sign opposite to the single
effects, neutral and social alike. See `results/analysis_e1_answer_logprob.txt`.

Validation: `validate_e1_floor_extension.py`.
