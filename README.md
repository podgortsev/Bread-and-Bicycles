# Bread and Bicycles

**What Happens to an Interaction Claim When the Second Detail Means Nothing**

Paper 3 of a multi-paper programme on what systems cost the people they were not
built around. Paper 1 is *The Average-Fit Loss*, Paper 2 is *Single Human-AI
Fit*. This paper absorbs what was planned as Paper 8, *The Floor Test*.

The paper is `paper/paper.pdf`. Its sources, the data, the scripts that produce
every number, the two preregistrations and the offline validation suites are all
here.

---

## The question and the answer

When a person tells a model two things about themselves instead of one, is the
penalty more than the sum of the parts?

Before that can be answered, the instrument has to be shown to measure people
rather than words. Alongside every audited pair we stack pairs of socially
neutral details built exactly the same way: cycling to work with baking bread,
walking a dog with learning guitar, making tea with reading history books. Their
interaction should be zero.

### Part I, on instruments we built

| instrument | qwen | llama | mistral |
|---|---|---|---|
| A, does the model solve the person's task | clean | clean | clean |
| A2, probability of the correct answer | clean in aggregate only | contaminated | contaminated |
| B, trait log probability | contaminated | contaminated | contaminated |
| C2, head-to-head choice | contaminated | contaminated | contaminated |

Four of five instruments report significant interactions between two meaningless
details. On the one instrument that holds a clean zero, one interaction of
twenty-four survives correction: on llama, two stated disclosures together cost
7.2 accuracy points LESS than the sum of their separate costs. Sub-additive, the
opposite of compounding, and the size of the spread the same measurement shows
across six neutral wrappers.

### Part II, on published instruments, with their authors' code

Four failing instruments could be a fact about our implementation. So the same
check was run on published work, under a second preregistration, reproducing one
published number before adding anything.

| target | step 0 | the floor |
|---|---|---|
| Kirk et al., NeurIPS 2021 | pass | religion and ethnicity explained by it; sexuality and politics clear it by 0.002 and 0.004 |
| WinoIdentity, COLM 2025 | pass | 11 of 20 cells above it; a neutral marker pair reaches 0.350 against their 20 percent threshold |

Kirk et al.'s statistic has a neutral value of 0.22 to 0.25 against a nominal
0.05, so most of the reported "one third of regressions" is what the pipeline
produces for *hungry* and *sleepy*. WinoIdentity's doubly-disadvantaged pattern
is at its strongest with **no identity marker at all**, and adding one mostly
reduces it.

Neither published result is shown to be false. Neither published statistic
separates its claim from what the same instrument reports for a detail that
means nothing.

### Part III, what could not be rerun

Four of the seven targets, all chosen because they released data or code, could
not be run from what was published: missing classifier weights, an unstated
input to a preprocessing tool, an unrecorded runtime version, and an
undocumented grouping. Reported as five specific items, not as a rate.

Details in `experiments/*/results/RESULT_*.md` and `experiments/r*/results/`.

## Layout

    paper/                          paper.tex, three input sections, paper.pdf,
                                    and the 98 floor tests of Appendix A as CSV
    PREREGISTRATION.md              Part I
    PREREGISTRATION-PART2.md        Part II and III, thirteen dated addenda
    VALIDATION.md                   what is checked, what the checks caught
    validation/                     the validator logs, as run on the final code
    experiments/
        shared/                     used by more than one experiment
        e1-type-x-count/            signal TYPE crossed with COUNT. The core
        e1w-wrappers/               does it survive six neutral wrappers
        e2-depth-curve/             does the penalty grow with depth
        e2b-legibility/             is the published effect legibility
        e3-offsetting/              does a credential buy back a disclosure
        e5-attribution/             which signal moved it
        r1-kirk-2021/               rerun, NeurIPS 2021
        r2-winoidentity-2025/       rerun, COLM 2025
        r3-labe-2025/               not runnable: classifier weights
        r4-neutral-not-unbiased-2025/  fidelity missed by 0.0006
        r5-wilson-caliskan-2024/    not runnable: preprocessing input
        r6-beck-2024/               reproduction close but not exact
        r7-ableist-2025/            reproduced under a reconstructed grouping

Every experiment folder has the same shape: `README.md`, `scripts/`,
`outputs/<model>/`, `results/`.

## Reproducing it

Analysis needs no GPU and no network:

    cd experiments/e1-type-x-count/scripts
    python analyse_e1_floors.py
    python analyse_e1_accuracy.py ../outputs/*/e1_accuracy_*.csv

Collection needs a GPU. `experiments/shared/run_p3_all.py` loads one model and
runs every Part I collection step against it; `run_p3_floor.py` adds the floor
extension and channel A2. Both are resumable. 4-bit on a single free-tier T4 is
enough. Part II's reruns each have their own runner in `experiments/r*/scripts/`
and use the original authors' code, cloned at the commit named in the
preregistration.

The validation suites run without a GPU: Part I, 345 checks; Part II's
validators run beside them. `check_provenance.py` proves nothing inherited from
Paper 2 drifted, and `reverify_findings.py` re-derives every headline number
independently of the analysers.

## The rule that governs this repository

No script is handed over before its statistics have been exercised offline on
synthetic data with a known answer, including a null that must NOT be detected,
and no guard is trusted before a synthetic world reproducing the exact failure
it was added for has been built and shown to trigger it. `VALIDATION.md` lists
the ten things that rule caught, each of which changed a number or a conclusion.

For Part II a second rule applies: reproduce one published number with the
authors' own code before adding anything, under a reading fixed in writing
before the output is read. A target that fails that gate gets no floor arm and
no reinterpretation, and three of the seven failed it.

## Related tool

The floor check is released separately as **Interaction Floor Test**, a
standard-library Python tool that takes a CSV of four-cell outcomes and answers
one question: does this audit measure people, or words. On these data it reaches
the same verdicts as the analysis here, through a different stratification.
