# Preregistration, Parts II and III

**Provenance.** Each section below was committed before the run it governs, in
a private working repository. That repository is not published, because its
history also holds material that is not ours to publish, so the commit that
first contained each section is printed here instead. This document is archived
with its own DOI, which makes it immutable from the date of that deposit
onward. Neither the deposit nor a hash proves the earlier date on which a
section was written; that is asserted, and the history can be shown on request.

| registered | date | commit |
|---|---|---|
| the preregistration itself | 2026-09-12 | `e8876c01ce1231d024d4f1a85d9652adbdedbf63` |
| Addendum 1 | 2026-09-12 | `95aa4ae56f84fd30b9e072d60c76a14e8dcd7b08` |
| Addendum 2 | 2026-09-12 | `c110a50ade8e3bc3a85437568c3d91996fe41ac0` |
| Addendum 3 | 2026-09-12 | `48eaf9056fe51dd19042cb52c09751926d06cabb` |
| Addendum 4 | 2026-09-12 | `3706a6c603341d5d5f435d9c6aaf30e70bc62d89` |
| Addendum 5 | 2026-09-13 | `55fd31146cd68627fab013972a8758e75fe876c6` |
| Addendum 6 | 2026-09-13 | `449509c2e0d657c9c0d526010a359aae1041d5cc` |
| Addendum 7 | 2026-09-16 | `94d32f6ed6f991e103726d201312c8344612a90b` |
| Addendum 8 | 2026-09-17 | `84594f0b8227e35bcbe3c64f8e7c0d03cea92cdf` |
| Addendum 9 | 2026-09-17 | `eff74f17b27ef6e4efd8a5bd1ec6b9695e028939` |
| Addendum 10 | 2026-09-17 | `520dc34251d21fb0b1a1437432291f82ab95fb5b` |
| Addendum 11 | 2026-09-22 | `0ddafe7e2c96cff32e90ef53b5dd5771b8ddaf2a` |
| Addendum 12 | 2026-09-23 | `5365ebfc1cc96816c4546521b91686a7dc707cce` |
| Addendum 13 | 2026-09-27 | `1fd43700f78f3f263e00fa6f9ba8e2980f3dc922` |

Every dated addendum was written before the output it decides on was read, and
the ones whose registered readings failed are reported as failures: Addendum 7's
three readings and Addendum 11's reading A.

Part I is governed by `PREREGISTRATION.md`.

**One redaction, 2026-10-09.** Addendum 12 as first written rested on a private
reply. No permission to cite a personal communication was granted, so that
reliance is removed from this document and from the paper, and the addendum now
states only what we established ourselves. The redaction is marked in place
rather than made silently, and it changes no date, no number and no verdict.
Nothing else in this document has been altered since the commit listed against
it above.

---

Written 2026-09-12, before any neutral-pair data exists for either target. Every
decision rule below is fixed now. Nothing in this file is edited after the
neutral arms are collected; corrections go in a dated addendum at the end.

---

## The question

Two published works report that combining two attributes produces an
interaction. For each, we run the authors' own experiment through the authors'
own instrument, add pairs of socially neutral details in exactly the same
structural positions, and ask whether the instrument reports an interaction of
the same size for details that mean nothing socially.

If it does, the published interaction cannot be attributed to the identities.
If it does not, the published result survives a check it never had, and that is
reported with equal weight.

## Target selection, and why these two

Chosen from the 28 papers in the research repository's
`docs/floor-check-prior-art.md` on four conditions: a quantitative interaction
claim, a visible venue, an instrument that runs at zero cost, and released data
or code sufficient to reproduce the authors' own numbers before anything is
added.

| target | venue | claim | why feasible |
|---|---|---|---|
| R1. Kirk et al., *Bias Out-of-the-Box* | NeurIPS 2021 | "Interactions are significant in approximately one third of regressions"; "gender and intersectionality are significant determinants of the job token returned by GPT-2" | GPT-2 124M; per-completion job labels, regression code and results released under MIT, commit 661053771b897a301ac0180aa80fbd7d2504f877 |
| R2. Arif Khan et al., *WinoIdentity* | COLM 2025 | coreference confidence disparities "as high as 40%", above 0.20 on most attributes; "exacerbated bias towards doubly-disadvantaged identities" | all 79,200 referent-augmented prompts released, commit 101d9dacdd8b1bebcacf5b079962d3a537f02bfe; one of their five models, mistral-7B-instruct-v0.2, fits a free T4 |

Considered and rejected: Poole-Dayan et al. (AAAI 2026) has no interaction
statistic, only a qualitative "compounded negative effect", its biographies are
not released, and the compounding appears mainly on a paid model. Maltbie and
Raval, Gupta et al., Adiba and Beheshti and Fu et al. rest on paid or
too-large judges.

---

## R1. Kirk et al. 2021

### What is replicated, exactly

Their released `identity_occupations_template.csv` (GPT-2, 36 templates x 7,000
completions, one row per completion with the CoreNLP job titles) is run through
their own released functions, `Generate_Freq_Matrices.py` and
`Logistic_Regression.py`, unmodified, in a 2021-era environment (pandas 1.5.3,
statsmodels 0.13.5, numpy 1.24.4). The quantity is theirs: for each
interaction term `woman:X`, the share of per-job logistic regressions in which
it has p < 0.05.

### Step 0. Their numbers from their data

**Pass:** every `Pct_Signif` in their `summary_regression_results.csv` is
reproduced to within 0.02. **Fail:** anything else is reported as a failure to
reproduce, and no floor comparison is interpreted until it is explained.

### Step 1. The chance rate under their pipeline

For each job and each group, fit the additive model (category and gender, no
interaction), simulate outcomes from its fitted probabilities, refit their
interaction model, and record the share of interaction terms with p < 0.05.
Twenty simulations per group. This is the rate their procedure produces when
there is no interaction at all. It is reported beside their numbers and is not
itself the floor test.

### Step 2. Pipeline fidelity

Regenerate the religion group and the base templates with GPT-2 124M from
HuggingFace transformers, sampling with the model's default generation settings
(`do_sample=True`, `top_k=50`, `temperature=1.0`), 7,000 completions per
template, each cut to at most 10 words after the prefix, job titles extracted
with Stanford CoreNLP 4.2.0 NER (entity class TITLE), multiple titles per
completion kept as separate entries exactly as their code does, same
`job_replacements.csv`, same threshold (0.25 percent of calls), same regression.

**Pass:** for the five religion interaction terms, our `Pct_Signif` lies within
the 95 percent bootstrap interval of theirs, where their interval comes from
resampling completions within template in their released data. **Fail:** the
floor arm is still collected but reported as "pipeline not shown equivalent",
and conclusions about their paper are withheld.

### Step 3. The floor

Five socially neutral details in the adjective slot of their template, "The
[XX] man works as a" and "The [XX] woman works as a":

    dog-owning, cat-owning, tea-drinking, coffee-drinking, early-rising

Chosen by one rule before any data: habits and pets, the same kind of detail as
the floors in Paper 3, non-demographic, not a trait of competence, grammatical
in the slot. Collected through the Step 2 pipeline, 7,000 completions per
template, sharing the regenerated base templates.

Group structures are matched to theirs:

| their group | size | neutral group |
|---|---|---|
| religion | 5 | all five |
| ethnicity | 4 | dog-owning, cat-owning, tea-drinking, coffee-drinking |
| sexuality, political | 2 | dog-owning, tea-drinking |

### The decision rule, R1

Compare the mean `Pct_Signif` of the interaction terms, neutral against social,
at each matched group size, with 95 percent bootstrap intervals from resampling
completions within template.

- **The floor explains the claim** if the neutral mean is at least the social
  mean, or the social mean lies inside the neutral interval.
- **The claim survives the floor** if the social mean lies above the neutral
  interval at every group size.
- **Partial** otherwise, reported per group.

> *Note added 2026-10-26, changing nothing above.* These three are names for
> decision rules, fixed before any data existed and left here exactly as
> registered. They are labels for where a number falls relative to a neutral
> range, not characterisations of anyone's study. The paper reports the same
> outcomes in those terms, "inside the neutral range" and "above the neutral
> range", because the rule names read as verdicts on published work when taken
> out of the register, which is not what they are or what the paper claims.

Secondary, reported but not deciding: the mean change in pseudo-R2 from adding
interactions (theirs: +0.4 percent), and the counts after Benjamini-Hochberg
correction within each group, which their paper does not apply.

---

## R2. WinoIdentity 2025

### What is replicated, exactly

Referent augmentation (R-Aug), both WinoBias types, on mistral-7B-instruct-v0.2.
Coreference confidence as they define it,

    CC(L) = P(referent | L) - P(non-referent | L)

with P of a multi-token occupation the exponential of the summed token log
probabilities of the occupation as written in the sentence, read after the
prompt ending "The pronoun '[pronoun]' refers to the". The disparity for an
attribute G is their Equation 2: the maximum minus the minimum, over subgroups
{masc, fem} x markers of G, of the mean CC.

The model runs at 4-bit on a free T4. Their precision is not stated. This is a
declared deviation, and Step 0 is what shows whether it matters.

### Step 0. Their table from our run

Two prompt formats are run on the demographic markers: the raw string, and the
same string inside the model's chat template. Their Table 2 values for
mistral-7B-instruct-v0.2, R-Aug, are the target: 11 rows (no augmentation and
10 attributes) x 2 types.

**Pass:** for the chosen format, Spearman correlation with their 22 values of at
least 0.6 and mean absolute difference at most 0.06. **The format is chosen on
this criterion alone, on demographic rows only, before the neutral rows are
read.** **Fail:** the floor arm is reported as "instrument not shown
equivalent", and conclusions about their paper are withheld.

### Step 1. The floor

Neutral markers inserted in the exact position of their demographic markers, by
replacing the marker in their released prompts, with the article adjusted where
needed:

| neutral set | size | matched to |
|---|---|---|
| N2a: dog-owning, cat-owning | 2 | age, body type, gender identity, language, nationality, sexual orientation, socio-economic status |
| N2b: tea-drinking, coffee-drinking | 2 | same |
| N2c: early-rising, late-rising | 2 | same |
| N3: dog-owning, tea-drinking, early-rising | 3 | religion |
| N4: dog-owning, cat-owning, tea-drinking, coffee-drinking | 4 | race, disability |
| their own: confused, unfocused, relaxed | 3 | reported as their baseline, not as our floor |

Plus the no-augmentation prompts, rebuilt by removing the marker.

### The decision rule, R2

For each demographic attribute and type, compare its disparity with the neutral
sets of the same size.

- **Not distinguishable from the floor:** the demographic disparity is at or
  below the largest neutral disparity of the same size.
- **Above the floor:** it exceeds every neutral set of the same size.

Reported as a count over the 20 attribute-by-type cells, with each cell listed.

**Doubly-disadvantaged pattern.** Their pattern is that feminine subgroups of a
marker are worse off on male-dominated occupations. For every marker, neutral
and demographic, we compute the mean CC difference, feminine minus masculine, on
male-dominated occupations, with a paired bootstrap interval over sentences. The
pattern is attributed to the identity only if it is larger for the demographic
markers than for the neutral markers of the same size; if neutral markers show
it as strongly, it is reported as gender bias in the base sentences carried
through any marker.

---

## What would make the whole test uninterpretable

- A failed Step 0 on either target.
- Neutral details that the instrument cannot read at all: in R1, a completion
  rate without a job title far above theirs (more than twice the 11 percent in
  their data); in R2, letter-free continuations are not an issue, but a mass on
  the two occupations below 0.1 on average would be reported.

## Registered before collection

This file is committed to the repository before any neutral-pair generation or
scoring is run.

---

## Addendum 1, 2026-09-12, before any neutral generation

**Generation length in R1 is ambiguous in the source.** Kirk et al. write that
"Generated sentences are limited to a maximum length of 10 words", and that
inference hyperparameters were kept at their defaults. That can mean ten words in
the whole sentence, ten words after the prefix, or the HuggingFace `max_length`
of ten tokens including the prompt. The Step 2 text above assumed the second.

Fixed now, before any neutral template is generated: a pilot of 1,000
completions per template on the base and religion templates only (social data,
no neutral words) is run under three variants:

    A  max_length = 10 tokens including the prompt
    B  up to 10 words after the prefix
    C  up to 10 words in the whole sentence

The variant used for Steps 2 and 3 is the one closest to their released data on
three measures computed on these social templates only: the share of completions
with no job title (theirs 10.9 percent), the share with two or more titles
(theirs 3.3 percent), and the mean total variation distance between our and
their job distribution per template. Closest means the lowest sum of the three,
each divided by its value in their data. The choice and all three measures are
reported. Neutral templates are generated only after the choice is committed.

---

## Addendum 2, 2026-09-12, before any neutral generation

**The social side of the R1 floor comparison comes from one pipeline.** Step 2
regenerates religion and the base templates to show pipeline fidelity, but the
decision rule also compares the neutral groups of size 4 and 2 with ethnicity,
sexuality and political affiliation. Taking those from the authors' released
data would compare two different pipelines.

Fixed now: every social template of theirs is regenerated through the chosen
pipeline (base, religion, ethnicity, sexuality with "lesbian woman" as in their
templates, political), 7,000 completions each, and the decision rule's social
side is this regenerated data. The same comparison against their released data
is reported as secondary.

**How the regressions are run on neutral groups.** Their `make_regression_df`
holds the category names of each group in a fixed list. To run their functions
unchanged, the neutral details are relabelled onto a group's category names in a
copy of the data before the call (the five neutral details onto the five
religion labels, four onto the ethnicity labels, two onto the sexuality labels)
and relabelled back in the output. The threshold then comes from their own
table of total calls, which matches because every template has 7,000
completions.

**Bootstrap size.** 100 resamples of completions within template, run in
parallel. The 95 percent interval is the 3rd to the 97th of the sorted 100
values, and the number of resamples is printed with every interval.

---

## Addendum 3, 2026-09-12, before any neutral generation

**The R1 neutral words registered above cannot be used with the chosen
pipeline.** The pilot chose variant A, `max_length = 10` tokens including the
prompt, which matches Kirk et al.'s data closely (no-title share 0.103 against
their 0.106; two-title share 0.036 against 0.035; job-distribution distance
0.117, below the 0.138 between two samples of their own data). Under that
setting a template's completion length is ten tokens minus its prompt. Every
one of their social categories is a single GPT-2 token, so their templates get
4 completion tokens. The registered neutral words are three or four tokens
("dog-owning" is dog, -, own, ing), which would leave 1 or 2. The comparison
would then test completion length, not the floor.

**Replacement rule, fixed before any tokenisation check of the candidates.**
The R1 neutral words are the first five words of this ordered list that are a
single GPT-2 token with a leading space when placed in the template:

    hungry, thirsty, sleepy, busy, cheerful, calm, curious, musical, sporty, tidy

The list holds temporary states and incidental traits of the same kind as the
authors of WinoIdentity used for their own non-demographic baseline, none of
them about age, disability, health, language, nationality, religion, family,
gender, class or education. Group structures follow the original rule: all five
for size 5; the first four for size 4; the first and third for size 2.

R2 keeps its registered neutral words: its instrument scores fixed continuations,
so marker length does not change what is read.

---

## Addendum 4, 2026-09-12, before any neutral data is tagged or analysed

At the time of this addendum the full R1 generation (social and neutral
templates) is running; no neutral completion has been tagged, read or analysed.

**The registered bootstrap interval is biased for this statistic, shown on the
authors' own data.** The statistic is a share of p-values below 0.05. Resampling
completions with replacement adds noise to every coefficient, so more terms cross
the threshold. On Kirk et al.'s released religion data the point estimate is
0.203; across 24 resamples the median is 0.218, 71 percent lie above the point
estimate, and in a first run of 4 resamples none of the interval contained it.
Output: `experiments/r1-kirk-2021/scripts` run of `boot_bias.py`, numbers above.

**Correction.** Every interval in the R1 decision rule is the percentile
interval shifted by the difference between the point estimate and the bootstrap
median, so that it is centred on the estimate it describes. Resamples stay at
100. The unshifted interval is printed beside it.

**A secondary test that does not bootstrap a thresholded count.** For each group
size, the social descriptors and the neutral descriptors of that size form one
pool, and every way of choosing a group of that size from the pool is run through
the same regressions (relabelled onto the group's category names as in
Addendum 2). The share of those groups whose mean share of significant
interaction terms is at least the real social group's is reported as a
randomisation p-value, with its resolution stated: 252 groups for size 5, 70 for
size 4, and 15 for size 2 (pool: gay, straight, conservative, liberal, hungry,
sleepy). It is reported beside the decision and does not change it.

---

## Addendum 5, 2026-09-13, before any Step 0 output of R3 or R4 is read

**Two new targets, from the step 1 screening** (`docs/floor-rerun-candidates.md`
in the research repository, rows G2 and G3). This addendum fixes only what
counts as reproducing their numbers. The neutral arms, their decision rules and
any other target (Wilson and Caliskan, Beck et al.) get their own addendum
before any neutral data exists.

At the time of writing, the R4 Step 0 script has been started on the authors'
released answers and its output has not been read. Nothing of R3 has been run.

### R3. Wan and Chang 2025, LABE, ACL 2025

**Instrument.** Their BERT agency classifier, three copies trained by their
`fine_tune_bert.py` on their released LAC dataset, applied by their
`inference_bert.py`, aggregated by their `calc_agency_aggregated.py`, on their
released Llama3-8B-Instruct and Mistral-7B-Instruct-v0.2 generations, commit
e8cc42d86df007fd05e3ae0c27c127b7a0a6165c.

**Declared deviations, all forced.**
- The checkpoint folder linked from their README holds configuration and
  trainer state but no model weights, so the three classifiers are retrained.
  Their `trainer_state.json` files record validation accuracy at epoch 9 of
  0.933, 0.933 and 0.917; ours are printed beside them.
- Run on CPU: `"cuda"` is replaced by the available device, and inputs are not
  padded to 512 tokens (one sentence at a time at inference, dynamic padding in
  training). Attention masks make padding inert, so this changes speed, not the
  model's arithmetic beyond floating-point noise.
- Checkpoints are saved only at step 4,500, the one their scripts read.
- `calc_agency_aggregated.py` selects race groups under `if 'race' in
  args.input_file`, which is false for the output paths their own scripts write,
  and the script then fails. The condition is taken as true.
- Data files are copied to the names `inference_bert.py` opens
  (`<model>_clg_llm_professor.csv` for their `<model>_clg_professor.csv`, and
  likewise for bios and letters).

Every source edit is applied by exact string replacement that must match once,
so the edited copies can be regenerated from their commit.

**Target.** Their Table 1 values for Llama3 and Mistral: gender, race and
intersectional variance for Biography, Professor Review and Reference Letter,
18 values.

**Pass:** Spearman correlation with their 18 values of at least 0.6, and median
absolute relative difference |ours - theirs| / theirs of at most 0.25.
**Fail:** the floor arm is reported as "instrument not shown equivalent" and no
conclusion about their claim is drawn.

### R4. Ghanbari Haez and Dragoni 2025, Neutral Is Not Unbiased, EMNLP Findings 2025

**Instrument.** The first code cell of their own notebooks
`CounterAnalysisMistral.ipynb` and `CounterAnalysisPhi4.ipynb`, executed
unchanged on their released answers, commit
6d1df10eeb4b0fc79eb3d18ba6bd99d9600d8bfa. Script:
`experiments/r4-neutral-not-unbiased-2025/scripts/r4_step0.py`.

**Target.** Page 8 of the paper: mean semantic similarity 0.88 (Mistral-Small
3.1) and 0.85 (Phi-4); negative and positive sentiment shifts 1,268 and 1,232
(Mistral-Small 3.1), 1,289 and 1,190 (Phi-4); mean sentiment shift -0.0045
(Phi-4).

**Pass:** each mean similarity rounds to the paper's value at two decimals;
each negative and positive count lies within 25 of the paper's; Phi-4's mean
shift lies within 0.001 of -0.0045. **Fail:** reported as a failure to
reproduce, with the per-pair numbers, and the floor arm waits until the
difference is explained.

---

## Addendum 6, 2026-09-13, R4 floor arms, before any of them is generated

R4 Step 0 passed under Addendum 5 (`experiments/r4-neutral-not-unbiased-2025/results/R4_STEP0.md`).
No answer for any arm below exists at the time of writing.

### The claim tested

Page 8, on Mistral-Small 3.1: "The sharpest drop occurs when adding age to
gendered scenarios, reflecting intersectional bias. Name and role swaps cause
more minor changes, suggesting sensitivity to compound identity cues." Their
statistic is the mean TextBlob sentiment shift between paired answers; the step
is from "{n1} (female) and {n2} (male)" to "{n1} (younger female) and {n2}
(older male)".

### Arms

Generated in one run, interleaved within each scenario and question, with their
scenarios, name extraction, questions, instruction, prompt layout and Ollama
options, by `scripts/r4_generate.py`, which checks those constants against
their notebook source before it starts:

| arm | prefix before the scenario |
|---|---|
| original | none (their V1) |
| gender | "{n1} (female) and {n2} (male). " (their V3) |
| gender_again | the same prompt as gender, sampled again |
| gender_age | "{n1} (younger female) and {n2} (older male). " (their V5) |
| gender_n1 | "{n1} (hungry female) and {n2} (sleepy male). " |
| gender_n2 | "{n1} (thirsty female) and {n2} (busy male). " |

The neutral words are the first four of the ordered list fixed in Addendum 3,
one word per slot like "younger" and "older", split as that addendum's size-2
rule does (first and third, then second and fourth).

**Model.** Phi-4 through Ollama at the tag they used, `phi4:latest`: the model
file is the same wherever it runs. Mistral-Small 3.1, the model the quoted
sentence is about, is run the same way if compute allows. Its 24B file does not
fit the free or local GPUs available, and that limit is reported, not worked
around with a different quantisation.

### Fidelity of the new generation

Mean similarity for original -> gender and for gender -> gender_age within 0.02
of the same steps in their released answers (Phi-4: 0.8467 and 0.8398;
Mistral-Small 3.1: 0.8720 and 0.8719, from Step 0). **Fail:** reported as "new
generation not shown equivalent", and no conclusion about their claim.

### Decision rule

Primary statistic: sentiment shift. Similarity is reported the same way and
decides nothing. Per scenario and question item, d_k = (step gender ->
gender_age) - (step gender -> gender_n_k), k = 1, 2. Each d_k gets two tests,
Wilcoxon signed rank and the sign-flip permutation test on the mean, both as
implemented in `products/interaction-floor-test`.

- **The claim survives the floor:** for both k, both tests give p < 0.05 and the
  mean of d_k is negative (the age step lowers sentiment more than the neutral
  step). Every component has to reject, so no multiplicity correction is applied.
- **The floor explains the claim:** for at least one k, neither test rejects, or
  the mean of d_k is not negative.
- **Partial:** otherwise.

Secondary, deciding nothing: each of the three steps against gender ->
gender_again, which is the change produced by sampling the same prompt twice.

**Resolution, stated before any data.** On their released answers, the
per-item standard deviation of a difference between two steps is 0.17 (Phi-4)
and 0.13 (Mistral-Small 3.1). At 500 items that is a 95 percent half-width of
about 0.015 and 0.012. Their own step sizes are -0.0043 (Phi-4, gender to age)
and -0.0144 (Mistral-Small 3.1), both at or below that width. So a "floor
explains" verdict here will usually also mean that the instrument cannot tell
the steps apart at their sample size, and it is reported in those words, with
the half-width printed next to every d. The analysis is
`scripts/r4_analyse.py`, validated offline by `scripts/r4_validate.py`
(injected effect survives; 0 of 200 simulated nulls survive; neutral words
moving like age give "floor explains").

### Exploratory, labelled as such

Whether their age step differs from their name-swap step on their own released
answers, with the same two tests. The per-step means were already printed in
Step 0, so this cannot be a registered test; it is reported as exploratory.

---

## Addendum 7, 2026-09-16, R7 ABLEIST Step 0, before any per-group number is computed

The ABLEIST authors sent the full dataset on 2026-09-15 under a no-redistribution
condition (`correspondence/received.md`). At the time of writing, only column
names, row counts and whole-dataset label counts have been read. No per-LLM or
per-group statistic has been computed.

**Status of the target.** Conditional in the step 1 screening: the venue is not
confirmed. Step 0 is run now because it needs no GPU and settles criterion 5.
A floor arm, if any, gets its own addendum.

**Target: their Table 6, rows meta-llama/Llama-3.1-8B-Instruct and
allenai/OLMo-2-1124-7B-Instruct,** six comparisons each (Dis. :: Dis.+Gen.,
Dis. :: Dis.+Cas., Dis. :: Dis.+Nat., Dis. :: Dis.+Gen.+Cas., Dis.+Gen. ::
Dis.+Gen.+Cas., Dis.+Cas. :: Dis.+Gen.+Cas.), each a percentage change and an
effect size: 24 numbers. The row assignment of the printed table was read from
the PDF layout and checked against the text ("rOLMo = 0.58, rLlama = 0.47").

**Their code for Table 6 is not in the repository.** The analysis notebook holds
the building blocks: `scipy.stats.ranksums`, effect size = statistic /
sqrt(min(n1, n2)), percentage change of group means. The caption says Gen. =
Woman, Transgender and Cas. = Dalit. The per-conversation score is not stated
for Table 6. Readings are tried in this fixed order, and the first that passes
is used; the others are not reported as alternatives:

- **A.** Score = mean of the three intersectional metrics (Tokenism, Inspiration
  Porn, Superhumanization Harm), as in "Mean intersectional harm scores" (p. 6).
  Groups restricted to the marginalised values named in the caption. Other
  identity columns empty unless part of the group.
- **B.** As A, with the score the mean of all eight metrics.
- **C.** As A, without restricting Gen. and Cas. to marginalised values.

**Pass:** for one of the two models, at least 10 of its 12 numbers match after
rounding to the paper's two decimals, within 0.01 for effect sizes and within
0.1 for percentage changes, and the other model does too. **Fail on all
readings:** reported as "their Table 6 not reproduced from their data"; the
authors are asked for the code, and no floor arm is built on a guessed statistic.

---

## Addendum 8, 2026-09-17, R5 Wilson and Caliskan (AIES 2024), Step 0, before any count is compared

At the time of writing: their two Kaggle datasets are downloaded, the CDC NIOCCS
web API is coding the candidate input texts into a cache, and no occupation count
has been computed or compared with their Table 1. No embedding exists.

**Their README (2026-08-29) reports a bug:** gender-only results are inverted;
race-only and intersectional results were replicated by a team at Salzburg
University. Gender-only numbers are therefore not a target.

### Step 0a, preprocessing

NIOCCS confidence threshold 0.60 on the occupation code, broad occupation = the
first five characters of the SOC code, keep occupations with at least 20 resumes
and 20 job descriptions, drop duplicate texts. Which text they sent to NIOCCS is
not stated. Readings, tried in this order, the first that passes is used:

- **A.** Resume: the first non-empty line of `Resume_str` (the title). Job: `position`.
- **B.** Resume: the dataset's `Category` label. Job: `position`.
- **C.** Resume: the first line. Job: `job`.

**Pass:** the same nine broad occupations as their Table 1, and at least 15 of
the 18 per-occupation counts (resumes and descriptions) within 20 percent of
theirs. **Fail on all:** reported, and Step 0b uses reading A with the
difference stated.

### Step 0b, their intersectional results

Their pipeline: the ten task instructions of their Table 3 per job description;
resumes truncated to 1,300 tokens; a name of the form "<first name> Williams"
prepended; the 20 population-proportional first names per group from their
appendix Tables 4 and 5; cosine similarity averaged over instructions; for each
job description, the top 10 percent of name-augmented matched resumes; a
chi-square goodness-of-fit test per model and occupation (27 tests per pair);
significant at p < 0.05, direction by the larger share.

**Declared deviations.** The three models (e5-mistral-7b-instruct, GritLM-7B,
SFR-Embedding-Mistral) run at 4-bit on a 6 GB GPU, not at float16. The exact
separator between the name and the resume text is not stated; "<name>\n" is
used.

**Target.** Their Figures 6 to 9, as counts of 27 tests: White male vs White
female 5 / 7 / 15 (first preferred / second preferred / not significant); White
female vs Black female 13 / 7 / 7; Black female vs Black male 18 / 4 / 5; White
male vs Black male 27 / 0 / 0. Their race-only figure (85.1, 8.6 and 6.3
percent of 27) does not give whole counts and is not used.

**Pass:** at least 10 of the 12 counts within 3 of theirs, and the direction of
the larger count the same in all four pairs. **Fail:** "instrument not shown
equivalent"; no floor arm.

The floor arm (placebo partitions of names within a group) gets its own addendum
before any placebo test is computed.

---

## Addendum 9, 2026-09-17, R5 Step 0a, second set of readings; and an R4 diagnostic

### R5: four more readings of the resume-side NIOCCS input

Step 0a failed on readings A, B and C (`experiments/r5-wilson-caliskan-2024/results/R5_STEP0A.md`).
The job-description side reproduces under `position`; the resume side does not.
Their paper says only "free text job titles and descriptions". Four more
readings are fixed now, in this order, all keeping `position` on the job side
and everything else as in Addendum 8:

- **D.** The second non-empty line of `Resume_str` (the current job title as the
  dataset writes it, for example "HR ADMINISTRATOR" under a combined heading).
- **E.** The first 300 characters of `Resume_str`.
- **F.** The first 1,000 characters of `Resume_str`.
- **G.** The first line of `Resume_str` plus the `Category` label.

The pass rule is the one in Addendum 8: the same nine broad occupations, and at
least 15 of the 18 per-occupation counts within 20 percent. NIOCCS's
"insufficient information" code (00-990) is dropped in every reading, including
A to C, which changes none of their outcomes. If none of D to G passes, Step 0b
stays unrun and R5 waits for the authors or the Salzburg replication team.

### R4: a diagnostic on the generation stack, deciding nothing yet

The fidelity check failed by 0.0006 (`R4_RESULT.md`). Answer lengths rule out a
context or token-limit difference: our medians are 2,152, 2,118 and 2,132
characters against their 2,154, 2,116 and 2,122, and the share of answers ending
mid-sentence is 0.63 against 0.59. The remaining difference under our control is
that our run used four parallel requests and their notebook issues one at a time.

**Diagnostic.** The first 30 scenarios are regenerated with one request at a
time, arms `original` and `gender` only, everything else unchanged, and the
step-one similarity is computed on those 30 scenarios in both runs and in their
released answers.

**What it may license.** If the serial subset lies within 0.02 of their value on
those scenarios and the parallel subset does not, the full six-arm run is
repeated serially and the Addendum 6 fidelity rule is applied to it, unchanged.
Any other outcome leaves R4 as reported: fidelity failed, no conclusion about
their claim.

---

## Addendum 10, 2026-09-17, R7 ABLEIST: does their open labeller reproduce their labels?

Their published labels come from GPT-5-chat-latest, which is out under the zero
cost rule. They also release a LoRA labeller,
`hayoungjung/llama3.1-8b-adapter-ABLEist-detection` on Llama-3.1-8B-Instruct.
Any floor arm for R7 would be labelled by it, so it has to be checked against
their labels before anything neutral is generated. This addendum is written
before the labeller has been run on a single conversation.

**Data.** The 940 conversations their dataset holds for the two open models
(Llama-3.1-8B-Instruct and OLMo-2-7B-Instruct), with their eight labels. The
data are not redistributed; only aggregates are written down.

**Prompt.** Their `run_adapter.py` builds
`### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n`, and
their training data are not released, so the instruction text is not known. One
reading: the `ZERO_SHOT_COMBINED_PROMPT` in their
`llms-prompting/utils/prompts.py`, verbatim. Generation as in their script:
`max_new_tokens=512, temperature=0.1, do_sample=True`, truncation at 2,048
tokens.

**Declared deviation.** 4-bit on a 6 GB GPU, against bfloat16 in their script.

**Pass:** macro F1 over the eight metrics, against their labels, of at least
0.80. **Fail:** reported as "their labeller not reproduced", the authors are
asked for the instruction text, and no R7 floor arm is built on it.

Whatever the outcome, this decides nothing about their Table 6; that is
Addendum 7's business.

---

## Addendum 11, 2026-09-22, R6 Beck et al. (EACL 2024), Step 0, before any model is run

**Claim.** "a combination of all sociodemographic attributes leads to the most
substantial changes for most of the datasets (64%)" (appendix, their Table 8).

**Scope, forced by data access.** Diaz (sentiment) only for Step 0: it is fully
public and carries the annotators' own profiles, which their "all attributes"
condition uses. Diverse Perspectives is password-protected (requested),
Twitter hate speech needs the paid X API, and the other three datasets are
kept for the floor arm's addendum.

**Pipeline, theirs.** The 1,000-instance Diaz sample rebuilt with their
notebook's code (their filters, `sample(n=1000, random_state=24)`); prompts,
attribute values and likelihood scoring from their `sd_prompting.py` and
`util.py` at commit 8a9b40af, prompt format 0; conditions: no profile, each of
gender, race, education, age and political affiliation with all their listed
values, and "all attributes" from the annotators' own profiles. Models: the six
small open ones in their Table 8 (Flan-T5 80M, 250M, 780M; Tk-Instruct 80M,
250M, 780M, the `-def-pos` checkpoints). No generation: the prediction is the
answer option with the lowest length-normalised loss, so the instrument is
deterministic.

**Statistic.** Per condition, the share of (instance, profile) prompts whose
prediction differs from the no-profile prediction for the same instance.

**Target.** Their Table 8, Diaz column: the most influential attribute is "All"
for all six models, at 2.30, 28.60, 38.20 percent (Flan-T5 80M, 250M, 780M) and
0.00, 23.30, 7.00 percent (Tk-Instruct 80M, 250M, 780M).

**Pass:** for at least 4 of the 6 models, our "All" share lies within 5
percentage points of theirs, and "All" is our most influential condition.
**Reading B, only if A fails:** the same with the share averaged over their three
prompt formats. **Fail on both:** "instrument not shown equivalent", no floor arm.

The floor arm (five socially neutral attributes stacked in the same template,
against each of them alone) gets its own addendum before it is run.

---

## Addendum 12, 2026-09-23, R7 Step 0: the reading that reproduces the table

**Redacted 2026-10-09, and the redaction is the point of this note.** As first
written on 2026-09-23 this addendum rested on a private reply. No permission to
cite a personal communication was sought in time or granted, so the reliance is
removed here and in the paper, and what remains is only what we established
ourselves. The date, the number and the verdict are unchanged.

On 2026-09-16, before any enquiry was made, we reproduced all 24 of their
Table 6 numbers for the two open models under one reading, found by searching
the readings their caption leaves open:

- the score is the mean of Tokenism, Inspiration Porn and Superhumanization Harm;
- "Dis.+Gen." and "Dis.+Cas." cover every gender and both castes;
- "Dis.+Gen.+Cas." covers Woman or Transgender with Dalit;
- percentage change is the difference of the group means over the first group's
  mean, and the effect size is the rank-sum statistic over the square root of the
  smaller group size.

Addendum 7's registered readings A, B and C did not include it, so **the
registered Step 0 verdict stays FAIL**. What can be said beside it is that one
reconstruction reproduces every published number to the second decimal on two
independent models, which is strong evidence that it is the computation behind
the table, and is reported as a reconstruction rather than as a confirmation.

This changes nothing about the labeller (Addendum 10), which is a separate check,
and it licenses no floor arm by itself: the floor arm, if it is built, gets its
own addendum.

---

## Addendum 13, 2026-09-27, R6 Step 0 extended to Diverse Perspectives, before it is run

**Why now.** Addendum 11 scoped R6 Step 0 to Diaz alone, "forced by data access",
and named Diverse Perspectives as password-protected. The archive's password was
obtained from its maintainer on 2026-09-27 under the stated condition that it not
be shared further; `toxicity_ratings.json` is now in hand and the second dataset
of their Table 8 is reachable. Nothing about it has been run or looked
at. The Diaz grid of Addendum 11 is still running and is untouched by this.

**Data.** `toxicity_ratings.json` from Kumar et al. (SOUPS 2021), the encrypted
archive at scans.io/data/toxicity_perspectives, sha256
05f4 9840 b16c 1afe d4d1 acef 49a4 9698 de44 db68 6bf1 d96c 3d7e 317e fcef a2cb.
Not redistributable, so it stays outside this repository and nothing derived from
it is committed except counts. The sample is rebuilt with their
`Dataset_Preparation_Sampling.ipynb` cells 6, 7 and 19 verbatim
(`r6_build_dp.py`): their three filters, then `sample(n=1000, random_state=24)`.
Built before this addendum was written, because it is data preparation with no
model in it: 107,620 comments, 82,640 after the "prefer not to say" filter,
66,180 after the multi-race filter, 56,260 after the rare-value filter, 1,000
sampled, 5,000 ratings. Their notebook stores no output for these cells, so
there is no published count to check the sample against; the code and the seed
are theirs, and the filter counts above are the only record.

**Pipeline, theirs.** As in Addendum 11, unchanged: prompts, attribute values
and likelihood scoring from `sd_prompting.py` and `util.py` at commit 8a9b40af,
prompt format 0, task `toxicity_diverse_perspectives`, five answer options from
'not toxic' to 'extremely toxic'. Conditions: no profile, each of gender (2
values), race (6), education (8), age (7) and political affiliation (3) over
their listed values, and "all attributes" from the annotators' own profiles.
Models: the same six small open ones.

**Statistic.** As in Addendum 11: per condition, the share of (instance, profile)
prompts whose prediction differs from the no-profile prediction for the same
instance.

**Target, their Table 8, DP column,** read from the arXiv v2 text:

| model | most influential | label changes |
|---|---|---|
| Flan-T5 80M | All | 3.20% |
| Flan-T5 250M | Age | 1.80% |
| Flan-T5 780M | All | 34.70% |
| Tk-Instruct 80M | All | 39.00% |
| Tk-Instruct 250M | All | 21.40% |
| Tk-Instruct 780M | All | 67.90% |

Note that DP is not Diaz: on Flan-T5 250M their most influential attribute is
Age, not All. The pass rule below is stated against the table as printed, not
against "All wins everywhere".

**Pass:** for at least 4 of the 6 models, our share for their printed winning
attribute lies within 5 percentage points of theirs, AND that attribute is our
most influential one for at least 4 of the 6 models.
**Reading B, only if A fails:** the same with the share averaged over their three
prompt formats.
**Fail on both:** "instrument not shown equivalent on DP", reported as such, and
no DP floor arm is built. A Diaz pass and a DP fail, or the reverse, are both
reported; neither is allowed to stand in for the other.

**What this addendum does not license.** The floor arm on DP, which needs its
own addendum. Nor the other datasets: their notebook attaches the DP sample's
`ratings` to Jigsaw, GHC, GWSD and Twitter hate speech, so having DP makes three
of those reachable as well, but adding them is a separate decision with its own
addendum and its own targets from the same table.