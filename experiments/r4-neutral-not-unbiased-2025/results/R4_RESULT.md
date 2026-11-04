# R4 result: Neutral Is Not Unbiased (EMNLP Findings 2025) against the floor

> **Redacted 2026-10-09.** Statements that rested on a private reply are removed
> from this document, because no permission to cite a personal communication was
> granted. Dates, numbers and verdicts are unchanged, and nothing here now
> depends on anything but the published artifacts and our own runs.

Phi-4 through Ollama (`phi4:latest`, digest ac896e5b..., the digest registered on
2026-09-13), their scenarios, names, questions, instruction, prompt layout and
options; six arms, 100 scenarios, 5 questions, 3,000 answers, generated in Colab
on 2026-09-17 by `scripts/r4_generate.py`. Analysed by `scripts/r4_analyse.py`,
written and validated before any answer existed. Rule: Addendum 6.
Console: `results/r4_floor_console.txt`. Answers: `outputs/answers_phi4.json`.

## Fidelity of the new generation: FAIL

| step | ours | theirs | rule |
|---|---|---|---|
| original -> gender | 0.8673 | 0.8467 | within 0.02: **fail by 0.0006** |
| gender -> gender_age | 0.8573 | 0.8398 | within 0.02: pass |

Under Addendum 6 this means "new generation not shown equivalent", and **no
conclusion about their claim follows from this run**. The rule was fixed before
the data existed and is not reinterpreted now that the miss is small.

Our answers are more similar to each other than theirs are, on both steps, in the
same direction and by almost the same amount (+0.021 and +0.018). The likely
cause is the generation stack: their answers were produced on an earlier Ollama
version, ours on 0.34.0 with four parallel requests, and the default context
length changed between those versions. Nothing about that can be checked from
outside their run.

## What the run does show, about our own generation

Everything below is a statement about answers generated here, not about their
published numbers.

| comparison | mean d | p signed-rank | p sign-flip |
|---|---|---|---|
| age step minus neutral step n1 (hungry / sleepy) | -0.0031 | 0.435 | 0.310 |
| age step minus neutral step n2 (thirsty / busy) | +0.0007 | 0.377 | 0.811 |
| age step minus resampling the same prompt | -0.0156 | 0.0000 | 0.0001 |
| neutral step n1 minus resampling | -0.0125 | 0.0001 | 0.0004 |
| neutral step n2 minus resampling | -0.0163 | 0.0000 | 0.0001 |

- Adding "younger" and "older" to a gendered scenario moves the sentiment of the
  answers **no more than adding "hungry" and "sleepy"**, or "thirsty" and "busy",
  in the same slot. The smallest difference this data can see is 0.006, and the
  differences are 0.003 and 0.0007.
- All three steps differ clearly from asking the same prompt twice, so the
  instrument does register that a word was added. It does not distinguish which
  word.
- The same holds for their similarity measure.

## What cannot be claimed

- Their published claim is untouched by this run, because the fidelity check
  failed.
- One model of their five. Mistral-Small 3.1, the model their quoted sentence is
  about, needs more memory than the free and local GPUs have.
- Their own step sizes (-0.0043 for Phi-4) are below the resolution of 500 items
  (0.006), as Addendum 6 stated before the data existed.

## Next

Fidelity could be retried on a generation stack closer to theirs, but the
version of the local model server they used is not recorded in the release and
is not recoverable from it. R4 therefore stays as reported.

## Addendum 9 diagnostic: one request at a time does not close the gap

Run 2026-09-21 (`scripts/r4_diagnostic.py`, console
`r4_diagnostic_console.txt`; the 300 serial answers in
`outputs/answers_phi4_serial30.json`). Serial generation at the default context
length was registered as a diagnostic in Addendum 9 and run as registered.

Step-one similarity (original to gender) on the same first 30 scenarios:

| answers | similarity | difference from theirs |
|---|---|---|
| theirs | 0.8583 | |
| ours, four parallel requests | 0.8680 | +0.0097 |
| ours, one request at a time | 0.8721 | +0.0137 |

Serial generation is not closer to theirs than parallel; it is slightly further.
The parallel-requests explanation is ruled out. Both subsets are within 0.02 on
these 30 scenarios, so the rule's condition ("serial within, parallel not") is
not met, and **R4 stays as reported: fidelity failed by 0.0006 on the full run,
no conclusion about their claim.**

What remains is a small, consistent difference in the same direction (our
answers about 0.01 more similar to each other), which the Ollama version is the
only known candidate for; the authors cannot recover it.