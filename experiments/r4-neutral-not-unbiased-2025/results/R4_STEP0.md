# R4 Step 0: Neutral Is Not Unbiased, their numbers from their data

Run 2026-09-13 on CPU, `scripts/r4_step0.py`, against commit
6d1df10eeb4b0fc79eb3d18ba6bd99d9600d8bfa. The first code cell of their
`CounterAnalysisMistral.ipynb` and `CounterAnalysisPhi4.ipynb` was executed
unchanged on their released answers. Output: `r4_step0_summary.txt`.
Rule: Preregistration Addendum 5, fixed and committed before this output was
read.

| quantity (paper p. 8) | paper | ours | rule | |
|---|---|---|---|---|
| Mistral-Small 3.1 mean similarity | 0.88 | 0.8764 | rounds to 0.88 | pass |
| Mistral-Small 3.1 negative / positive | 1,268 / 1,232 | 1,268 / 1,232 | within 25 | pass |
| Phi-4 mean similarity | 0.85 | 0.8488 | rounds to 0.85 | pass |
| Phi-4 negative / positive | 1,289 / 1,190 | 1,289 / 1,190 (21 zero) | within 25 | pass |
| Phi-4 mean sentiment shift | -0.0045 | -0.0045 | within 0.001 | pass |

**Verdict: PASS.** The counts match exactly, so the instrument (MiniLM
embeddings and TextBlob polarity, current library versions) reproduces
theirs on their data.

What this does not show: that a new generation through Ollama reproduces their
answers. Generation runs at temperature 0.5 and each prompt was sampled once,
so only distributions can be compared, and that comparison belongs in the
addendum that registers the neutral arms.

Per-pair numbers are in the summary file and are not interpreted here: no
neutral arm exists yet.
