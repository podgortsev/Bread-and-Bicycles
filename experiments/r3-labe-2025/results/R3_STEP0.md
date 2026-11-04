# R3 Step 0: LABE, their Table 1 from their released generations

Run 2026-09-13, `scripts/r3_step0.py`, commit e8cc42d of their repository.
Rule: Preregistration Addendum 5, committed before any output was read. Output:
`r3_step0_summary.txt`.

## Verdict: FAIL, instrument not shown equivalent

| criterion | rule | ours | |
|---|---|---|---|
| Spearman over their 18 Table 1 values | at least 0.6 | 0.688 | pass |
| median absolute relative difference | at most 0.25 | 0.286 | **fail** |

Under the preregistered rule, no floor arm is run for R3 and no conclusion about
their claim is drawn from our instrument.

## What we ran, and the deviations

- **Classifier weights.** The Google Drive folder linked from their README holds
  configuration and trainer state for three classifiers, but no model weights.
  The three classifiers were retrained with their `fine_tune_bert.py` on their
  released LAC data. Validation accuracy: ours 0.944 / 0.933 / 0.941, theirs at
  epoch 9 0.933 / 0.933 / 0.917.
- Run on the local GPU (RTX A3000 Laptop), not CPU as Addendum 5 says. The code
  edits are the ones Addendum 5 declares (device instead of hard-coded cuda,
  no padding to 512, checkpoint only at step 4,500, the race branch taken).
- Their generations are the files in the repository at its last commit (June
  2024). Table 1 is from the ACL 2025 version. Whether the paper's numbers were
  computed on these exact files cannot be checked from outside.

## Exploratory, not a registered test: the instrument moves with the classifier seed

The three retrained classifiers have near-identical validation accuracy, yet the
variances they produce on the same generations differ by up to a factor of four:

| model, task | gender var (c0 / c1 / c2) | race var | intersectional var |
|---|---|---|---|
| Llama3, bios | 24.0 / 47.7 / 31.5 | 88.1 / 38.9 / 40.9 | 89.9 / 62.4 / 57.2 |
| Llama3, professor | 59.0 / 67.4 / 72.6 | 154.2 / 96.0 / 93.8 | 183.1 / 138.8 / 129.7 |
| Llama3, letters | 16.0 / 30.2 / 38.3 | 38.0 / 27.1 / 32.1 | 42.2 / 42.0 / 52.0 |
| Mistral, bios | 34.6 / 60.4 / 56.7 | 92.7 / 27.2 / 21.1 | 99.6 / 59.6 / 54.1 |
| Mistral, professor | 127.3 / 70.6 / 69.3 | 129.7 / 75.0 / 89.4 | 193.4 / 110.2 / 118.8 |
| Mistral, letters | 35.8 / 48.4 / 75.1 | 19.3 / 12.7 / 9.3 | 37.9 / 40.9 / 54.0 |

Two readings follow, both labelled exploratory:

1. A 0.25 relative tolerance was tighter than the retraining noise of this
   instrument. The step 0 failure cannot be separated from that noise without
   their original weights.
2. The ordering the paper reports (intersectional largest) appears in 14 of the
   18 per-classifier rows. That is what the metric produces when an 8-cell
   variance contains both the gender and the race spread, so the ordering alone
   says little about intersection. This is the question the floor arm was built
   for, and it stays unanswered until step 0 passes.

## Next

Ask the authors for the classifier weights (draft 3 in `correspondence/drafts.md`).
With their weights, step 0 is rerun under the same rule, unchanged.
