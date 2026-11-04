# R6, Beck et al. (EACL 2024): step 0, Diaz

Registered in `PREREGISTRATION.md`, Addendum 11 (2026-09-22), before any run.
Reading A is the registered primary. Reading B is the registered fallback and is
running; this document will be extended, not rewritten, when it lands.

**Their claim.** "a combination of all sociodemographic attributes leads to the
most substantial changes for most of the datasets (64%)" (their Table 8).

**What we ran.** Their `sd_prompting.py` and `util.py` at commit 8a9b40af,
unchanged, driven by `r6_run.py` over six models x seven conditions, prompt
format 0, on the 1,000-instance Diaz sample rebuilt with their notebook's cells
27 to 35 and 46 (`r6_build_diaz.py`). 42 runs, all completed.

**Statistic, as registered.** Per condition, the share of (instance, profile)
prompts whose prediction differs from the no-profile prediction for the same
instance.

---

## Reading A: FAIL

Pass required their printed winner to be ours, and our "All" share to be within
5 percentage points of theirs, on at least 4 of the 6 models. **3 of 6 match.**

| model | G | R | E | A | PA | All | ours: top, All | theirs: top, All | |
|---|---|---|---|---|---|---|---|---|---|
| Flan-T5 80M | 0.7 | 1.3 | 0.7 | 0.8 | 2.8 | 3.7 | All, 3.72 | All, 2.30 | match |
| Flan-T5 250M | 35.5 | 16.1 | 28.7 | 24.8 | 18.5 | 33.4 | **G**, 33.40 | All, 28.60 | no |
| Flan-T5 780M | 17.5 | 16.7 | 20.9 | 16.3 | 14.4 | 42.4 | All, 42.38 | All, 38.20 | match |
| Tk-Instruct 80M | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | **R**, 0.00 | All, 0.00 | no |
| Tk-Instruct 250M | 9.2 | 9.2 | 7.3 | 2.1 | 7.0 | 27.9 | All, 27.92 | All, 23.30 | match |
| Tk-Instruct 780M | 4.2 | 2.5 | 3.5 | 2.9 | 2.8 | 13.0 | All, 13.02 | All, **7.00** | no |

**Verdict: FAIL on the registered reading A.** Reported as a failure, as
registered, and not rescued by anything below.

## The three misses, described and not argued with

Stated because a reader should see what kind of failure this is, not to change
the verdict.

1. **Tk-Instruct 80M is a degenerate cell.** Every condition changes between 0.0
   and 0.1 percent of labels, and their own printed value is 0.00. Our "All" is
   0.00, matching exactly. The model fails only the "their winner is our winner"
   clause, because the argmax over six near-zero shares is decided by a single
   label change. The registered rule has no exception for a degenerate argmax and
   none is being invented now.
2. **Flan-T5 250M misses on the winner, not the size.** Our All is 33.40 against
   their 28.60, a difference of 4.80 and so inside the 5-point band; gender beats
   it by 2.1 points. On this model the Diaz column of their Table 8 is not
   reproduced in its ordering.
3. **Tk-Instruct 780M is a genuine size miss.** 13.02 against their 7.00, 6.02
   outside the band, with All clearly the winner in both.

So the direction of their claim, that combining all attributes moves predictions
most, reproduces on four of the six models. What does not reproduce is the
printed table, either in ordering or in magnitude, on three of six.

## The collection was lost and rebuilt, and reading A reproduces exactly

On 2026-10-15 the working directory holding this collection was deleted. The
environment was rebuilt from nothing at a different path, from the versions the
authors pin, and the Diaz sample was rebuilt from their notebook code to the
same counts (14,071 instances, 1,000 sampled, 4,269 annotator profiles).

Format 0 was recollected in full on 2026-10-22. **All 36 cells of the table
above reproduce to the second decimal, and the verdict is the same: 3 of 6,
FAIL.** Since the environment, the virtual environment, the paths and the
download of the source data were all new, this is an independent reproduction of
reading A rather than a rerun of the same state, and it is the reason the
recollected reading B can be compared with the reading A written here.

The numbers can be recomputed from this repository alone:

    cd scripts
    python r6_step0.py --predictions ../outputs/r6_predictions.csv.gz --formats 0

which reads the distilled predictions in `outputs/` and prints the table above.
`outputs/r6_manifest.txt` gives the row count and the SHA-256 of every raw file
those predictions were built from.

## Reading B, registered and running

Addendum 11: "the same with the share averaged over their three prompt formats".
Formats 1 and 2 are collecting now, 84 further runs. When they land, the
comparison is recomputed on the average of the three formats and the verdict is
reported for both readings side by side.

**If both readings fail,** the registered consequence stands: "instrument not
shown equivalent", and no Diaz floor arm is built on it.

## A bug this check caught

`r6_step0.py` reported every model as incomplete on its first run. Its filename
builder put the attribute after the model, while their `build_output_name` puts
it before. The same fault was in `r6_run.py`, where it silently defeated the
resume check and made every restart redo the finished runs. Both are fixed and
the numbers above come from the corrected reader. No output was lost: their
script writes each file whole.
