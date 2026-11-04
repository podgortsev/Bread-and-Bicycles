# R5 Step 0a: preprocessing against Wilson and Caliskan's Table 1

Run 2026-09-17, `scripts/r5_nioccs.py` (4,859 distinct texts coded by the CDC
NIOCCS web API, no errors) and `scripts/r5_preprocess.py`. Rule: Addendum 8,
committed before any count was compared. Console: `r5_step0a_console.txt`.

## Verdict: FAIL on all seven registered readings

Readings A to C are from Addendum 8, D to G from Addendum 9 (the second set was
registered after the first failed and before it was computed). Since Addendum 9
the NIOCCS "insufficient information" code is dropped everywhere.

| reading | resume text sent to NIOCCS | same nine occupations | counts within 20 percent (of 18) |
|---|---|---|---|
| A | first line of `Resume_str` | no | 8 |
| B | `Category` label | no | 9 |
| C | first line (job side: `job`) | no | 3 |
| D | second line of `Resume_str` | no | 8 |
| E | first 300 characters | no | 8 |
| F | first 1,000 characters | no | 8 |
| G | first line plus `Category` | no | 8 |

## What the failure shows

- **Job descriptions reproduce.** Under reading A, 8 of 9 description counts are
  within 20 percent, and three match exactly or nearly (52/52, 20/20, 97/99).
  Their job-side input was almost certainly `position`.
- **Resumes do not, under any of the seven.** The differences have the same
  shape every time: too many Misc. Managers (205 to 218 against their 112), too
  few Accountants (54 to 69 against 133), and almost no Secondary School Teachers
  (0 or 1 against 20), although the dataset holds 102 resumes labelled TEACHER.
  Their resume-side input produced 25-203 (secondary school) for 20 resumes;
  none of our seven readings does.
- The remaining candidates need information we do not have: a different version
  of the Kaggle dataset, a manual mapping, or a NIOCCS setting other than the
  default scheme.

## Consequence

Under Addendum 8, Step 0b would proceed on reading A with the differences
stated. With 1 teacher resume and 6 sales resumes, three of their nine
occupations cannot be tested, so the 27-test counts that are the target cannot
be formed. **Step 0b is not run** until the resume-side input is known. The
Salzburg replication team (draft 6) or the authors can settle it; with that,
Step 0a is rerun under the same rule.
