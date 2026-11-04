# e2b: is the published form-by-form result a legibility effect

> **Correction, 2026-09-12.** This experiment was designed on a misreading of
> Liu et al. as super-additive. Their own text says the combined drop is "often
> smaller than the sum of the two individual effects", with "small but
> consistently positive" delta = sum minus combined across all models, and only
> "in some cases" super-additive. The design below still answers a real
> question, whether corrupting the task changes the interaction, but the reading
> table's first and third rows were written against the wrong prior. See
> `results/RESULT_e2b_legibility.md`.

The cheapest experiment in Paper 3 and possibly the most valuable.

**Question.** Liu et al. (arXiv:2604.04723) cross ESL variants with typo levels
and report a combined effect larger than either alone and not simply additive.
They apply both factors to the TASK. Two corruptions of an input the model must
parse compound mechanically, because legibility failures multiply. Is that what
their result is.

## The two arms

    arm A   lead-in styled, task byte-identical      collected by e1 channel A
    arm B   lead-in styled AND task styled           collected here

Same 600 tasks, same keys, same lead-ins, same parser. The only difference is
the task text.

    legibility component = interaction(arm B) - interaction(arm A)

paired within task, because both arms saw the same items.

Three outcomes, all worth having:

| result | reading |
|---|---|
| B super-additive, A not | the published effect is legibility, not treatment |
| both super-additive | the form pair compounds even with a clean task, a stronger claim than the literature has shown |
| neither | Liu et al. do not replicate here, and the MDE says how large an effect this run could have seen |

## The rule that makes the rewriting safe

A rewritten task must have the same answer. Asserted at build time, not assumed:
every digit run in the original appears in the rewrite in the same multiset,
none is added, and the key is copied rather than recomputed.

The ESL rules are deliberately narrow (article deletion, three preposition
calques, two auxiliary drops) and none of them touches a comparative, a
superlative or a quantity word. Typos are adjacent-character transpositions,
never on the first or last letter, never on a word carrying what the question
asks.

**What this cannot guarantee, and the paper says so:** `validate_tasks.py`
re-derives keys by parsing Paper 2's exact phrasing, so it cannot parse a
rewritten question. The rewritten sets get no independent re-derivation, and the
numeric assertions stand in for it.

On 59 of the 600 tasks there is no word eligible for a typo, so the typo arm
equals std there. Those rows are flagged `text_changed=0` and the analysis
reports the interaction with and without them. Dropping them silently would
overstate the arm; keeping them silently would dilute it.

## Scripts

    build_e2b_tasks.py           the rewrites, with the assertions
    run_e2b_legibility.py        4 x 600 = 2,400 generations per model
    analyse_e2b_legibility.py    give it BOTH this file and e1's channel A file
