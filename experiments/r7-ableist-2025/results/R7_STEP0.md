# R7 Step 0: ABLEIST Table 6 from the authors' full dataset

> **Redacted 2026-10-09.** Statements that rested on a private reply are removed
> from this document, because no permission to cite a personal communication was
> granted. Dates, numbers and verdicts are unchanged, and nothing here now
> depends on anything but the published artifacts and our own runs.

Run 2026-09-16, `scripts/r7_step0.py`, on the dataset the authors sent
(`correspondence/received.md`; kept outside version control). Rule:
Preregistration Addendum 7, committed before any per-group number was computed.
Console: `r7_step0_console.txt` (aggregates only).

## Registered verdict: FAIL

| reading | Llama-3.1-8B | OLMo-2-7B | rule (both at least 10 of 12) |
|---|---|---|---|
| A: intersectional-metric mean, groups restricted to marginalised values | 4/12 | 4/12 | fail |
| B: all-eight-metric mean, restricted | 1/12 | 1/12 | fail |
| C: intersectional-metric mean, unrestricted | 6/12 | 6/12 | fail |

Under the registered rule, their Table 6 is not reproduced, and no floor arm is
built on it.

## The reconstruction, and what it rests on

The reading below was found by searching the readings their caption leaves open,
and was fixed on 2026-09-16. It reproduces every published number to the second
decimal on two independent models, which is why it is reported as the
computation behind the table. It is a reconstruction, not a confirmation: the
registered verdict under Addendum 7 remains a failure, and both are reported
side by side.

## The statistic behind Table 6

The failures are informative. Reading C matches the first three columns to the
second decimal, and reading A matches the fourth column to the second decimal.
One mixed reading reproduces **all 24 numbers exactly** (12 of 12 on each model):

- score per conversation: mean of Tokenism, Inspiration Porn and
  Superhumanization Harm;
- "Dis.", "Dis.+Gen." and "Dis.+Cas." **unrestricted**: every gender (Man, Woman,
  Transgender) and both castes (Brahmin, Dalit);
- "Dis.+Gen.+Cas." **restricted** to the marginalised values (Woman or
  Transgender, and Dalit), as the caption says.

Group sizes per model: Dis. 30, Dis.+Gen. 90, Dis.+Cas. 60, Dis.+Nat. 60,
Dis.+Gen.+Cas. 60.

## Why this matters for their claim

The caption defines Gen. as Woman and Transgender and Cas. as Dalit, but only the
stacked group is built that way. The last two columns, which carry the
compounding reading ("harms are not simply additive but compounded across
marginalized identities", p. 8), therefore compare:

- Dis.+Gen. with men included, against Dis.+Gen.+Cas. with men excluded;
- Dis.+Cas. with Brahmins included, against Dis.+Gen.+Cas. with Brahmins excluded.

Part of each rise is the removal of the dominant identities from the second
group, not the addition of a second marginalised one. Like for like (reading A,
both groups restricted), the rise from Dis.+Gen. to Dis.+Gen.+Cas. is 15.73
percent (r 0.15) on Llama against the printed 26.64 (r 0.26), and on OLMo 20.00
percent (r 0.24) against 15.79 (r 0.22). For Dis.+Cas. to Dis.+Gen.+Cas.:
Llama 60.94 (r 0.46) against 68.85 (r 0.42), OLMo 10.00 (r 0.18) against 24.53
(r 0.31).

Every group holds 30 to 90 conversations: 5 per profile, occupation and model.

## Next

1. Done: the grouping is reconstructed and reproduces all 24 numbers (Addendum 12).
2. A new addendum registers, before anything neutral exists: the confirmed
   statistic as Step 0 target, the like-for-like comparison as the claim tested,
   and a floor arm. The floor arm needs generation with Llama-3.1-8B or OLMo-2-7B
   and labels from their open LoRA labeller (Llama-3.1-8B base), so it needs a
   Hugging Face login on this machine for the gated Llama weights.