# RESULT: e2b, does corrupting the task change the form-by-form interaction

> **Correction, 2026-09-12.** The first version of this document, and the design
> rationale behind the experiment, described Liu et al. (2026) as reporting a
> SUPER-additive combination of ESL variation and typos. They do not. Their text:
> "the combined effect is often smaller than the sum of the two individual
> effects", "small but consistently positive" delta (sum minus combined) "across
> all models", and only "in some cases become superadditive". The data below are
> unchanged; the reading against their result is corrected.

The one published result closest to Paper 3's design. Liu et al.
(arXiv:2604.04723) cross English-as-a-second-language variation with typo
severity, applied to the task text itself, and report that combining them drops
performance more than either factor alone but, in most settings, less than the
sum of the two.

Both of their factors corrupt the text the model must read. Our e1 puts form
signals in the person's lead-in only. This experiment asks whether that
difference changes the interaction.

- **Arm A:** the signal appears in the person's lead-in; the task text is
  byte-identical in all four cells. This is e1's form-by-form pair.
- **Arm B:** the styling is applied to the task text as well, which is the
  condition Liu et al. measure.

600 tasks per arm per model. Source: `analysis_e2b.txt`.

---

## Result

| model | arm A, task clean | arm B, task styled | arm B minus arm A |
|---|---|---|---|
| llama | -0.0100 (p rank 0.42, mean 0.52) | **+0.0533** (0.0003, 0.0005) | **+0.0633** (0.0002, 0.0004) |
| mistral | -0.0083 (0.52, 0.61) | +0.0250 (0.17, 0.20) | +0.0333 (0.13, 0.11) |
| qwen | -0.0183 (0.15, 0.19) | -0.0117 (0.44, 0.51) | +0.0067 (0.63, 0.72) |

Negative is super-additive, positive is sub-additive. Minimum detectable
effects 0.025 to 0.035.

## Reading, against what Liu et al. actually report

**Task corrupted (their design).** On llama two form signals together cost 5.3
accuracy points less than the sum of their separate costs, significant on both
tests. That is the direction Liu et al. report for most of their settings. On
mistral the estimate points the same way and is not significant; on qwen it is
near zero. Their direction appears on one model of three.

**Task clean (our e1 design).** Null on all three models. When the form signal
lives only in the person's message and the task is untouched, the two form
signals do not interact detectably.

**The difference between arms** is +0.063 on llama and significant: corrupting
the task changes the interaction on that model.

So the honest summary is not "Liu et al. do not replicate", which the first
version of this document said. It is: their sub-additive direction reappears on
one model when the task itself is corrupted, and disappears when only the
person's message carries the signal.

## What cannot be claimed

- That their effect sizes replicate. Their benchmarks, models and severity levels
  differ from ours, and our significant cell is one model.
- That legibility explains their result. The legibility hypothesis was written to
  explain a super-additive combination that they did not report.
- Anything beyond the wrapper spread: e1w measured 0.05 to 0.16 for the same kind
  of interaction, and neither study carries a wrapper-level error bar.

## The subset check

For 59 of 600 tasks no eligible word existed for a typo, so their typo cell
equals the clean cell. Restricted to the 541 tasks that actually changed, the
llama arm B interaction is +0.041 (p rank 0.009, mean 0.012): same direction,
slightly smaller.
