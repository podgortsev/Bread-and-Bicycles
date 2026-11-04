# R7: does the authors' open labeller reproduce their labels?

> **Redacted 2026-10-09.** Statements that rested on a private reply are removed
> from this document, because no permission to cite a personal communication was
> granted. Dates, numbers and verdicts are unchanged, and nothing here now
> depends on anything but the published artifacts and our own runs.

Run 2026-09-17 and 18, `scripts/r7_label.py`. Rule: Addendum 10, committed
before the labeller was run once.

## Verdict under the registered reading: FAIL

Their prompt file's `ZERO_SHOT_COMBINED_PROMPT` in the
`### Instruction: / ### Input: / ### Response:` frame their inference script
builds gives **no labels at all**: the model continues the hiring conversation
instead of emitting the YAML block. Macro F1 0.000 on the pilot of 5
conversations, so the full 940 were not run.

## What was checked before calling it a failure

- **The adapter is really merged.** Their LoRA (r=128, seven target modules) was
  merged into Llama-3.1-8B-Instruct; `model.layers.0.self_attn.q_proj.weight`
  differs from the base by up to 1.95e-3 (mean 7.3e-5), so the weights changed.
- **Both prompt files were tried.** The repository has two: one beside their
  GPT prompting code, one beside the fine-tuning code. Neither instruction
  produces the format.
- **Three serving routes were tried** before Ollama: bitsandbytes 4-bit with the
  output layer on the CPU, 4-bit with the output layer quantised on the GPU, and
  bfloat16. On 6 GB the first two ran at minutes per conversation, and bfloat16
  does not fit. The deciding route is their adapter merged into the base model
  and quantised to Q4_K_M, served by Ollama.

## Exploratory, not registered

Adding a "```yaml" hint directly after `### Response:` makes the model emit the
metric block: 15 of 24 metric lines over three conversations, against 0 without
it. The chat template gives prose, never the block. So the fine-tuning used a
response format that the released files do not pin down, and the instruction
text is not in the repository (their training data are not released).

Nothing here is evidence about their published labels. It is evidence that
their open labeller cannot be driven from the released material alone.

## Next

The instruction string and an example response are not in the released files, so
the frame has to be inferred from the released inference code. Until Addendum
10's check passes, no R7 floor arm is built.

## Update 2026-09-22: where the failure came from

The frame and the instruction used here are inferred from the released inference
code, since the training instruction is not in the released files. Run that way,
the adapter behaves:

- **In bfloat16 the adapter works.** Their adapter on Llama-3.1-8B-Instruct in
  bfloat16, run on the CPU here, returns the expected block on a conversation
  from their released data, **8 of 8 metrics**, in 268 seconds.
- **Our serving route was the fault.** The same sample through the adapter
  merged and imported into Ollama as Q4_K_M returns a list of metric names and
  no labels, with or without a begin-of-text token. The import or the
  quantisation breaks the fine-tuned behaviour. The earlier FAIL therefore says
  nothing about their labeller.
- **Truncation.** Their scripts truncate on the right at 2,048 tokens, which
  cuts the "### Response:" cue on longer conversations. Their released data stay
  under about 1,860 tokens, so the limit is rarely reached there. Any rerun here
  uses a larger limit.

**Consequence for R7.** A valid labeller exists, but here it runs only in
bfloat16 on the CPU, at about four to five minutes per conversation (about 70
hours for the 940 open-model conversations, more for a floor arm). Their data
may not leave this machine, so a hosted GPU is not an option. R7's floor arm is
paused: it is the costliest item on the list for a target whose venue is still
unconfirmed, while its main finding (the Table 6 grouping) needs no labeller.