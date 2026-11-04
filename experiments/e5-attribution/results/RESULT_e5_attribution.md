# RESULT: e5, can the model say which of two signals moved it

Paper 2's method 7 established that a model knows a disclosure was present and
does not know what it did, with one signal. With two present the question
sharpens into an attribution question that has a ground truth: the factorial in
e1 measured what each signal actually did, on the same task.

Design: replay the exchange e1 channel A actually collected, then ask the model
to rate, 0 to 10, how much each of the two signals changed its answer. Turn one
is never regenerated. 420 exchanges per model.

Source: `analysis_e5.txt`.

---

## Read the degeneracy check first

| model | parsed | mean rating | sd | most common value | usable |
|---|---|---|---|---|---|
| llama | 100 percent | 0.00 | 0.00 | 0 in 100 percent of answers | EXCLUDED |
| mistral | 98.3 percent | 0.70 | 1.76 | 0 in 74 percent | usable |
| qwen | 100 percent | 5.93 | 4.86 | 10 in 58 percent | usable |

llama answers zero to all 840 ratings. Its self-report is a constant. "The model
denies any influence" would be the wrong reading: the instrument said nothing.
This is the second model in the programme to fail exactly this way, after
llama's constant "6" in method 7.

## The ranking test

Within each pair, does the model rate the signal that actually moved the answer
above the one that did not. Ties in the measured effect are excluded rather than
counted as wins, which leaves few informative items.

| model | right | wrong | share | adjusted p |
|---|---|---|---|---|
| mistral | 2 | 7 | 22 percent | 0.18, not distinguishable from guessing |
| qwen | 16 | 6 | 73 percent | 0.105, not distinguishable from guessing |

qwen is right about three times in four, and with 22 informative items that does
not separate from chance after correction. Reported as a direction worth
powering properly, not as a finding.

## The item-level test

Does the rating on task T track the measured change on task T.

| model | signal | rho | adjusted p |
|---|---|---|---|
| qwen | F01 non-native form | -0.229 | 0.012 * |
| qwen | F04, S01, S08 | +0.000, +0.002, +0.085 | 1.00, 0.98, 0.19 |
| mistral | F01, S01 | +0.097, +0.026 | 0.21, 0.69 |
| mistral | F04, S08 | degenerate, no variance | excluded |

One of six testable cells is significant, and it is small and negative: qwen
claims more influence for non-native phrasing on the tasks where that phrasing
actually helped it least. Method 7 found zero of nine cells on single signals;
this is one of six on pairs.

Two cells are degenerate (a constant rating or a constant measured change) and
are excluded from the correction rather than counted as nulls. Their NaN
p-values previously entered the correction and turned one significant cell into
five; `p3_stats.bh` now refuses NaN outright.

## The unstated-cue question

The person never said anything about phrasing or typing. Those are properties of
the writing.

| model | claimed for form cues | claimed for stated cues |
|---|---|---|
| qwen | 9.65 of 10 | 3.15 of 10 |
| mistral | 1.01 | 0.46 |

qwen attributes near-maximal influence to cues the person never mentioned, three
times what it attributes to what the person actually said. Combined with the
item-level null, the picture is a model that produces confident attributions
uncorrelated with its own behaviour.

## What cannot be claimed

- Nothing about llama: a constant is not a judgement.
- Not that qwen can rank the two signals: 73 percent on 22 items does not
  survive correction.
- The item-level test is attenuated by construction, because per-item accuracy
  change is in {-1, 0, +1}. A null here is weaker evidence than a null on a
  continuous outcome.
