# Axes of deviation: how the catalogue is built

Twenty-eight axes. The data are in `axes.csv` and `axes.json`; this file
explains why they are organised the way they are.

This is a design artefact, not an experimental input. No run script reads it.
It is the catalogue the signals actually tested were drawn from, and it records
what was left on the table.

---

## Why not organised by group

The obvious way is a list of categories: disability, language, culture, age,
gender.

That reproduces exactly the thing being argued against, that a person is best
understood by which box they fall into.

It also does not work in practice. Someone making typos could be anyone: a motor
impairment, a phone keyboard, a hurry, dyslexia. There is no category there, but
there is a deviation.

So the catalogue is organised **by type of deviation**, not by type of person.

---

## Four types

**FORM, ten axes.** How the person writes. Nothing is stated to the model; it
infers from the shape of the text.

**STATED, ten axes.** What the person says about themselves in one sentence. The
task does not change.

**CONDUCT, six axes.** How the person runs the exchange. Asks again, unfolds a
thought gradually, changes the question, seeks reassurance.

**KNOWING, two axes.** What the person does not know. Lacks the terminology, or
uses the wrong terminology.

---

## The main distinction: implicit against explicit

Eighteen axes are implicit, ten are explicit. This is not presentation, it is a
separate variable.

**Under an explicit signal** the person says "I am autistic". The model sees
that as a fact, and it has a trained response to the fact. What is measured is a
**trained reaction to a word**.

**Under an implicit signal** there is nothing to conceal, because the model does
not register that it is being examined. It is simply processing text of an
unfamiliar shape.

Hence the difference in what the two designs probe. The explicit signal probes
the response to a **disclosure**; the implicit signal probes what the model
infers from **form** alone.

That is a difference in the source of the response, not a demonstration that one
measures a category attitude and the other measures a reaction to unlikeness as
such. Implicit phrasing can activate the same social associations, and an
explicit disclosure carries more than category membership: it also tells the
model something about the person's situation and intent. Neither design isolates
one mechanism, and nothing in this study separates them.

In practice, both occur. Many interactions carry no explicit disclosure at all;
in others a person discloses deliberately, often precisely in order to get an
accessibility-aware answer. The implicit form is the one that arises without
anyone choosing it, which is why it is worth measuring alongside the explicit.

The published literature disagrees about what naming a trait explicitly does.
That disagreement is what makes this axis worth carrying.

---

## Two pairs where the same thing is presented two ways

**F01 against S09.** Non-native English: shown by the form of the writing, or
declared in a sentence.

**F04 against S07.** Imprecise typing: shown by typos, or declared as a motor
impairment.

These two pairs are **matched hypotheses, not the same material**. F01 and S09
are close: non-native English shown by the writing against declared. F04 and S07
are looser, and should not be called the same characteristic: F04 is observed
typos, whose cause is unknown, while S07 is a stated motor impairment. A reader
could produce F04's text for many reasons that have nothing to do with S07.

Method 3a runs this comparison. It does **not** settle it: with one pair per
comparison there is no test to run, declaring costs more in four of the six
model-by-pair numbers, Llama's typing pair goes the other way, and Mistral's are
noise. See `RESULT_method3a_single.md`, which states this explicitly. The
group-level contrast between the ten FORM and ten STATED axes is the stronger
evidence and the one to cite.

---

## What can be measured with what

**Twenty-four axes are single-turn.** Four, C01 to C04, need several turns and
are usable only by the answer-quality method and by an interaction-cost method.
The split is by turn count, not by group: CONDUCT holds six axes, of which C05
and C06 are single-turn and behave like the rest.

**The `methods` field is planned scope, not what was run.** It reads "1,2,3,4,5"
on the 24 single-turn axes and "2,5" on C01 to C04. Two caveats. Method 5,
interaction cost, was never run and is not planned for this paper, so every
mention of it in that field is aspirational. Methods 6, 7 and 8 were added after
this catalogue was written and do not appear in it at all. Read the field as the
design intention at the time of writing; `STATUS.md` records what was actually
collected.

That constraint is worth holding in mind when planning: the behavioural axes are
the most expensive of all, because each one needs a dialogue rather than a
single request.

---

## What is deliberately out of scope

**Gender and race through a name.** Densely studied territory where it is easy
to drown in other people's framing.

F09 is **not** a gender axis and should not be read as one. It is defined by an
observable feature of the text, hedging and softening, and it is measured as
that. Describing it as "the speech form associated with women" would reintroduce
through the back door the variable this catalogue puts out of scope, and would
assert a link between a linguistic feature and a demographic group that this
study neither tests nor needs.

**Religion.**

**Income and access to paid tools.** Hard to express in one sentence without
changing the task itself.

The catalogue is built around a person the system fits worse, not around
protected characteristics as such.

---

## How it is used

**First round.** One axis at a time against the baseline version. This is what
the present study does. The count per method is **24, not 28**, for any
single-turn method: C01 to C04 need several turns and are out of reach for
methods 1, 3 and 4. Only a method that runs a multi-turn exchange sees all 28.

**Second round.** Two axes at once. For example F01 plus S01: writes as a
non-native speaker and mentions a screen reader.

**Beyond that.** Three or more, along fixed routes rather than all combinations.
There are too many combinations to take them all, and a random selection gives
mush.

The question all of this is for: **how the degradation changes as axes are
added.** Additive, saturating, or accelerating. Three answers give three
different pictures and three different conclusions. The accumulation experiment
in this repository is the first attempt at it.

---

## Fields in the data

| column | meaning |
|---|---|
| `id` | F01-F10, S01-S10, C01-C06, K01-K02 |
| `group` | FORM, STATED, CONDUCT, KNOWING |
| `name` | short name of the axis |
| `signal` | implicit or explicit |
| `turns` | single, or several |
| `methods` | which measurement methods the axis is usable by |
| `example` | a sample phrasing |
| `note` | what the axis is and is not |

Rebuild with `python build_axes.py`, which is deterministic and overwrites both
files.
