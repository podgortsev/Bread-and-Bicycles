# e2: does the penalty grow with the NUMBER of signals

**Question.** Depth, not interaction. e1 asks whether a pair interacts; this
asks whether three signals cost more than one.

Paper 2's method 3b rebuilt to a standard it can be believed at.

## What changed from method 3b

All of it comes from m3b's own list of defects.

- **Mistral is included.** m3b ran on two models.
- **Signals are no longer selected on the outcome.** m3b picked six signals
  because they had already shown an effect, which makes it a follow-up rather
  than a test. Here they are fixed in advance by TYPE.
- **Two ladders, not one.** Form-only and stated-only. m3b mixed both types into
  single routes, so depth and type were confounded. If the thesis holds the two
  ladders have different shapes, and that comparison is the point.
- **Clause count is constant at three** at every depth, so depth never means
  "one more sentence".
- **The shape is fitted, not labelled.** m3b assigned COMPOUNDING or MIXED with
  arbitrary thresholds and said small changes flip the label. The analyser
  reports the second difference with a bootstrap interval, and prints
  "NO SHAPE CAN BE CLAIMED" when the interval covers zero.
- **The depth comparison is paired.** m3b used an unpaired Mann-Whitney across
  routes that carry both depths; corrected, its headline p moved from 0.031 to
  0.0625.
- **Multiplicity is corrected.** m3b applied no correction at depth.

## Design

    FORM ladder     F01 non-native, F04 typos, F07 informal
    STATED ladder   S01 screen reader, S08 age 74, S04 ADHD

Depths 0 to 3, seventeen conditions. The stated ladder carries three rotations
per depth so no single ordering drives the curve. The form ladder carries one
condition per combination, because two styles applied to the same clauses have
no order to vary.

That asymmetry is a property of what a form signal is, not an omission, and the
analyser prints the number of routes behind every point so a mean over one route
is never read as if it had a spread.

## Scripts

    ../shared/conditions/build_e2_conditions.py
    run_e2_depth.py       17 x 400 = 6,800 generations per model
    analyse_e2_depth.py   the read to trust, no GPU

400 tasks rather than 600: this estimates a curve from four points, and the
question is the shape across routes rather than the precision of any one point.
The first 400 rows include all 200 of Paper 2's own tasks, so the depth-0
baseline is directly comparable with method 3b's.
