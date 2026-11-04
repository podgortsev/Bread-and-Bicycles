# RESULT: e2, the depth curve. No shape, on any ladder

The question the literature merges with interaction: does the penalty grow with
the NUMBER of signals, and if so, with what shape.

Design: two ladders, form and stated, at depths 0, 1, 2, 3, with three routes
through each depth where the design allows, clause count held constant at three
in every condition. 6,800 generations per model, accuracy outcome.

Source: `analysis_e2.txt`.

---

## The shape test

The quantity is the second difference, (D3 - D2) - (D2 - D1), computed per task
and bootstrapped. Negative means the curve bends downward, which is what
compounding looks like as depth grows. Positive means it saturates.

| model | ladder | depth means (1, 2, 3) | second difference | interval | reading |
|---|---|---|---|---|---|
| llama | form | +0.004, +0.014, +0.018 | -0.0067 | [-0.028, +0.014] | no shape |
| llama | stated | -0.008, -0.003, -0.005 | -0.0067 | [-0.039, +0.023] | no shape |
| mistral | form | +0.017, +0.012, +0.020 | +0.0133 | [-0.010, +0.038] | no shape |
| mistral | stated | -0.008, +0.003, +0.007 | -0.0075 | [-0.036, +0.020] | no shape |
| qwen | form | +0.008, -0.002, -0.003 | +0.0092 | [-0.013, +0.033] | no shape |
| qwen | stated | -0.008, -0.032, -0.034 | +0.0208 | [-0.008, +0.051] | no shape |

Six ladders, six intervals covering zero. No shape can be claimed anywhere.

The quadratic fit is deliberately not used: four points and three free
parameters cannot fail to improve on a line, and the second difference is the
quantity with an interval.

## What does move: qwen, stated, depth 2 and 3

Every depth against depth 0, corrected together across models and ladders:

| model | ladder | depth | net accuracy | adjusted p |
|---|---|---|---|---|
| qwen | stated | 2 | -0.0317 | 0.0117 * |
| qwen | stated | 3 | -0.0342 | 0.0117 * |

Every other cell of 18 is null (adjusted p 0.29 to 1.00). So on qwen, two or
three stated signals cost about 3 accuracy points against none, while one costs
nothing measurable, and the step from two to three adds nothing. That is a
threshold, not a slope, and the second difference above says the shape is not
established.

Note what this does NOT say. Channel A in e1 found no main effect for single
stated signals on qwen, and here two of them cost 3 points. The two readings are
consistent (one signal null, two signals cost, no further growth), and both are
small enough that the wrapper spread of e1w (0.05 to 0.16) covers them.

## Method 3b, corrected

Paper 2's method 3b compared depth 3 against depth 1 with an unpaired
Mann-Whitney and reported p=0.031. Paired over the routes that actually share
items, the same comparison gives p=0.0625 on the ladders where more than one
route exists at depth 3. The correction is recorded here because the earlier
number is in Paper 2.

## Refusal against depth

Reported per ladder and depth, and flat: refusal does not rise with the number
of signals on any model.

## What cannot be claimed

- Not that depth has no effect. Six nulls with intervals of roughly plus or
  minus 0.03 mean effects smaller than 3 accuracy points are invisible here.
- Not that the qwen threshold is a general result: it is one model, one ladder,
  and inside the wrapper spread.
