#!/usr/bin/env python3
"""
run_e3_choice.py - e3 channel C: does a credential win back the choice.

    10 conditions x 100 profiles x 2 questions x 2 orders
    = 4,000 forward passes per model.

Upload next to this script in Colab:
    e3_conditions.json      from experiments/shared/conditions/
    profiles.json           from experiments/shared/profiles/
    run_e1_choice.py        the instrument this reuses

WHY THIS IS A THIN FILE
-----------------------
The head-to-head choice instrument is written once, in
`e1-type-x-count/scripts/run_e1_choice.py`, and validated once, by
`validate_e1_channels.py`. This experiment points it at a different condition
set and changes nothing else.

That is deliberate. Two copies of a log-probability reader with a letter-mass
gate is two chances to fix a bug in one of them. The settings are passed as
ARGUMENTS rather than assigned onto the other module: an earlier version
assigned them, and the bundle smoke test caught that it left e1's file prefix
permanently pointing at e3. Everything that makes the
instrument trustworthy, both slot orders, the mass recorded on every read, the
gate applied in the analysis rather than at collection, and only within-slot
differences ever reported, comes along unchanged.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "e1-type-x-count", "scripts")))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_e1_choice as CHOICE      # noqa: E402

SETTINGS = dict(conditions_file="e3_conditions.json",
                prefix="e3_choice",
                reference="NONE",
                out_dir="/content/drive/MyDrive/afl/p3_e3")


def collect(model):
    CHOICE.collect(model, **SETTINGS)


def summary(model_key):
    CHOICE.summary(model_key, **SETTINGS)


if __name__ == "__main__":
    import p3_model
    key = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else "qwen"
    if "--analyse-only" in sys.argv:
        summary(key)
    else:
        collect(p3_model.P3Model(key))
        summary(key)
