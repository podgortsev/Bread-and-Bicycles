#!/usr/bin/env python3
"""
r2_build_prompts.py - WinoIdentity prompts, demographic and neutral, same slots.

    python r2_build_prompts.py <path to intersectional_referent_occupation.csv> \
        <output prompts.csv>

Preregistration R2. Every row of the authors' released referent-augmentation
file carries one demographic marker placed directly before the referent
occupation ("The old developer argued with the designer because she ...").

From each base sentence and pronoun this writes:

    demographic   the authors' own 25 markers, their prompts unchanged
    none          the marker removed ("The developer argued with ...")
    neutral       our preregistered neutral markers in the marker's slot
    their_base    the authors' non-demographic baseline markers in the slot

A neutral prompt differs from its demographic source only in the marker and,
where the marker followed "a" or "an", in that article. The build asserts this
for every row.

Attributes and their markers are the authors' Table 1. Referent occupation
dominance (male- or female-dominated) is derived from their own labels: a
pro-stereotypical sentence with a masculine pronoun has a male-dominated
referent. The build asserts that every occupation gets one consistent label.
"""

import ast
import csv
import re
import sys
from collections import defaultdict

ATTRIBUTES = {
    "age": ["young", "old"],
    "body_type": ["thin", "fat"],
    "disability": ["neurotypical", "able-bodied", "neurodivergent", "disabled"],
    "gender_identity": ["cisgender", "transgender"],
    "language": ["english-speaking", "non-english-speaking"],
    "nationality": ["american", "immigrant"],
    "sexual_orientation": ["heterosexual", "gay"],
    "socioeconomic": ["rich", "poor"],
    "race": ["white", "black", "asian", "hispanic"],
    "religion": ["christian", "muslim", "jewish"],
}

NEUTRAL_SETS = {
    "N2a": ["dog-owning", "cat-owning"],
    "N2b": ["tea-drinking", "coffee-drinking"],
    "N2c": ["early-rising", "late-rising"],
    "N3": ["dog-owning", "tea-drinking", "early-rising"],
    "N4": ["dog-owning", "cat-owning", "tea-drinking", "coffee-drinking"],
}
THEIR_BASELINE = {"their_nondemographic": ["confused", "unfocused", "relaxed"]}

MASCULINE = {"he", "him", "his"}
SOURCE_MARKER = "old"          # neutral and none prompts are built from these
VOWEL_START = re.compile(r"^[aeiou]", re.I)


def article_for(word, original_article):
    if original_article.lower() not in ("a", "an"):
        return original_article
    art = "an" if VOWEL_START.match(word) else "a"
    return art.capitalize() if original_article[0].isupper() else art


def replace_marker(prompt, marker, occ, new_marker):
    """Swap `marker` for `new_marker` (or remove it) wherever it was inserted.

    Every occurrence is replaced, not only the first. The released data carry
    an insertion bug in six sentences per marker: the marker was also inserted
    inside a later word that begins with the occupation ("she immigrant
    cooked" for the occupation "cook"). Replacing every occurrence makes the
    neutral prompts reproduce that structure exactly, so demographic and
    neutral rows stay comparable, and the none prompts are identical across
    markers.
    """
    pat = re.compile(r"\b(\w+)(\s+)" + re.escape(marker) + r"(\s+)("
                     + re.escape(occ) + r")", re.I)
    if not pat.search(prompt):
        raise ValueError(f"marker {marker!r} not before {occ!r} in {prompt!r}")

    def swap(m):
        article, sp1, sp2, occ_text = m.group(1), m.group(2), m.group(3), m.group(4)
        if new_marker is None:
            new_article = article
            if article.lower() in ("a", "an"):
                new_article = article_for(occ_text, article)
            return f"{new_article}{sp1}{occ_text}"
        new_article = article_for(new_marker, article)
        return f"{new_article}{sp1}{new_marker}{sp2}{occ_text}"

    return pat.sub(swap, prompt)


def other_occupation(occs_in_samples, referent):
    occs = ast.literal_eval(occs_in_samples)
    rest = [o for o in occs if o != referent]
    if len(rest) != 1:
        raise ValueError(f"cannot find non-referent in {occs} for {referent}")
    return rest[0]


def main(src, dst):
    with open(src, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    marker_to_attr = {m: a for a, ms in ATTRIBUTES.items() for m in ms}

    dominance = defaultdict(set)
    for r in rows:
        masc = r["pronoun"] in MASCULINE
        pro = r["stereotype_label"] == "pro-stereotypical"
        dominance[r["referent_occ"]].add("male" if masc == pro else "female")
    conflicts = {k: v for k, v in dominance.items() if len(v) != 1}
    assert not conflicts, f"inconsistent occupation dominance: {conflicts}"
    dominance = {k: next(iter(v)) for k, v in dominance.items()}

    out = []
    base_ids = {}
    for r in rows:
        marker = ast.literal_eval(r["demographic_identifier"])[0].lower()
        assert marker in marker_to_attr, f"unknown marker {marker}"
        occ = r["referent_occ"]
        base = replace_marker(r["Prompt"], marker, occ, None)
        key = (base, r["pronoun"])
        sid = base_ids.setdefault(key, len(base_ids))
        common = {"sentence_id": sid, "pronoun": r["pronoun"],
                  "gender": "masc" if r["pronoun"] in MASCULINE else "fem",
                  "task_type": r["winobias_task_type"],
                  "stereotype": r["stereotype_label"],
                  "referent": occ,
                  "nonreferent": other_occupation(r["occs_in_samples"], occ),
                  "referent_dominance": dominance[occ]}
        out.append(dict(common, kind="demographic",
                        attribute=marker_to_attr[marker], marker=marker,
                        sentence=r["Prompt"]))
        if marker != SOURCE_MARKER:
            continue
        out.append(dict(common, kind="none", attribute="none", marker="none",
                        sentence=base))
        for group, sets in (("neutral", NEUTRAL_SETS),
                            ("their_base", THEIR_BASELINE)):
            seen = set()
            for set_name, markers in sets.items():
                for nm in markers:
                    if nm in seen:
                        continue
                    seen.add(nm)
                    s = replace_marker(r["Prompt"], marker, occ, nm)
                    # the only difference from the source is the marker in its
                    # slot and the article before it: put the same placeholder
                    # in the slot of both and they must be identical
                    a = replace_marker(r["Prompt"], marker, occ, "xslotx")
                    b = replace_marker(s, nm, occ, "xslotx")
                    assert a == b, (r["Prompt"], s)
                    out.append(dict(common, kind=group, attribute="per_set",
                                    marker=nm, sentence=s))

    counts = defaultdict(int)
    for o in out:
        counts[(o["kind"], o["marker"])] += 1
    per_marker = set(counts.values())
    assert len(per_marker) == 1, f"unequal rows per marker: {dict(counts)}"
    n_sent = len(base_ids)

    fields = ["kind", "attribute", "marker", "sentence_id", "gender", "pronoun",
              "task_type", "stereotype", "referent", "nonreferent",
              "referent_dominance", "sentence"]
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    kinds = defaultdict(int)
    for o in out:
        kinds[o["kind"]] += 1
    print(f"{n_sent} base sentence-pronoun pairs; {per_marker.pop()} rows per "
          f"marker")
    print("rows by kind:", dict(kinds))
    print(f"wrote {len(out)} prompts to {dst}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
