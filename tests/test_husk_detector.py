"""Amendment-husk detector (scripts/verify_completeness.py).

A husk is an amendment instruction whose quoted replacement text was dropped at
parse — the class that hid the AI Act date amendment. The signature: the lead-in
ends at its colon with nothing but closing punctuation. A genuine amendment keeps
its quoted replacement inline, so it must NOT match.
"""
from scripts.verify_completeness import _AMENDMENT_HUSK_RE

HUSKS = [
    "point (c) is replaced by the following: ;",          # the AI Act date case
    "point (a) is replaced by the following: ;",
    "the following point is added: ;",
    "the following point is added:",
    "Article 6 is replaced by the following:",
    "in Annex III, point 2 is replaced by the following: ‘",
]

NOT_HUSKS = [
    # genuine amendment — quoted replacement folded in (the fixed parse)
    "point (c) is replaced by the following: ‘(c) Chapter III, Sections 1, 2, "
    "and 3 shall apply from: (i) 2 December 2027 … (ii) 2 August 2028 …’ ;",
    "the following point is added: ‘(68) Regulation (EU) 2024/1689 …’",
    # ordinary provisions that merely contain a colon or the word "following"
    "The provider shall ensure the following requirements are met before placing "
    "the system on the market.",
    "‘safety component’ means a component of a product or of an AI system.",
    "This Regulation lays down harmonised rules on artificial intelligence.",
]


def test_husk_signatures_are_flagged():
    for t in HUSKS:
        assert _AMENDMENT_HUSK_RE.search(t), f"should flag husk: {t!r}"


def test_genuine_amendments_and_ordinary_text_are_not_flagged():
    for t in NOT_HUSKS:
        assert not _AMENDMENT_HUSK_RE.search(t), f"should NOT flag: {t!r}"
