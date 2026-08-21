"""Legal-Markdown normalizer for the .md export (demo/md_normalize.py).

The server/export twin of the front-end ``normalizeLegalLists`` — it must keep
the exported .md rendering identically to the live view: real nested lists,
expanded inline point-runs, and consistent citation bolding.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

from md_normalize import (
    normalize_legal_markdown as N,
    _bold_reference_runs as B,
    _split_glued_headings as SPLIT,
)


# ── bold-reference runs ──────────────────────────────────────────────────────

def test_bold_run_extends_over_digit_enumeration():
    assert B("comply with **Articles 26**, 27, and 49(3) now.") == \
        "comply with **Articles 26, 27, and 49(3)** now."


def test_bold_run_stops_before_trailing_prose():
    assert B("**Articles 26**, 27 (if a public entity)") == \
        "**Articles 26, 27** (if a public entity)"


def test_bold_run_handles_roman_annexes():
    assert B("See **Annex III**, VIII and IX.") == "See **Annex III, VIII and IX**."


def test_bold_run_does_not_match_a_roman_looking_word():
    # "did" is all roman letters but lowercase → never a reference.
    s = "under **Article 5**, or did the system change?"
    assert B(s) == s


def test_bold_run_leaves_a_lone_reference_untouched():
    s = "**Article 6** alone governs classification."
    assert B(s) == s


# ── inline enumeration → real list ───────────────────────────────────────────

def test_inline_enumeration_becomes_a_list():
    out = N("repealed with the exception of: (a) the vigilance obligations in "
            "Annex III; (b) the performance-study obligations in Annex XIV.")
    lines = out.split("\n")
    assert lines[0] == "repealed with the exception of:"
    assert lines[1] == "- (a) the vigilance obligations in Annex III;"
    assert lines[2] == "- (b) the performance-study obligations in Annex XIV."


def test_reference_chain_is_not_split_into_a_list():
    # A citation of several points must stay one line (its items include a bare
    # connector like "and"/"of" — not real clauses).
    s = "This applies to points (a), (b) and (d) of the first subparagraph."
    assert N(s) == s


# ── native 1-space bullets → real nesting ────────────────────────────────────

def test_one_space_bullets_are_reindented_to_valid_nesting():
    out = N("Duties:\n - First duty.\n  - Nested detail.\n - Second duty.")
    assert "   - First duty." in out       # 1 space → 3
    assert "      - Nested detail." in out  # 2 spaces → 6 (nests under First)


def test_code_fences_are_left_untouched():
    src = "```\n - not a list, code\n(a) not an enum\n```"
    assert N(src) == src


# ── heading glued to a run-in answer part ────────────────────────────────────

def test_heading_glued_to_answer_part_is_split():
    # "### Final AnswerA. …" would otherwise swallow the whole sentence into <h3>.
    out = SPLIT("### Final AnswerA. The R&D exemption applies only if X.")
    assert out == "### Final Answer\n\nA. The R&D exemption applies only if X."


def test_heading_glued_via_punctuation_is_split():
    # "?" or ")" immediately followed by a capitalised word is a run-in glue.
    assert SPLIT("### 1. Is my system high-risk?Key provision: Article 6 applies.") == \
        "### 1. Is my system high-risk?\n\nKey provision: Article 6 applies."
    assert SPLIT("### A. Prohibited Practices (Article 5)Banned outright from 2025.") == \
        "### A. Prohibited Practices (Article 5)\n\nBanned outright from 2025."


def test_legit_letter_headings_are_not_split():
    # The model's own section headings ("### A. Scope", "#### B. Deployer"), a
    # "Section A." title (space before the letter), and a heading that merely
    # ends in ")" or "?" must all be left intact.
    for h in ["### A. Scope of the R&D Exemption",
              "#### B. Deployer status of the university department",
              "### Section A. Overview",
              "### Scope (Article 2)",
              "### Is my system high-risk?",
              "### Requirements (Chapter III) and obligations",
              "### Summary of Obligations"]:
        assert SPLIT(h) == h
