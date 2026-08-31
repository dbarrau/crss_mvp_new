"""Tests for the date-currency guard (application/_date_currency.py).

The Digital Omnibus (Reg 2026/1744) delayed the AI Act high-risk application
dates; the model states the pre-Omnibus dates from memory. The guard rewrites a
superseded date ONLY when pinned to the matching high-risk scope — never the
unchanged general-application 2 August 2026.
"""
from application._date_currency import correct_superseded_dates as fix


# ── must correct ─────────────────────────────────────────────────────────────

def test_annex_iii_stale_date_corrected():
    out, notes = fix("Annex III high-risk AI systems apply from 2 August 2026.")
    assert "2 December 2027" in out
    assert "2 August 2026" not in out
    assert len(notes) == 1 and "113(3)(c)(i)" in notes[0]


def test_article_6_2_scope_also_triggers():
    out, notes = fix("High-risk AI under Article 6(2) applies from 2 August 2026.")
    assert "2 December 2027" in out and notes


def test_annex_i_stale_date_corrected():
    out, notes = fix("Annex I high-risk (safety components) apply from 2 August 2027.")
    assert "2 August 2028" in out
    assert "2 August 2027" not in out
    assert len(notes) == 1 and "113(3)(c)(ii)" in notes[0]


def test_nbsp_dated_line_corrected():
    # EUR-Lex/model output can carry a non-breaking space in the date.
    out, notes = fix("Annex III high-risk systems apply from 2 August 2026.")
    assert "2 December 2027" in out and notes


def test_transparency_note_returned_for_appending():
    _, notes = fix("Annex III high-risk applies 2 August 2026.")
    assert notes and "Digital Omnibus" in notes[0]


# ── must NOT touch (false-positive guards) ───────────────────────────────────

def test_general_application_date_untouched():
    # 2 August 2026 is the correct, unchanged general-application date.
    ans = "The AI Act generally applies from 2 August 2026."
    out, notes = fix(ans)
    assert out == ans and not notes


def test_originally_then_now_phrasing_untouched():
    # Current date already stated → the answer is already correct; leave it.
    ans = ("Annex III high-risk originally applied from 2 August 2026 but the "
           "Omnibus delayed it to 2 December 2027.")
    out, notes = fix(ans)
    assert out == ans and not notes


def test_unrelated_date_untouched():
    ans = "The Article 5 prohibitions apply from 2 February 2025."
    out, notes = fix(ans)
    assert out == ans and not notes


def test_annex_i_rule_does_not_fire_on_annex_iii_line():
    # "Annex III" must not be caught by the Annex I rule's "annex i" scope.
    ans = "Annex III high-risk systems apply from 2 August 2027."  # wrong scope+date pair
    out, notes = fix(ans)
    # The Annex I rule is excluded on Annex III; the Annex III rule targets a
    # different superseded date (2 Aug 2026), so this line is left unchanged.
    assert "2 August 2028" not in out
    assert not any("113(3)(c)(ii)" in n for n in notes)


def test_safety_component_marker_triggers_annex_i_rule():
    # The model often writes "safety components" instead of the literal "Annex I".
    out, notes = fix("From 2 August 2027, providers of high-risk AI systems that "
                     "are safety components of products must comply with the AI Act.")
    assert "2 August 2028" in out and "2 August 2027" not in out
    assert notes


def test_sandbox_deadline_2027_untouched():
    # 2 August 2027 is also the (unchanged) regulatory-sandbox deadline — must stay.
    ans = "Member States shall ensure regulatory sandboxes are operational by 2 August 2027."
    out, notes = fix(ans)
    assert out == ans and not notes


def test_empty_answer():
    assert fix("") == ("", [])
