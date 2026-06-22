"""Tests for the curated temporal-applicability model (#6)."""
from datetime import date

from domain.ontology.applicability import (
    APPLICABILITY,
    applicability_note,
    status_as_of,
)


def test_ai_act_staged_application_boundaries():
    celex = "32024R1689"
    # Before entry into force.
    assert status_as_of(celex, date(2024, 7, 1))["in_force"] is False
    # After EIF but before prohibitions apply.
    s = status_as_of(celex, date(2024, 9, 1))
    assert s["in_force"] is True and len(s["applied"]) == 0
    # Prohibitions (2025-02-02) in force, general application not yet.
    s = status_as_of(celex, date(2025, 3, 1))
    assert len(s["applied"]) == 1 and s["generally_applicable"] is False
    # Mid-2026: prohibitions + GPAI/governance apply; general + Annex I pending.
    s = status_as_of(celex, date(2026, 6, 22))
    assert len(s["applied"]) == 2 and len(s["pending"]) == 2
    assert s["generally_applicable"] is False
    # After general application.
    s = status_as_of(celex, date(2026, 9, 1))
    assert s["generally_applicable"] is True and len(s["applied"]) == 3
    # After the Annex I high-risk date everything has applied.
    assert len(status_as_of(celex, date(2027, 9, 1))["pending"]) == 0


def test_unknown_celex_returns_none():
    assert status_as_of("99999R9999", date(2026, 6, 22)) is None


def test_note_flags_ai_act_pending_general_application_in_mid_2026():
    note = applicability_note({"32024R1689"}, date(2026, 6, 22))
    assert "EU AI Act" in note
    assert "NOT YET applicable" in note          # general application is pending
    assert "Art 113" in note                     # carries the governing citation
    assert "2026-08-02" in note


def test_note_omits_long_applicable_single_stage_regulation():
    # GDPR has one milestone long in the past -> not temporally interesting,
    # so it is suppressed to avoid noise.
    assert applicability_note({"32016R0679"}, date(2026, 6, 22)) == ""


def test_note_includes_mdr_transitional_when_pending():
    # MDR legacy-device transition runs to 2027/2028, so as of mid-2026 it is
    # surfaced with its transitional note.
    note = applicability_note({"32017R0745"}, date(2026, 6, 22))
    assert "MDR" in note and "transitional" in note.lower()


def test_note_empty_for_no_known_celexes():
    assert applicability_note(set(), date(2026, 6, 22)) == ""
    assert applicability_note({"99999R9999"}, date(2026, 6, 22)) == ""


def test_milestones_are_chronological_and_cited():
    for reg in APPLICABILITY.values():
        dates = [m.date for m in reg.milestones]
        assert dates == sorted(dates), f"{reg.celex} milestones out of order"
        assert all(m.citation for m in reg.milestones)
