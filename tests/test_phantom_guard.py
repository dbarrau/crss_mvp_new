"""Tests for the phantom-provision guard (application/_phantom.py).

The failure class: prose citations to provisions that do not exist in the
cited regulation — dominated by draft-numbering leakage (the AI Act's
"Title IA, Articles 4a–4c" from the Council/Parliament drafts). Fixtures
mirror the real cases from the 14 Jul 2026 forensic adjudication: the CRSS
self-classification answer (true positives) and the stored-eval false-positive
sweep (per-kind + adjacency rules).
"""
import pytest

from application._phantom import (
    _celex_numeric_form,
    build_provision_families,
    strip_phantom_citations,
)

AI = "32024R1689"
MDR = "32017R0745"
GDPR = "32016R0679"


@pytest.fixture()
def ref_index() -> dict:
    entries: dict[str, tuple[str, str]] = {}
    for n in (3, 4, 5, 6, 9, 10, 11, 13, 14, 15, 25, 28, 50, 51, 53, 95, 111, 113):
        entries[f"{AI}_article_{n}"] = (f"Article {n}", "EU AI Act")
    for r in ("I", "III", "V"):
        entries[f"{AI}_annex_{r.lower()}"] = (f"Annex {r}", "EU AI Act")
    for n in (43, 44, 71, 84):
        entries[f"{AI}_recital_{n}"] = (f"Recital {n}", "EU AI Act")
    for n in (2, 10, 61, 83, 110, 120):
        entries[f"{MDR}_article_{n}"] = (f"Article {n}", "MDR")
    # The consolidated MDR genuinely has a lettered Article 10a (EUDAMED
    # amendment) — existence must come from the graph, not a numbering rule.
    entries[f"{MDR}_article_10a"] = ("Article 10a", "MDR")
    entries[f"{MDR}_annex_i"] = ("Annex I", "MDR")
    # GDPR: articles only — recitals/annexes deliberately NOT ingested,
    # mirroring the live graph (99 article families, nothing else).
    for n in (5, 6, 9, 22, 32, 35, 99):
        entries[f"{GDPR}_article_{n}"] = (f"Article {n}", "GDPR")
    # Guidance ids must be ignored by the family index.
    entries["MDCG_2019_11_section_5_1"] = ("Section 5.1", "MDCG 2019-11")
    return entries


# ── the class that motivated the guard ─────────────────────────────────────

def test_draft_numbering_leakage_is_stripped(ref_index):
    """CRSS self-classification answer: 'Title IA (Articles 4a–4c)' is draft
    AI Act structure that never made the final text."""
    answer = (
        "CRSS is not a high-risk AI system.\n"
        "It is subject to the general obligations in Title IA (Articles 4a–4c) "
        "and transparency requirements under Article 50.\n"
        "Providers must maintain human oversight under Article 14."
    )
    cleaned, refs = strip_phantom_citations(answer, ref_index)
    assert refs == ["article 4a", "article 4c"]
    assert "4a" not in cleaned.split("PHANTOM CITATION FLAG")[1].split("\n\n")[1]
    assert "Article 50" not in cleaned.splitlines()[0]  # warning leads
    assert "PHANTOM CITATION FLAG" in cleaned
    assert "human oversight under Article 14" in cleaned  # clean lines kept


def test_markdown_bold_inside_range_still_caught(ref_index):
    """The demo report renders '**Articles 4a**–4c' — emphasis marks must not
    hide the range from the mention grammar."""
    answer = "Subject to Title IA (**Articles 4a**–4c) of the AI Act."
    cleaned, refs = strip_phantom_citations(answer, ref_index)
    assert "article 4a" in refs


def test_true_phantom_with_adjacent_act_alias(ref_index):
    """HQ_024/v5: 'MDR **Article 110a** explicitly requires compliance with
    GDPR' — MDR has Article 110 and Article 10a, but no 110a."""
    answer = (
        "The manufacturer must comply with post-market surveillance "
        "(**Article 83**), and data protection (MDR **Article 110a** "
        "explicitly requires compliance with GDPR)."
    )
    cleaned, refs = strip_phantom_citations(answer, ref_index)
    assert refs == ["article 110a"]


# ── silence rules (can't adjudicate → don't flag) ──────────────────────────

def test_out_of_corpus_act_is_silent(ref_index):
    """'Article 30 of Regulation (EU) 2019/1020' — we hold no text for that
    act, so even though no corpus act has the mention nearby, stay silent."""
    answer = (
        "Market surveillance follows Article 300 of Regulation (EU) 2019/1020, "
        "which the AI Act references."
    )
    _, refs = strip_phantom_citations(answer, ref_index)
    assert refs == []


def test_uningested_recitals_stay_silent(ref_index):
    """Stored-eval false positive: '(**Recital 71** GDPR)' is real law, but
    GDPR recitals are not in the graph — per-kind adjudication must skip."""
    answer = "Automated decision-making safeguards derive from Recital 71 GDPR."
    _, refs = strip_phantom_citations(answer, ref_index)
    assert refs == []


def test_non_adjacent_act_does_not_capture_mention(ref_index):
    """Stored-eval false positive (HQ_030): the AI Act's Annex V in a line
    that quotes a GDPR compliance statement must not be scoped to GDPR."""
    answer = (
        "Annex V — EU declaration of conformity | a statement that the AI "
        "system complies with Regulation (EU) 2016/679 where personal data "
        "is processed."
    )
    _, refs = strip_phantom_citations(answer, ref_index)
    assert refs == []


def test_empty_reference_index_is_noop():
    answer = "Subject to Articles 4a–4c."
    cleaned, refs = strip_phantom_citations(answer, {})
    assert cleaned == answer
    assert refs == []


# ── no false flags on real citation shapes ─────────────────────────────────

def test_real_lettered_article_kept(ref_index):
    answer = "Registration duties follow MDR Article 10a for EUDAMED."
    _, refs = strip_phantom_citations(answer, ref_index)
    assert refs == []


def test_numeric_range_endpoints_pass(ref_index):
    answer = "High-risk requirements are set out in Articles 9–15 of the AI Act."
    _, refs = strip_phantom_citations(answer, ref_index)
    assert refs == []


def test_paragraph_depth_never_checked(ref_index):
    """'Article 6(9)' passes when Article 6 exists — retrieval parses to
    varying depths, so a missing depth node must never flag a real article."""
    answer = "The derogation in Article 6(9) applies, per Article 6(3) AI Act."
    _, refs = strip_phantom_citations(answer, ref_index)
    assert refs == []


def test_explicit_valid_cross_reg_citation_kept(ref_index):
    answer = "A DPIA is required under Article 35 GDPR."
    _, refs = strip_phantom_citations(answer, ref_index)
    assert refs == []


def test_explicit_adjacent_phantom_in_named_act_flagged(ref_index):
    """'GDPR Article 113' — Article 113 exists (AI Act) but not in GDPR; the
    tight-adjacent alias makes the mis-scope adjudicable."""
    answer = "Erasure duties follow GDPR Article 113."
    _, refs = strip_phantom_citations(answer, ref_index)
    assert refs == ["article 113"]


# ── plumbing ───────────────────────────────────────────────────────────────

def test_family_index_skips_guidance_and_collapses_depth(ref_index):
    fams = build_provision_families(ref_index)
    assert set(fams) == {AI, MDR, GDPR}
    assert "article 10a" in fams[MDR]
    assert "annex v" in fams[AI]


def test_celex_numeric_form_derivation():
    assert _celex_numeric_form("32024R1689") == "2024/1689"
    assert _celex_numeric_form("32016R0679") == "2016/679"
    assert _celex_numeric_form("32026R0977") == "2026/977"
    assert _celex_numeric_form("MDCG_2019_11") is None


def test_every_catalog_regulation_gets_a_numeric_alias():
    """Regression net for the reg-detection-pattern-gap class: a newly
    ingested regulation must be attributable by number with zero curation."""
    from domain.legislation_catalog import LEGISLATION

    for celex in LEGISLATION:
        assert _celex_numeric_form(celex), f"no numeric alias for {celex}"


def test_verify_answer_integration(monkeypatch, ref_index):
    """The guard runs inside verify_answer when a reference index is passed,
    and is disabled by CRSS_PHANTOM_GUARD=0."""
    from application.verify import verify_answer

    answer = "Obligations arise from Title IA (Articles 4a–4c) of the AI Act."
    kwargs = dict(
        provisions=[],
        definitions=[],
        role_provisions=[],
        sufficiency={"ok": True},
        target_celexes=None,
        mentioned_regs=set(),
        role_specs=[],
        corrective_actions=[],
        question="Is CRSS high-risk?",
        reference_index=ref_index,
    )
    result = verify_answer(answer, **kwargs)
    assert "PHANTOM CITATION FLAG" in result.answer

    monkeypatch.setenv("CRSS_PHANTOM_GUARD", "0")
    result_off = verify_answer(answer, **kwargs)
    assert "PHANTOM CITATION FLAG" not in result_off.answer
