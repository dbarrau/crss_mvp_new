"""Superseded-provision guard + record computation.

The guard flags a citation to a provision an amendment DELETED (AI Act Article
10(5), deleted by the Digital Omnibus, relocated to Article 4a(1)) — but ONLY when
the citation resolves to the act that deleted it. MDR Article 10(5) is real law
and must never be touched; an unattributed "Article 10(5)" must stay. Deletions
are sourced from the consolidation record, never inferred from node absence.
No Neo4j / LLM.
"""
from __future__ import annotations

from application._superseded import strip_superseded_citations, _SUPERSEDED_BY_ACT

_AI = "32024R1689"


def test_record_is_present_for_ai_act_10_5():
    # the generated record wires the guard: Article 10(5) → deleted, see 4a(1)
    rec = _SUPERSEDED_BY_ACT[_AI]["article 10(5)"]
    assert rec["deleted_ref"] == "Article 10(5)"
    assert rec["see_ref"] == "Article 4a(1)"
    assert rec["amender_name"] == "Digital Omnibus on AI"
    assert rec["point_num"] == "9"


def test_ai_act_scoped_citation_is_flagged_and_relocated():
    ans = "Providers may process special categories under Article 10(5) of the AI Act."
    out, notes = strip_superseded_citations(ans)
    assert len(notes) == 1
    assert "SUPERSEDED PROVISION FLAG" in out
    assert "Article 10(5)" in out and "was deleted by the Digital Omnibus" in out
    assert "see **Article 4a(1)**" in out
    assert "point (9)" in out
    # the offending line is removed from the body
    assert "may process special categories" not in out.split("FLAG")[1].split("\n\n", 1)[-1]


def test_eur_lex_link_celex_scopes_the_citation():
    # a mention linked to the AI Act CELEX is flagged even without the words "AI Act"
    ans = ("The permission is [**Article 10(5)**]"
           "(https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689&qid=1#art_10).")
    out, notes = strip_superseded_citations(ans)
    assert len(notes) == 1 and "Article 4a(1)" in out


def test_mdr_scoped_10_5_is_never_flagged():
    # MDR Article 10(5) is real current law — the guard must not touch it
    ans = "Under Article 10(5) of the MDR, manufacturers keep the QMS current."
    out, notes = strip_superseded_citations(ans)
    assert notes == []
    assert out == ans


def test_unattributed_citation_is_not_flagged():
    # no act reference on the line → cannot attribute → must not flag
    ans = "This turns on Article 10(5)."
    out, notes = strip_superseded_citations(ans)
    assert notes == [] and out == ans


def test_live_provision_is_untouched():
    ans = "Data governance practices are set out in Article 10(2) of the AI Act."
    out, notes = strip_superseded_citations(ans)
    assert notes == [] and out == ans


def test_answer_scoping_catches_an_unattributed_straggler():
    # one attributed AI-Act mention promotes a later bare "Article 10(5)" (no act
    # word on its own line) — since the family resolves only to the AI Act.
    ans = (
        "Processing is permitted under Article 10(5) of the AI Act.\n"
        "In short, Article 10(5) is your basis for bias detection."
    )
    out, notes = strip_superseded_citations(ans)
    assert len(notes) == 1
    body = out.split("\n\n", 1)[1]
    assert "your basis for bias detection" not in body     # straggler line removed too


def test_conflicting_attribution_does_not_over_strip():
    # 10(5) cited for BOTH the AI Act and the MDR → not answer-promoted; only the
    # AI-Act line is stripped, the (real) MDR line is kept.
    ans = (
        "Bias data falls under Article 10(5) of the AI Act.\n"
        "Separately, Article 10(5) of the MDR governs the QMS."
    )
    out, notes = strip_superseded_citations(ans)
    assert len(notes) == 1
    assert "of the MDR governs the QMS" in out              # MDR line preserved
    assert "Bias data falls under" not in out.split("\n\n", 1)[1]


def test_blockquote_footer_is_never_stripped():
    # the amendment-provenance footer legitimately references the change
    ans = "> Article 10 — paragraph 5 is deleted (Regulation (EU) 2026/1744)."
    out, notes = strip_superseded_citations(ans)
    assert notes == [] and out == ans


def test_deleted_annex_point_is_detected_when_scoped_to_ai_act():
    # coverage is driven by the record, not just article-paragraphs: a deleted
    # Annex point (Annex I, Section A, point 1) is caught too, scoped to the AI Act
    ans = "The harmonisation act is listed in Annex I, Section A, point 1 of the AI Act."
    out, notes = strip_superseded_citations(ans)
    assert len(notes) == 1
    assert "Annex I, Section A, point 1" in out and "was deleted" in out


def test_deleted_annex_point_not_flagged_for_another_act():
    ans = "See Annex I, Section A, point 1 of the MDR."
    out, notes = strip_superseded_citations(ans)
    assert notes == [] and out == ans


def test_computed_record_matches_the_generated_module():
    # the committed module must equal a fresh computation (no drift)
    from consolidation.superseded import compute_superseded, record_dicts
    from domain.ontology.superseded_provisions import SUPERSEDED
    fresh = record_dicts(compute_superseded(_AI, "32026R1744"))
    assert fresh == SUPERSEDED
