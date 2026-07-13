"""Regression tests for the four misattribution false-flag classes found by
forensic adjudication of the v2 eval runs (13 Jul 2026). Each fixture models
the real case that exposed the class.
"""
from application._faithfulness import (
    _base_ref_family,
    _nearest_citation_ref,
    _unique_grounding_ref,
    check_faithfulness,
)

_ART5_POINT_F = (
    "the placing on the market, the putting into service for this specific "
    "purpose, or the use of AI systems to infer emotions of a natural person "
    "in the areas of workplace and education institutions."
)
_RECITAL_44 = (
    "There are serious concerns about the scientific basis of AI systems "
    "aiming to identify or infer emotions, particularly as expression of "
    "emotions vary considerably across cultures and situations."
)


def test_family_union_grounds_descendant_keyed_source():
    """HQ_006/a: verbatim Article 5(1)(f) text, source keyed at a different
    depth than the citation label — must NOT flag."""
    provisions = [
        # Source keyed at paragraph depth (qualified display_ref era)
        {"article_ref": "Article 5(1)", "article_text": _ART5_POINT_F},
    ]
    answer = f"Under **Article 5(1)(f)**, “{_ART5_POINT_F.rstrip('.')}”"
    report = check_faithfulness(answer, provisions)
    assert report.misattributed_count == 0
    assert report.verified_count == 1


def test_family_union_still_flags_cross_family_misattribution():
    """Recital 44 text cited as Article 5 (the one TRUE misattribution in the
    forensics) must still flag."""
    provisions = [
        {"article_ref": "Article 5", "article_text": "Prohibited practices text."},
        {"article_ref": "Recital 44", "article_text": _RECITAL_44},
    ]
    answer = f"**Article 5** states: “{_RECITAL_44.rstrip('.')}”"
    report = check_faithfulness(answer, provisions)
    assert report.misattributed_count == 1


def test_guidance_section_label_blocks_adjudication():
    """HQ_030/v2: a quote attributed to 'Section 17' (a guidance label the
    source map cannot key) must not be adjudicated against an unrelated
    earlier Article label in the window."""
    provisions = [
        {"article_ref": "Article 6", "article_text": "Classification rules for high-risk systems."},
        # The guidance text lives in the corpus (grounded) but is keyed
        # under a ref the citation regex cannot produce.
        {"article_ref": "Annex I", "article_text": _RECITAL_44},
    ]
    answer = (
        "High-risk status follows **Article 6**. Guidance in Section 17 adds: "
        f"“{_RECITAL_44.rstrip('.')}”"
    )
    report = check_faithfulness(answer, provisions)
    assert report.misattributed_count == 0


def test_nearest_label_extraction():
    answer = "Per **Article 6** and then MDCG 2019-11 Section 5.1 says “x”"
    assert _nearest_citation_ref(answer, answer.index("“")) is None
    answer2 = "Per Section 3 and then **Article 6(3)** says “x”"
    assert _nearest_citation_ref(answer2, answer2.index("“")) == "Article 6(3)"
    answer3 = "Under **Annex IX, Chapter I, point 3.5** the NB shall “x”"
    assert _nearest_citation_ref(answer3, answer3.index("“")) == "Annex IX"


def test_scenario_quote_is_exempt():
    """HQ_030/rerun: the model quoting the user's own scenario back is not a
    regulatory quote and must not count as fabrication."""
    question = (
        "Our AI system assists radiologists in detecting lung nodules in "
        "chest X-rays and flags AI-generated output for radiologist review."
    )
    provisions = [{"article_ref": "Article 6", "article_text": "Classification rules."}]
    answer = (
        "Your system “assists radiologists in detecting lung nodules in chest "
        "X-rays” and is therefore a medical device."
    )
    report = check_faithfulness(answer, provisions, question=question)
    assert report.unverified_count == 0
    assert report.verified_count == 1
    # Without the question, the same quote is (correctly) unverifiable
    report_no_q = check_faithfulness(answer, provisions)
    assert report_no_q.unverified_count == 1


def test_base_ref_family():
    assert _base_ref_family("Article 5(1)(f)") == "Article 5"
    assert _base_ref_family("Annex IX, Chapter I, point 3.5") == "Annex IX"
    assert _base_ref_family("Recital 44") == "Recital 44"
    assert _base_ref_family("article 43(4)") == "Article 43"


def test_unique_grounding_collapses_one_family_to_most_specific():
    from application._faithfulness import _normalize

    src = _normalize(_ART5_POINT_F)   # source_map holds normalized texts
    source_map = {
        "Article 5": "prefix " + src,
        "Article 5(1)": src,
    }
    assert _unique_grounding_ref(_ART5_POINT_F, source_map) == "Article 5(1)"
    # Two distinct families grounding the quote → ambiguous → None
    source_map["Recital 44"] = src
    assert _unique_grounding_ref(_ART5_POINT_F, source_map) is None
