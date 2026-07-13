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


def test_family_union_still_flags_cross_operative_family_misattribution():
    """A quote whose text lives in a different *operative* article than the one
    cited still flags. (Recital-only grounding is handled separately — a recital
    reciting a rule is not proof of displacement; see the recital test below.)"""
    art15 = (
        "High-risk AI systems shall achieve an appropriate level of accuracy, "
        "robustness and cybersecurity throughout their lifecycle."
    )
    provisions = [
        {"article_ref": "Article 5", "article_text": "Prohibited practices text."},
        {"article_ref": "Article 15", "article_text": art15},
    ]
    answer = f"**Article 5** states: “{art15.rstrip('.')}”"
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


def test_truncated_cited_source_does_not_flag_without_displacement_proof():
    """v3 residuals (HQ_014): verbatim Article 48(7) text, correctly cited,
    was flagged because the retrieved Article 48 source text is capped and
    stopped before paragraph (7). Absence from an incomplete copy is not
    proof of displacement — flag only when a *different* keyed family
    positively grounds the quote."""
    art48_7 = (
        "Manufacturers of class C devices, other than devices for performance "
        "study, shall be subject to a conformity assessment as specified in "
        "Chapters I and III of Annex IX."
    )
    provisions = [
        # The cited family is present but its (capped) text lacks the quote…
        {"article_ref": "Article 48", "article_text": "Truncated body: paragraphs 1-6 only."},
        # …and the quote's true home reaches the pooled corpus only via an
        # entry the citation regex cannot key (no Article/Annex/Recital ref).
        {"article_ref": "Chapter VII overview", "article_text": art48_7},
    ]
    answer = f"Class C assessment is set by **Article 48(7)**: “{art48_7.rstrip('.')}”"
    report = check_faithfulness(answer, provisions)
    assert report.misattributed_count == 0
    assert report.unverified_count == 0


def test_recital_grounding_is_not_displacement_proof():
    """v4 residuals (HQ_006, HQ_027): verbatim operative-article text, correctly
    cited, grounded ONLY in a recital because the cited article's retrieved copy
    was truncated. A recital recites the rule its article enacts — it is not a
    different provision, so it must not prove displacement."""
    art25_1c = (
        "modifies the intended purpose of an AI system, including a "
        "general-purpose AI system, which has not been classified as high-risk "
        "and has already been placed on the market, in such a manner that the "
        "AI system concerned becomes a high-risk AI system."
    )
    provisions = [
        # Cited article present but its retrieved copy is truncated (lacks (1)(c))
        {"article_ref": "Article 25", "article_text": "Article 25(1): a distributor, importer, deployer..."},
        # The quote's only complete copy is in the reciting recital
        {"article_ref": "Recital 84", "article_text": "Whereas " + art25_1c},
    ]
    answer = f"Under **Article 25(1)(c)**, a third party that “{art25_1c.rstrip('.')}” becomes a provider."
    report = check_faithfulness(answer, provisions)
    assert report.misattributed_count == 0
    assert report.unverified_count == 0


def test_genuine_operative_displacement_still_flags():
    """The fix must not blind the guard: text cited as Article 9 that actually
    lives in a different *operative* article (Article 15) still flags."""
    provisions = [
        {"article_ref": "Article 9", "article_text": "A risk management system shall be established."},
        {"article_ref": "Article 15", "article_text": (
            "High-risk AI systems shall achieve an appropriate level of accuracy, "
            "robustness and cybersecurity throughout their lifecycle."
        )},
    ]
    answer = (
        "Under **Article 9**, “High-risk AI systems shall achieve an appropriate "
        "level of accuracy, robustness and cybersecurity throughout their lifecycle.”"
    )
    report = check_faithfulness(answer, provisions)
    assert report.misattributed_count == 1


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
