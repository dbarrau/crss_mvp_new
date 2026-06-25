"""Unit tests for the faithfulness verification module."""
from __future__ import annotations

import pytest

from application._faithfulness import (
    FaithfulnessReport,
    Quote,
    build_warning_block,
    check_faithfulness,
    extract_citation_refs,
    extract_context_refs,
    extract_quotes,
    faithfulness_mode,
    grounding_verdict,
    out_of_scope_citation_refs,
    remove_unverified_quotes,
    verify_quote,
    _build_corpus,
    _normalize,
)


# ---------------------------------------------------------------------------
# extract_quotes
# ---------------------------------------------------------------------------


def test_extract_quotes_picks_up_straight_smart_and_italic_wrapped_forms():
    answer = (
        'The MDR states that "manufacturers shall establish a post-market '
        'surveillance system proportionate to the risk class." Then the AI '
        "Act says \u201Cproviders of high-risk AI systems shall implement a "
        "risk management system throughout the lifecycle.\u201D Finally, "
        '*"deployers shall ensure that input data is relevant and '
        'sufficiently representative."*'
    )
    quotes = extract_quotes(answer)
    assert len(quotes) == 3
    assert "post-market surveillance" in quotes[0].text
    assert "high-risk AI systems" in quotes[1].text
    assert "deployers shall ensure" in quotes[2].text


def test_extract_quotes_ignores_quotes_below_threshold():
    # 20-char quote -> dropped; 60-char quote -> kept
    answer = (
        'The term "manufacturer" is short. '
        'But "providers of high-risk AI systems shall implement risk management." '
        "matters."
    )
    quotes = extract_quotes(answer)
    assert len(quotes) == 1
    assert "providers of high-risk" in quotes[0].text


def test_extract_quotes_ignores_single_quotes():
    answer = (
        "The provider's obligations under 'Article 16' include 'risk "
        "management throughout the entire lifecycle of the system'."
    )
    quotes = extract_quotes(answer)
    assert quotes == []


def test_quote_preview_truncates_long_text_with_ellipsis():
    long_text = "a" * 200
    q = Quote(text=long_text, start=0, end=200)
    preview = q.preview
    assert preview.endswith("...")
    assert len(preview) <= 120


# ---------------------------------------------------------------------------
# verify_quote / _normalize
# ---------------------------------------------------------------------------


def test_normalize_collapses_whitespace_and_smart_quotes():
    raw = "Providers   shall\timplement\n\nrisk management."
    assert _normalize(raw) == "providers shall implement risk management."


def test_verify_quote_accepts_whitespace_and_case_differences():
    corpus = _normalize(
        "Providers of high-risk AI systems shall implement a risk "
        "management system throughout the lifecycle."
    )
    quote = (
        "Providers   of high-risk AI systems\nshall implement a risk "
        "management system throughout the lifecycle."
    )
    assert verify_quote(quote, corpus) is True


def test_verify_quote_accepts_markdown_emphasis_inside_quote():
    corpus = _normalize(
        "The placing on the market, the putting into service or the use of "
        "AI systems that create or expand facial recognition databases "
        "through untargeted scraping is prohibited."
    )
    quote = (
        "The placing on the market, the putting into service or the use of "
        "AI systems that **create or expand facial recognition databases** "
        "through untargeted scraping is prohibited."
    )
    assert verify_quote(quote, corpus) is True


def test_verify_quote_tolerates_hyphen_vs_space_orthography():
    """An LLM 'tidying' a verbatim quote (machine readable -> machine-readable)
    must not turn a grounded operational obligation into a false flag."""
    corpus = _normalize(
        "The provider shall draw up a written machine readable, physical or "
        "electronically signed EU declaration of conformity for each high-risk "
        "AI system."
    )
    quote = (
        "The provider shall draw up a written machine-readable, physical or "
        "electronically signed EU declaration of conformity"
    )
    assert verify_quote(quote, corpus) is True


def test_verify_quote_tolerates_apostrophe_glyph_variants():
    corpus = _normalize(
        "‘AI system’ means a machine-based system that is designed to "
        "operate with varying levels of autonomy."
    )
    # Model emits a straight ASCII apostrophe instead of the typographic glyph.
    quote = "'AI system' means a machine-based system that is designed to operate"
    assert verify_quote(quote, corpus) is True


def test_verify_quote_still_rejects_fabrication_after_orthographic_folding():
    """Dash/apostrophe folding must not let a genuinely fabricated quote pass."""
    corpus = _normalize(
        "Annex III, Point 1: remote biometric identification systems."
    )
    fabricated = (
        "AI systems intended to be used for the analysis of medical data for "
        "diagnostic and treatment purposes."
    )
    assert verify_quote(fabricated, corpus) is False


def test_verify_quote_handles_ellipsis_split():
    corpus = _normalize(
        "Manufacturers shall establish, document, implement and maintain a "
        "post-market surveillance system proportionate to the risk class of "
        "the device."
    )
    # Single ellipsis splits into two long, individually-grounded fragments.
    quote = (
        "Manufacturers shall establish, document, implement and maintain "
        "[...] proportionate to the risk class of the device."
    )
    assert verify_quote(quote, corpus) is True


def test_verify_quote_rejects_fabricated_content():
    # Corpus represents the actual AI Act Annex III, Point 1 (biometrics).
    corpus = _normalize(
        "Annex III, Point 1: Biometrics, in so far as their use is permitted "
        "under relevant Union or national law, namely remote biometric "
        "identification systems."
    )
    # Fabricated quote attributing medical-diagnosis content to Annex III.
    fabricated = (
        "Annex III, Point 1(a) covers AI systems intended for medical "
        "diagnosis of patients in clinical settings."
    )
    assert verify_quote(fabricated, corpus) is False


# ---------------------------------------------------------------------------
# _build_corpus
# ---------------------------------------------------------------------------


def test_build_corpus_includes_article_text_and_children():
    provisions = [
        {
            "article_text": "Article 6: High-risk AI systems",
            "children": [
                {"raw_text": "Paragraph one body."},
                {"text": "Paragraph two body."},
            ],
        },
        {"text": "Fallback provision text."},
    ]
    corpus = _build_corpus(provisions)
    # Dashes fold to spaces during normalization (high-risk -> high risk).
    assert "article 6: high risk ai systems" in corpus
    assert "paragraph one body." in corpus
    assert "paragraph two body." in corpus
    assert "fallback provision text." in corpus


def test_build_corpus_includes_interpretive_guidance_links():
    provisions = [
        {
            "article_text": "Article 52: GPAI transparency.",
            "children": [],
            "interpreting_guidance": [
                {"ref": "MDCG 2025-6", "text": "Satisfying AI Act risk management "
                 "does not substitute for a GDPR DPIA."},
            ],
            "interpreted_provisions": [
                {"ref": "Article 9", "text": "Risk management system text."},
            ],
        },
    ]
    corpus = _build_corpus(provisions)
    assert "does not substitute for a gdpr dpia" in corpus
    assert "risk management system text." in corpus


def test_build_corpus_includes_definitions_block():
    provisions = [{"article_text": "Some obligation text."}]
    definitions = [
        {
            "term": "AI system",
            "definition_text": (
                "'AI system' means a machine-based system that is designed to "
                "operate with varying levels of autonomy."
            ),
        },
    ]
    corpus = _build_corpus(provisions, definitions)
    # Dashes fold to spaces during normalization (machine-based -> machine based).
    assert "machine based system" in corpus


def test_check_faithfulness_verifies_quote_drawn_from_definition():
    """A verbatim quote of a definition must verify once the definitions block
    is part of the corpus (regression: the AI Act Article 3(1) definition quote
    used to be falsely flagged because definitions were excluded)."""
    provisions = [{"article_text": "Unrelated obligation text about logging."}]
    definitions = [
        {
            "term": "AI system",
            "definition_text": (
                "'AI system' means a machine-based system that is designed to "
                "operate with varying levels of autonomy and that may exhibit "
                "adaptiveness after deployment."
            ),
        },
    ]
    answer = (
        "Per Article 3(1): “'AI system' means a machine-based system that "
        "is designed to operate with varying levels of autonomy and that may "
        "exhibit adaptiveness after deployment.”"
    )
    report = check_faithfulness(answer, provisions, definitions)
    assert report.ok is True
    assert report.verified_count == 1


def test_check_faithfulness_flags_training_memory_quote_not_in_context():
    """A quote pulled from training memory (not in provisions or definitions)
    is flagged even when it carries a plausible provision label — the v3 MDR
    Annex II 6.1 leak."""
    provisions = [{"article_text": "Manufacturers shall draw up technical documentation."}]
    definitions = []
    answer = (
        "The MDR requires: “A description of any software updates, including "
        "the procedure for the validation of the updated software, and the impact "
        "of the update on the device performance.” [MDR Annex II, point 6.1]"
    )
    report = check_faithfulness(answer, provisions, definitions)
    assert report.ok is False
    assert report.unverified_count == 1


# ---------------------------------------------------------------------------
# check_faithfulness
# ---------------------------------------------------------------------------


def test_check_faithfulness_returns_clean_report_when_all_quotes_grounded():
    provisions = [
        {
            "article_text": (
                "Providers of high-risk AI systems shall implement a risk "
                "management system throughout the lifecycle of the system."
            )
        }
    ]
    answer = (
        'Per the AI Act, "providers of high-risk AI systems shall implement '
        'a risk management system throughout the lifecycle of the system."'
    )
    report = check_faithfulness(answer, provisions)
    assert report.ok is True
    assert report.total_quotes == 1
    assert report.verified_count == 1
    assert report.unverified_count == 0


def test_check_faithfulness_flags_fabricated_quote():
    provisions = [
        {
            "article_text": (
                "Manufacturers shall establish a post-market surveillance "
                "system proportionate to the risk class of the device."
            )
        }
    ]
    answer = (
        'The MDR requires that "manufacturers shall establish a post-market '
        'surveillance system proportionate to the risk class of the device." '
        'However, "Annex III, Point 1(a) covers AI systems intended for '
        'medical diagnosis of patients in clinical settings."'
    )
    report = check_faithfulness(answer, provisions)
    assert report.ok is False
    assert report.total_quotes == 2
    assert report.verified_count == 1
    assert report.unverified_count == 1
    assert "Annex III, Point 1(a) covers" in report.unverified[0].text


def test_check_faithfulness_with_no_quotes_returns_clean_empty_report():
    report = check_faithfulness("An answer without any quotations at all.", [])
    assert report.ok is True
    assert report.total_quotes == 0


# ---------------------------------------------------------------------------
# Graduated grounding verdict (exact / near / absent)
# ---------------------------------------------------------------------------


def test_grounding_verdict_exact_for_verbatim_quote():
    corpus = _normalize(
        "Providers of high-risk AI systems shall implement a risk management "
        "system throughout the lifecycle of the system."
    )
    quote = (
        "Providers of high-risk AI systems shall implement a risk management "
        "system throughout the lifecycle of the system."
    )
    assert grounding_verdict(quote, corpus) == "exact"


def test_grounding_verdict_near_for_dropped_article():
    """A dropped 'the' is near-verbatim, not a fabrication."""
    corpus = _normalize(
        "that, for explicit or implicit objectives, infers, from the input it "
        "receives, how to generate outputs such as predictions, content, "
        "recommendations, or decisions."
    )
    quote = (
        "infers, from input it receives, how to generate outputs such as "
        "predictions, content, recommendations, or decisions."
    )
    assert grounding_verdict(quote, corpus) == "near"
    assert verify_quote(quote, corpus) is True


def test_grounding_verdict_absent_for_fabricated_clause():
    corpus = _normalize(
        "Post-market monitoring shall be based on a post-market monitoring plan."
    )
    quote = (
        "AI systems intended to be used for the analysis of medical data for "
        "diagnostic, prognostic, or treatment purposes in clinical settings."
    )
    assert grounding_verdict(quote, corpus) == "absent"


def test_near_verbatim_quote_is_kept_not_redacted():
    provisions = [{
        "article_text": (
            "For high-risk AI systems which are safety components of devices "
            "covered by Regulations (EU) 2017/745 and (EU) 2017/746, the "
            "notification of serious incidents shall be limited to those "
            "referred to in Article 3, point 49(c)."
        ),
    }]
    # Model reworded slightly ('which are safety components of devices' ->
    # 'that are safety components of devices', dropped the regulation cite).
    answer = (
        "Per Article 73(10): “For high-risk AI systems that are safety "
        "components of devices, the notification of serious incidents shall be "
        "limited to those referred to in Article 3, point 49(c).”"
    )
    report = check_faithfulness(answer, provisions)
    assert report.unverified_count == 0          # not treated as fabrication
    assert report.near_verbatim_count == 1       # surfaced for wording check
    assert report.ok is True
    # The grounded quote survives redaction.
    kept = remove_unverified_quotes(answer, report)
    assert "notification of serious incidents shall be limited" in kept


def test_warning_block_separates_removed_from_near_verbatim():
    removed = Quote("Annex III point 1(a) covers medical diagnosis in clinics.", 0, 56)
    near = Quote("infers from input it receives how to generate outputs.", 60, 114)
    report = FaithfulnessReport(
        total_quotes=2, verified=[], unverified=[removed], near_verbatim=[near],
    )
    block = build_warning_block(report)
    assert block is not None
    assert "FAITHFULNESS FLAG" in block          # loud tier for removed
    assert "Wording check" in block               # light tier for near-verbatim
    for line in block.splitlines():
        assert line.startswith(">")


# ---------------------------------------------------------------------------
# build_warning_block
# ---------------------------------------------------------------------------


def test_build_warning_block_returns_none_on_clean_report():
    report = FaithfulnessReport(total_quotes=3, verified=[Quote("x" * 60, 0, 60)] * 3)
    assert build_warning_block(report) is None


def test_build_warning_block_lists_unverified_quotes_with_preview():
    fabricated = Quote(
        text="Annex III, Point 1(a) covers AI systems intended for medical diagnosis.",
        start=0,
        end=70,
    )
    report = FaithfulnessReport(total_quotes=2, verified=[], unverified=[fabricated])
    block = build_warning_block(report)
    assert block is not None
    assert "FAITHFULNESS FLAG" in block
    assert "1 of 2" in block
    assert "Annex III, Point 1(a)" in block
    # Block is markdown blockquote
    for line in block.splitlines():
        assert line.startswith(">")


def test_remove_unverified_quotes_removes_only_flagged_spans():
    answer = (
        'Grounded: "providers shall implement risk management throughout the lifecycle." '
        'Ungrounded: "Annex III point 1(a) covers medical diagnosis in clinical settings." '
        "Final sentence."
    )
    quotes = extract_quotes(answer)
    assert len(quotes) == 2
    report = FaithfulnessReport(
        total_quotes=2,
        verified=[quotes[0]],
        unverified=[quotes[1]],
    )
    redacted = remove_unverified_quotes(answer, report)
    assert "Annex III point 1(a) covers medical diagnosis" not in redacted
    assert "Final sentence." in redacted


# ---------------------------------------------------------------------------
# Structural guards: concatenation + misattribution
# ---------------------------------------------------------------------------


_STRUCTURAL_PROVISIONS = [
    {
        "article_ref": "Article 43",
        "article_text": (
            "Where a high-risk AI system is to undergo a substantial "
            "modification, the high-risk AI system shall undergo a new "
            "conformity assessment procedure."
        ),
    },
    {
        "article_ref": "Article 15",
        "article_text": (
            "High-risk AI systems shall be designed and developed in such a "
            "way that they achieve an appropriate level of accuracy, robustness "
            "and cybersecurity."
        ),
    },
    {
        "article_ref": "Article 47",
        "article_text": (
            "The provider shall draw up a written EU declaration of conformity "
            "and keep it at the disposal of the national competent authorities."
        ),
    },
]


def test_concatenation_blob_attributed_to_one_article_is_flagged_and_removed():
    # The v7 failure mode: a single quote concatenating the verbatim text of
    # three distinct provisions, all jammed under one [Article 43] citation.
    # Each fragment is individually grounded, so grounding alone would pass it.
    answer = (
        "Under [Article 43 AI Act] the duties are: "
        '"Where a high-risk AI system is to undergo a substantial modification, '
        "the high-risk AI system shall undergo a new conformity assessment "
        "procedure. High-risk AI systems shall be designed and developed in such "
        "a way that they achieve an appropriate level of accuracy, robustness and "
        "cybersecurity. The provider shall draw up a written EU declaration of "
        'conformity and keep it at the disposal of the national competent authorities."'
    )
    report = check_faithfulness(answer, _STRUCTURAL_PROVISIONS)
    assert report.misattributed_count == 1
    assert report.verified_count == 0
    assert report.ok is False
    redacted = remove_unverified_quotes(answer, report)
    assert "EU declaration of conformity" not in redacted


def test_real_text_under_wrong_citation_is_flagged_as_misattributed():
    # Real Article 47 text, but the answer attributes it to [Article 15].
    answer = (
        "The accuracy duty in [Article 15 AI Act] requires that "
        '"The provider shall draw up a written EU declaration of conformity and '
        'keep it at the disposal of the national competent authorities."'
    )
    report = check_faithfulness(answer, _STRUCTURAL_PROVISIONS)
    assert report.misattributed_count == 1
    assert report.verified_count == 0


def test_correctly_cited_quote_passes_structural_guards():
    answer = (
        'Per [Article 15 AI Act], "High-risk AI systems shall be designed and '
        "developed in such a way that they achieve an appropriate level of "
        'accuracy, robustness and cybersecurity."'
    )
    report = check_faithfulness(answer, _STRUCTURAL_PROVISIONS)
    assert report.verified_count == 1
    assert report.misattributed_count == 0
    assert report.ok is True


def test_misattribution_silent_when_cited_provision_not_retrieved():
    # If the cited provision was never retrieved we cannot adjudicate
    # attribution — the quote must not be falsely flagged as misattributed.
    answer = (
        "As set out in [Article 99 AI Act], "
        '"High-risk AI systems shall be designed and developed in such a way '
        "that they achieve an appropriate level of accuracy, robustness and "
        'cybersecurity."'
    )
    report = check_faithfulness(answer, _STRUCTURAL_PROVISIONS)
    assert report.misattributed_count == 0
    assert report.verified_count == 1


def test_attribution_flag_distinct_from_fabrication_flag_in_warning_block():
    blob = Quote("x" * 250, 0, 250)
    report = FaithfulnessReport(total_quotes=1, misattributed=[blob])
    block = build_warning_block(report)
    assert block is not None
    assert "ATTRIBUTION FLAG" in block
    assert "FAITHFULNESS FLAG" not in block       # not a fabrication
    for line in block.splitlines():
        assert line.startswith(">")


def test_extract_citation_refs_detects_articles_and_annexes():
    answer = "See Article 16(a), Article 53(1)(b), Annex xi, and Recital 178."
    refs = extract_citation_refs(answer)
    assert "Article 16(a)" in refs
    assert "Article 53(1)(b)" in refs
    assert "Annex XI" in refs
    assert "Recital 178" in refs


def test_out_of_scope_citation_refs_deterministic_against_context():
    answer = "Relevant refs: Article 16, Article 53, Article 999, Annex XI, Annex XV."
    provisions = [
        {"article_ref": "Article 16", "children": []},
        {"article_ref": "Article 53", "children": [{"ref": "Annex XI"}]},
    ]
    ctx = extract_context_refs(provisions)
    assert "Article 16" in ctx
    assert "Article 53" in ctx
    assert "Annex XI" in ctx
    missing = out_of_scope_citation_refs(answer, provisions)
    assert "Article 999" in missing
    assert "Annex XV" in missing
    assert "Article 16" not in missing


def test_out_of_scope_citation_refs_treats_nested_articles_as_in_scope_when_parent_present():
    answer = "See Article 5(1)(a), Article 5(1)(h), and Article 79(2)."
    provisions = [
        {"article_ref": "Article 5", "children": []},
        {"article_ref": "Article 79", "children": []},
    ]
    missing = out_of_scope_citation_refs(answer, provisions)
    assert "Article 5(1)(a)" not in missing
    assert "Article 5(1)(h)" not in missing
    assert "Article 79(2)" not in missing


# ---------------------------------------------------------------------------
# faithfulness_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_value,expected",
    [
        (None, 0),
        ("", 0),
        ("0", 0),
        ("off", 0),
        ("false", 0),
        ("no", 0),
        ("1", 1),
        ("flag", 1),
        ("true", 1),
        ("on", 1),
        ("2", 2),
        ("strict", 2),
        ("STRICT", 2),
        ("unknown", 0),
        ("3", 0),
    ],
)
def test_faithfulness_mode_parses_env_values(env_value, expected):
    assert faithfulness_mode(env_value) == expected


# ---------------------------------------------------------------------------
# Redaction-corruption regression (a stray quote glyph used to swallow analysis
# across newlines; removing it by offset fused the surrounding words).
# ---------------------------------------------------------------------------

from application._faithfulness import (  # noqa: E402
    extract_quotes as _extract_quotes,
    remove_unverified_quotes as _remove_unverified_quotes,
    FaithfulnessReport as _FR,
)


def test_quote_extraction_never_spans_newlines_or_runs_away():
    answer = (
        'Article 2 MDR ("medical device includes software for a medical purpose).\n'
        '- **AI Act**: the same entity is the provider of the AI system" then more).'
    )
    quotes = _extract_quotes(answer)
    assert all("\n" not in q.text for q in quotes)
    assert all(len(q.text) <= 600 for q in quotes)


def test_oversized_single_line_quote_is_skipped():
    answer = 'The rule states: "' + ("x" * 700) + '".'
    assert _extract_quotes(answer) == []


def test_removal_inserts_marker_instead_of_fusing_words():
    answer = 'under the AI Act"fabricated clause that is well over forty characters in length"Actor role'
    quotes = _extract_quotes(answer)
    assert len(quotes) == 1
    out = _remove_unverified_quotes(answer, _FR(total_quotes=1, unverified=quotes))
    assert "ActActor" not in out      # words not fused
    assert "[…]" in out               # marker preserves the boundary
