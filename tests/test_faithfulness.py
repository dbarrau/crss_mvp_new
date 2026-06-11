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
    assert "article 6: high-risk ai systems" in corpus
    assert "paragraph one body." in corpus
    assert "paragraph two body." in corpus
    assert "fallback provision text." in corpus


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
