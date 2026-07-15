"""Regression tests for the v6-eval forensic findings (14–15 Jul 2026).

Each fixture models the real failure that exposed the class:

- illustrative-quote tier: HQ_037's 8 drafted notification templates and
  HQ_028's 5 embellished app-marketing echoes were counted (and redacted) as
  fabricated regulatory quotes;
- cite-after attribution window: HQ_012 formats quotes as ``“…” (Article 10)``
  — the before-only window blamed the previous bullet's label and produced
  4 false positive-displacement flags;
- preamble supplement: HQ_041 asked what the GDPR preamble says, retrieval
  never surfaced Recital 43, the model quoted it from memory and was flagged.
"""
from application._faithfulness import (
    _is_illustrative,
    _nearest_citation_ref,
    check_faithfulness,
)


ART10 = (
    "Manufacturers of devices shall establish, document and maintain a risk "
    "management system as described in Section 3 of Annex I."
)
ART88 = (
    "Manufacturers shall report any statistically significant increase in the "
    "frequency or severity of incidents that are not serious incidents."
)


# ── illustrative-quote tier ─────────────────────────────────────────────────

def test_drafted_template_quote_is_illustrative_not_fabricated():
    """HQ_037: a rejection-notice template the model drafted itself."""
    provisions = [{"article_ref": "Article 26", "article_text": "Deployer duties."}]
    answer = (
        "Your Article 26 notification could state: "
        "“Your application was screened by an AI tool that evaluates CVs "
        "based on predefined criteria before human review of the outcome.”"
    )
    report = check_faithfulness(answer, provisions)
    assert report.unverified_count == 0
    assert report.illustrative_count == 1


def test_second_person_scenario_echo_is_illustrative():
    """HQ_028: embellished app-marketing wording, beyond the question's text."""
    provisions = [{"article_ref": "Article 2", "article_text": "Scope of the MDR."}]
    answer = (
        "The disclaimer matters: “This app is not a medical device and is not "
        "intended for diagnosis, treatment, or medical decision-making by you.”"
    )
    report = check_faithfulness(answer, provisions)
    assert report.unverified_count == 0
    assert report.illustrative_count == 1


def test_cited_ungrounded_quote_is_still_fabrication():
    """A quote attributed to a provision is a legal-quote claim — the
    illustrative tier must never launder it."""
    provisions = [{"article_ref": "Article 26", "article_text": "Deployer duties."}]
    answer = (
        "**Article 79** states: “Member States shall lay down rules that you "
        "must follow on penalties applicable to infringements of this "
        "Regulation by any operator.”"
    )
    report = check_faithfulness(answer, provisions)
    assert report.unverified_count == 1
    assert report.illustrative_count == 0


def test_plain_ungrounded_law_like_quote_still_fabrication():
    """No cue, no address, no citation — an ungrounded law-like quote keeps
    the loud flag."""
    provisions = [{"article_ref": "Article 26", "article_text": "Deployer duties."}]
    answer = (
        "The regulation requires that “providers shall ensure conformity of "
        "the system with the essential requirements before any placing on the market”."
    )
    report = check_faithfulness(answer, provisions)
    assert report.unverified_count == 1
    assert report.illustrative_count == 0


# ── cite-after attribution window ───────────────────────────────────────────

def test_cite_after_layout_attributes_to_trailing_label():
    answer = (
        "- “Manufacturers shall report any statistically significant increase "
        "in the frequency of incidents” (Article 88 MDR).\n"
    )
    start = answer.index("“")
    end = answer.index("”") + 1
    assert _nearest_citation_ref(answer, start, end) == "Article 88"


def test_cite_after_beats_stale_before_label_and_prevents_false_flag():
    """HQ_012: quote grounded in Article 88, trailing (Article 88) label, but
    the PREVIOUS bullet cites Article 10 — before-only adjudication displaced
    it; the nearer trailing label must win."""
    provisions = [
        {"article_ref": "Article 10", "article_text": ART10},
        {"article_ref": "Article 88", "article_text": ART88},
    ]
    answer = (
        f"- Quality management (**Article 10**): manufacturers keep a QMS.\n"
        f"- “{ART88.rstrip('.')}” (**Article 88**).\n"
    )
    report = check_faithfulness(answer, provisions)
    assert report.misattributed_count == 0
    assert report.verified_count >= 1


def test_after_window_stays_on_same_line():
    """A label on the NEXT line belongs to the next bullet — it must not be
    adopted as this quote's citation."""
    answer = (
        "Some text “a quoted span of at least forty characters to trigger "
        "extraction here” and nothing.\n(**Article 55**) The next point.\n"
    )
    start = answer.index("“")
    end = answer.index("”") + 1
    assert _nearest_citation_ref(answer, start, end) is None


def test_is_illustrative_respects_trailing_citation():
    answer = (
        "e.g. “a quoted span of at least forty characters to trigger "
        "extraction right here” (Article 12)"
    )
    start = answer.index("“")
    end = answer.index("”") + 1

    class Q:
        pass

    q = Q(); q.start = start; q.end = end; q.text = answer[start + 1 : end - 1]
    assert _is_illustrative(answer, q) is False
