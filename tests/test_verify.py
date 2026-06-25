"""Deterministic tests for the unified post-generation verification stage.

The pre-fold pipeline ran citation-scope, faithfulness/attribution, and
confidence as three scattered inline blocks in ``ask_stream`` with **zero**
direct test coverage on the orchestration. ``verify_answer`` is a pure function,
so these pin its behaviour against fixed answers + evidence (no LLM, no Neo4j).
"""
import pytest

from application.verify import verify_answer, VerificationResult


_AI_ACT = "32024R1689"
_GDPR = "32016R0679"

_GROUNDED = "High-risk AI systems shall be subject to a conformity assessment."
_FABRICATED = (
    "All AI systems are hereby permanently banned across the entire Union "
    "without exception or transitional relief."
)


def _kwargs(**over):
    base = dict(
        provisions=[
            {
                "article_text": _GROUNDED,
                "article_ref": "Article 6",
                "celex": _AI_ACT,
                "children": [],
                "binding_force": "binding",
            }
        ],
        definitions=[],
        role_provisions=[],
        sufficiency={"checks": [], "ok": True},
        target_celexes={_AI_ACT},
        mentioned_regs={"AI Act"},
        role_specs=[],
        corrective_actions=[],
        question="What are the obligations for high-risk AI systems?",
    )
    base.update(over)
    return base


def test_verify_returns_confidence_with_expected_shape():
    res = verify_answer("High-risk AI systems must undergo conformity assessment.", **_kwargs())
    assert isinstance(res, VerificationResult)
    c = res.confidence
    assert {"confidence_score", "confidence_level", "breakdown", "legal_force_distribution"} <= set(c)
    assert 0.0 <= c["confidence_score"] <= 1.0
    assert c["confidence_level"] in {"HIGH", "MEDIUM", "LOW", "CRITICAL"}


def test_grounded_quote_survives_unredacted(monkeypatch):
    monkeypatch.setenv("CRSS_FAITHFULNESS_CHECK", "1")
    answer = f'The rule is clear: "{_GROUNDED}"'
    res = verify_answer(answer, **_kwargs())
    assert _GROUNDED in res.answer
    assert "FAITHFULNESS FLAG" not in res.answer


def test_fabricated_quote_redacted_and_warned(monkeypatch):
    monkeypatch.setenv("CRSS_FAITHFULNESS_CHECK", "1")
    answer = f'The regulation states: "{_FABRICATED}"'
    res = verify_answer(answer, **_kwargs())
    # The in-body quotation is removed; it survives only inside the warning
    # block's "removed quotes" list (curly-quoted), which is the intended flag.
    assert f'"{_FABRICATED}"' not in res.answer     # straight-quoted in-body form gone
    assert "FAITHFULNESS FLAG" in res.answer         # warning block prepended


def test_faithfulness_flag_off_keeps_fabricated_quote(monkeypatch):
    monkeypatch.setenv("CRSS_FAITHFULNESS_CHECK", "0")
    answer = f'The regulation states: "{_FABRICATED}"'
    res = verify_answer(answer, **_kwargs())
    assert _FABRICATED in res.answer               # check disabled → no redaction
    assert "FAITHFULNESS FLAG" not in res.answer


def test_citation_scope_note_flags_out_of_scope_ref(monkeypatch):
    monkeypatch.setenv("CRSS_CITATION_SCOPE_CHECK", "1")
    monkeypatch.setenv("CRSS_FAITHFULNESS_CHECK", "0")  # isolate the scope path
    answer = "This obligation is governed by Article 99 of the framework."
    res = verify_answer(answer, **_kwargs())  # context only holds Article 6
    assert "Citation scope note" in res.answer
    assert "Article 99" in res.answer


def test_citation_scope_flag_off_suppresses_note(monkeypatch):
    monkeypatch.setenv("CRSS_CITATION_SCOPE_CHECK", "0")
    monkeypatch.setenv("CRSS_FAITHFULNESS_CHECK", "0")
    answer = "This obligation is governed by Article 99 of the framework."
    res = verify_answer(answer, **_kwargs())
    assert "Citation scope note" not in res.answer


def test_citation_scope_silent_when_multi_reg(monkeypatch):
    monkeypatch.setenv("CRSS_CITATION_SCOPE_CHECK", "1")
    monkeypatch.setenv("CRSS_FAITHFULNESS_CHECK", "0")
    answer = "This obligation is governed by Article 99 of the framework."
    res = verify_answer(
        answer,
        **_kwargs(target_celexes={_AI_ACT, _GDPR}, mentioned_regs={"AI Act", "GDPR"}),
    )
    # Scope note only fires for single-regulation questions.
    assert "Citation scope note" not in res.answer


def test_in_scope_citation_not_flagged(monkeypatch):
    monkeypatch.setenv("CRSS_CITATION_SCOPE_CHECK", "1")
    monkeypatch.setenv("CRSS_FAITHFULNESS_CHECK", "0")
    answer = "This obligation flows from Article 6 of the framework."
    res = verify_answer(answer, **_kwargs())  # Article 6 IS in context
    assert "Citation scope note" not in res.answer


# ---------------------------------------------------------------------------
# Confidence reads the pre-redaction faithfulness report (regression pin for the
# dead-component fix: the post-redaction answer is always clean, so recomputing
# there made this component a constant 1.0).
# ---------------------------------------------------------------------------


def test_confidence_faithfulness_reflects_fabrication(monkeypatch):
    monkeypatch.setenv("CRSS_FAITHFULNESS_CHECK", "1")
    answer = f'The regulation states: "{_FABRICATED}"'
    res = verify_answer(answer, **_kwargs())
    # One quote, fabricated → (total - unverified)/total = (1-1)/1 = 0.0.
    # The quote is still redacted from the body, but confidence sees the
    # pre-redaction report, so the faithfulness component is NOT a constant 1.0.
    assert res.confidence["breakdown"]["faithfulness"] == 0.0
    assert f'"{_FABRICATED}"' not in res.answer  # body still redacted


def test_confidence_faithfulness_full_for_grounded_quote(monkeypatch):
    monkeypatch.setenv("CRSS_FAITHFULNESS_CHECK", "1")
    answer = f'The rule is clear: "{_GROUNDED}"'
    res = verify_answer(answer, **_kwargs())
    assert res.confidence["breakdown"]["faithfulness"] == 1.0


def test_confidence_faithfulness_full_when_no_quotes(monkeypatch):
    monkeypatch.setenv("CRSS_FAITHFULNESS_CHECK", "1")
    res = verify_answer("A paraphrased answer with no verbatim quotations.", **_kwargs())
    # No quotes to verify → neutral 1.0 (nothing to penalise).
    assert res.confidence["breakdown"]["faithfulness"] == 1.0