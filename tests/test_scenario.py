"""Unit tests for the detection stage (``application/scenario.py``).

``detect_scenario`` is the single source of truth for "understand the question"
— driven by both ``ask_stream`` and the retrieval net. These tests pin its
output (previously the detection stage had no direct coverage; it was only
exercised end-to-end). No Neo4j / LLM: a stub retriever with an empty
defined-term index is enough for the deterministic regex/keyword detectors.
"""
from __future__ import annotations

from application.scenario import detect_scenario, Detection
from application.contracts import Scenario
from domain.legislation_catalog import AI_ACT_CELEX as _AI_ACT


class _StubRetriever:
    """Minimal retriever: no defined terms (the regex detectors don't need them)."""

    def get_defined_terms_index(self) -> dict:
        return {}

    def find_by_term(self, _term: str) -> list:
        return []


def _detect(question: str, k: int = 20) -> Detection:
    return detect_scenario(question, _StubRetriever(), k)


def test_detection_returns_internally_consistent_scenario():
    # The typed Scenario contract must mirror the loose detection locals exactly
    # — this is the invariant that lets every consumer read either form.
    det = _detect("What are the obligations of a provider under the EU AI Act?")
    s = det.scenario
    assert isinstance(s, Scenario)
    assert s.question == "What are the obligations of a provider under the EU AI Act?"
    assert s.mentioned_regs == frozenset(det.mentioned_regs)
    assert s.target_celexes == frozenset(det.target_celexes or ())
    assert s.role_specs == tuple(det.role_specs)
    assert s.explicit_refs == tuple(det.explicit_refs)
    assert s.route_id == det.route.id
    assert s.is_definition_question == det.is_def_q


def test_ai_act_provider_question_is_in_scope_with_role():
    det = _detect("What are the obligations of a provider under the EU AI Act?")
    assert _AI_ACT in det.target_celexes
    assert det.scenario.in_scope(_AI_ACT) is True
    assert det.scenario.has_role is True  # 'provider' detected


def test_definition_question_sets_flag():
    det = _detect("What is a medical device?")
    assert det.is_def_q is True
    assert det.scenario.is_definition_question is True


def test_explicit_provision_ref_is_extracted():
    det = _detect("What does Article 6 of the EU AI Act require?")
    assert any("Article 6" in r for r in det.explicit_refs)


def test_no_regulation_leaves_target_celexes_none():
    # target_celexes stays None (not an empty set) when nothing is in scope —
    # the exact type the downstream stages branch on.
    det = _detect("What general principles should I keep in mind?")
    assert det.target_celexes is None
    assert det.scenario.target_celexes == frozenset()
    assert det.scenario.in_scope(_AI_ACT) is False


def test_k_is_never_reduced():
    # The per-regulation budget bump can only raise k, never lower the caller's.
    det = _detect("What are the obligations of a provider under the EU AI Act?", k=20)
    assert det.k >= 20


def test_context_anchor_rides_separate_channel_not_explicit_refs():
    # A wellbeing/MDR qualification question yields a decisive context anchor
    # (MDR Annex XVI) that must NOT land in explicit_refs — doing so would flip
    # the route to a narrow provision_lookup. It rides context_anchor_refs and the
    # route stays broad; retrieval merges the anchor regardless of route.
    det = _detect(
        "Our wellbeing app is not for medical use — is it a device under MDR 2017/745?"
    )
    assert "Annex XVI" in det.context_anchor_refs
    assert "Annex XVI" not in det.explicit_refs
    assert det.route.id != "provision_lookup"


def test_cdss_question_anchors_classification_annex():
    det = _detect(
        "Is standalone clinical decision-support software a medical device under MDR 2017/745?"
    )
    assert "Annex VIII" in det.context_anchor_refs
    assert "Annex VIII" not in det.explicit_refs


def test_no_context_anchor_for_unrelated_question():
    det = _detect("What are the obligations of a provider under the EU AI Act?")
    assert det.context_anchor_refs == []


def test_timing_question_anchors_article_113_across_apply_synonyms():
    # AI Act Article 113 is the sole authority for application dates, and its
    # Omnibus amendment carries the DEFERRED high-risk dates (2 Dec 2027 / 2 Aug
    # 2028) that supersede the model's pre-Omnibus training memory. A date
    # question is as often phrased "enforced / in force / comes into force" as
    # "apply" — every synonym must force-load Article 113, or the amended dates
    # never reach the model and it answers the stale pre-Omnibus dates.
    for q in (
        "When will high-risk AI systems under the EU AI Act be enforced for route I and route II?",
        "What is the enforcement date for high-risk AI systems under the EU AI Act?",
        "When do the high-risk obligations of the EU AI Act come into force?",
        "When does the EU AI Act apply?",
    ):
        assert "Article 113" in _detect(q).context_anchor_refs, q


def test_timing_anchor_stays_off_non_timing_ai_act_questions():
    # The cue must not fire on ordinary obligation/classification questions.
    assert "Article 113" not in _detect(
        "What are the obligations of a provider under Article 16 of the EU AI Act?"
    ).context_anchor_refs
    assert "Article 113" not in _detect(
        "Is my medical device with an integrated LLM high-risk under the EU AI Act?"
    ).context_anchor_refs