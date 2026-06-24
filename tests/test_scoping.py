"""Unit tests for the ask-first scope-assessment module."""
from __future__ import annotations

from application._scoping import (
    Clarification,
    ClarificationOption,
    assess_scope,
    render_clarification_markdown,
)

_AI_ACT = "32024R1689"
_MDR = "32017R0745"
_GDPR = "32016R0679"


def _assess(question, **overrides):
    kwargs = dict(
        route_id="classification_chain",
        target_celexes={_AI_ACT, _MDR},
        role_specs=[],
        explicit_refs=[],
        is_definition_question=False,
    )
    kwargs.update(overrides)
    return assess_scope(question, **kwargs)


# ---------------------------------------------------------------------------
# When the gate SHOULD fire
# ---------------------------------------------------------------------------


def test_obligation_question_without_role_triggers_clarification():
    result = _assess(
        "What obligations apply to a Class IIb SaMD with continuous learning "
        "under MDR and the AI Act?"
    )
    assert result.needs_clarification is True
    assert result.clarification is not None
    assert result.clarification.slot == "actor_role"
    assert len(result.clarification.options) >= 2


def test_clarification_options_are_scoped_to_in_scope_regulations():
    result = _assess("What obligations apply?", target_celexes={_AI_ACT})
    assert result.needs_clarification is True
    values = {o.value for o in result.clarification.options}
    # AI Act roles present; MDR-only roles (user) absent.
    assert "provider" in values
    assert "deployer" in values
    assert "user" not in values


def test_options_capped_and_ordered_by_priority():
    result = _assess("What duties apply?", target_celexes={_AI_ACT, _MDR, _GDPR})
    opts = result.clarification.options
    assert len(opts) <= 6
    # Provider is highest priority and must lead.
    assert opts[0].value == "provider"


# ---------------------------------------------------------------------------
# When the gate should STAY SILENT
# ---------------------------------------------------------------------------


def test_no_clarification_when_role_already_present():
    result = _assess(
        "What obligations does a deployer have?",
        role_specs=[("deployer", _AI_ACT)],
    )
    assert result.needs_clarification is False


def test_no_clarification_for_definition_question():
    result = _assess("What is an AI system?", is_definition_question=True)
    assert result.needs_clarification is False


def test_no_clarification_for_explicit_provision_lookup():
    result = _assess("What does Article 43 require?", explicit_refs=["Article 43"])
    assert result.needs_clarification is False


def test_no_clarification_without_obligation_focus():
    # A pure classification question is about the system's status, not duties.
    result = _assess("Is a Class IIb SaMD high-risk under the AI Act?")
    assert result.needs_clarification is False


def test_no_clarification_when_no_regulation_in_scope():
    result = _assess("What obligations apply?", target_celexes=set())
    assert result.needs_clarification is False


def test_no_clarification_on_role_agnostic_routes():
    result = _assess(
        "What obligations exist across the corpus?",
        route_id="community_summary_search",
    )
    assert result.needs_clarification is False


def test_guidance_only_scope_does_not_fire():
    # MDCG guidance CELEX carries no actor roles -> no real options -> silent.
    result = _assess("What obligations apply?", target_celexes={"MDCG_2019_11"})
    assert result.needs_clarification is False


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_markdown_lists_options_and_rationale():
    clar = Clarification(
        slot="actor_role",
        question="Which role are you asking about?",
        rationale="Actor role is the backbone.",
        options=[
            ClarificationOption("Provider", "provider", frozenset({_AI_ACT}), "EU AI Act"),
            ClarificationOption("Manufacturer", "manufacturer", frozenset({_MDR}), "MDR"),
        ],
    )
    md = render_clarification_markdown(clar)
    assert "Which role are you asking about?" in md
    assert "Provider" in md
    assert "Manufacturer" in md
    assert "EU AI Act" in md
