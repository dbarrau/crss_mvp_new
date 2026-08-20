"""Obligation-scope answer discipline for AI Act actor-obligation questions.

Actor status under the AI Act does not by itself create obligations — Article
26/27 deployer duties apply only to deployers of HIGH-RISK systems, and Article
50 only to specific system types. The guidance fires (content-triggered, any
route) when the AI Act is in scope and the question names an actor role, so the
model states obligations conditionally instead of "deployer ⇒ all obligations".
"""
from application._prompts import _build_obligation_scope_guidance


def test_fires_for_ai_act_deployer_question():
    g = _build_obligation_scope_guidance(
        "Is a university department using a third-party AI system a deployer with obligations?",
        {"EU AI Act"},
    )
    assert g is not None
    # It must push the conditional framing and name the decisive gates.
    assert "CONDITIONAL" in g
    assert "high-risk" in g.lower() and "Article 6" in g
    assert "Article 50" in g
    assert "Article 26" in g


def test_silent_without_ai_act_in_scope():
    # A deployer/manufacturer question about a different framework must not get
    # the AI-Act-specific classification discipline.
    assert _build_obligation_scope_guidance(
        "Is the hospital a manufacturer with obligations under the MDR?",
        {"MDR 2017/745"},
    ) is None


def test_silent_when_no_actor_role_named():
    # AI Act in scope but no actor role → not an actor-obligation question.
    assert _build_obligation_scope_guidance(
        "What is a high-risk AI system under the AI Act?",
        {"EU AI Act"},
    ) is None


def test_silent_when_no_regs():
    assert _build_obligation_scope_guidance("Is a deployer subject to obligations?", None) is None
