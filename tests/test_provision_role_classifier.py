"""Unit tests for the provision-role classifier rules.

These tests target the pure-Python classification function (no Neo4j). They
serve as a regression suite: every time a rule is added or tightened, the
expected-classification examples here must continue to pass.

Each test uses a *realistic* sample of regulatory text, lightly trimmed.
"""
from __future__ import annotations

from domain.ontology.provision_roles import (
    PROVISION_ROLE_TAXONOMY,
    classify_provision,
)


def _classify(text: str, **kwargs):
    return classify_provision(
        text=text,
        kind=kwargs.pop("kind", "paragraph"),
        title=kwargs.pop("title", None),
        provision_id=kwargs.pop("provision_id", ""),
        celex=kwargs.pop("celex", ""),
    )


# ---------------------------------------------------------------------------
# DEFINES
# ---------------------------------------------------------------------------

def test_defines_quoted_means_straight_quotes():
    text = "'provider' means a natural or legal person who develops an AI system."
    result = _classify(text)
    assert result.role == "DEFINES"
    assert result.rule_id == "defines.quoted_means.v1"


def test_defines_quoted_means_curly_quotes():
    text = "\u2018manufacturer\u2019 means a natural or legal person who manufactures a device."
    result = _classify(text)
    assert result.role == "DEFINES"


def test_defines_inside_definitions_article_without_quotes():
    text = "deployer means a natural or legal person using an AI system under its authority."
    result = _classify(
        text,
        provision_id="32024R1689_art_3_pt_4",
        celex="32024R1689",
    )
    assert result.role == "DEFINES"
    assert result.rule_id == "defines.definitions_article.v1"


def test_defines_inside_gdpr_definitions_article():
    text = "controller means the natural or legal person which determines the purposes and means of the processing of personal data."
    result = _classify(
        text,
        provision_id="32016R0679_art_4_pt_7",
        celex="32016R0679",
    )
    assert result.role == "DEFINES"


# ---------------------------------------------------------------------------
# EXEMPTS
# ---------------------------------------------------------------------------

def test_exempts_shall_not_apply_with_scope():
    text = (
        "With the exception of the relevant general safety and performance "
        "requirements set out in Annex I, the requirements of this Regulation "
        "shall not apply to devices manufactured and used only within health "
        "institutions established in the Union."
    )
    result = _classify(text)
    assert result.role == "EXEMPTS"


def test_exempts_title():
    result = _classify("Some neutral text.", title="Derogations from certain provisions")
    assert result.role == "EXEMPTS"
    assert result.rule_id == "exempts.title.v1"


def test_exempts_does_not_match_pure_shall():
    text = "The provider shall ensure compliance with the requirements."
    result = _classify(text)
    assert result.role != "EXEMPTS"


# ---------------------------------------------------------------------------
# EXTENDS_STATUS
# ---------------------------------------------------------------------------

def test_extends_status_deemed_provider():
    text = (
        "Any distributor, importer, deployer or other third-party shall be "
        "considered to be a provider of a high-risk AI system for the purposes "
        "of this Regulation if they put their name or trademark on the system."
    )
    result = _classify(text)
    assert result.role == "EXTENDS_STATUS"


def test_extends_status_deemed_manufacturer():
    text = "A natural or legal person shall be deemed a manufacturer where it places a device on the market under its own name."
    result = _classify(text)
    assert result.role == "EXTENDS_STATUS"


# ---------------------------------------------------------------------------
# PROHIBITION
# ---------------------------------------------------------------------------

def test_prohibition_is_prohibited():
    text = "The placing on the market of AI systems for social scoring is prohibited."
    result = _classify(text)
    assert result.role == "PROHIBITION"


def test_prohibition_shall_not_make_available():
    text = "Distributors shall not make available on the market a device that does not comply with the requirements."
    result = _classify(text)
    assert result.role == "PROHIBITION"


# ---------------------------------------------------------------------------
# OBLIGATION
# ---------------------------------------------------------------------------

def test_obligation_providers_shall():
    text = "Providers of high-risk AI systems shall ensure that their systems undergo the conformity assessment."
    result = _classify(text)
    assert result.role == "OBLIGATION"


def test_obligation_notified_body_shall():
    text = "The notified body shall verify the conformity of the device with the applicable requirements."
    result = _classify(text)
    assert result.role == "OBLIGATION"


def test_obligation_member_states_shall():
    text = "Member States shall designate one or more competent authorities to enforce this Regulation."
    result = _classify(text)
    assert result.role == "OBLIGATION"


def test_obligation_does_not_misfire_on_shall_not_apply():
    text = "This Article shall not apply to research conducted prior to entry into force."
    result = _classify(text)
    assert result.role != "OBLIGATION"


# ---------------------------------------------------------------------------
# SCOPE
# ---------------------------------------------------------------------------

def test_scope_title():
    result = _classify("Some text.", title="Scope")
    assert result.role == "SCOPE"


def test_scope_opening_phrase():
    text = "This Regulation applies to providers placing on the market AI systems in the Union."
    result = _classify(text)
    assert result.role == "SCOPE"


def test_scope_opening_with_leading_number():
    text = "1. This Regulation applies to providers placing on the market AI systems in the Union."
    result = _classify(text)
    assert result.role == "SCOPE"


# ---------------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------------

def test_classification_class_assignment():
    text = "Software intended to provide information used to take decisions with diagnosis purposes is classified as class IIa."
    result = _classify(text)
    assert result.role == "CLASSIFICATION"


def test_classification_rule_title():
    result = _classify("Some text about device classes.", title="Rule 11")
    assert result.role == "CLASSIFICATION"


def test_classification_deemed_high_risk_article_6_1():
    """AIA Art 6(1): 'AI system shall be considered to be high-risk where...'.
    Without this rule the most decisive classification anchor in the AI Act
    falls through to UNCLASSIFIED."""
    text = (
        "Irrespective of whether an AI system is placed on the market or put "
        "into service independently of the products referred to in points (a) "
        "and (b), that AI system shall be considered to be high-risk where "
        "both of the following conditions are fulfilled:"
    )
    result = _classify(text)
    assert result.role == "CLASSIFICATION"
    assert result.rule_id == "classification.deemed_high_risk.v1"


def test_classification_deemed_high_risk_article_6_2():
    """AIA Art 6(2): Annex III route to high-risk classification."""
    text = (
        "In addition to the high-risk AI systems referred to in paragraph 1, "
        "AI systems referred to in Annex III shall be considered to be high-risk."
    )
    result = _classify(text)
    assert result.role == "CLASSIFICATION"
    assert result.rule_id == "classification.deemed_high_risk.v1"


def test_classification_deemed_high_risk_excludes_negation():
    """AIA Art 6(3): 'shall NOT be considered to be high-risk' must NOT be
    tagged CLASSIFICATION — it is a carve-out (EXEMPTS family). The rule
    relies on adjacency between 'shall' and 'be' to exclude the negation."""
    text = (
        "By derogation from paragraph 2, an AI system referred to in Annex III "
        "shall not be considered to be high-risk where it does not pose a "
        "significant risk of harm to the health, safety or fundamental rights "
        "of natural persons."
    )
    result = _classify(text)
    assert result.role != "CLASSIFICATION"


def test_classification_presumed_high_impact_capabilities():
    """AIA Art 51(2): the 10^25 FLOPs presumption triggers systemic-risk
    classification of a general-purpose AI model."""
    text = (
        "A general-purpose AI model shall be presumed to have high impact "
        "capabilities pursuant to paragraph 1, point (a), when the cumulative "
        "amount of computation used for its training measured in floating "
        "point operations is greater than 10^25."
    )
    result = _classify(text)
    assert result.role == "CLASSIFICATION"
    assert result.rule_id == "classification.presumed_capability.v1"


# ---------------------------------------------------------------------------
# PROCEDURAL
# ---------------------------------------------------------------------------

def test_procedural_title():
    result = _classify("Some neutral text.", title="Conformity assessment procedure")
    assert result.role == "PROCEDURAL"


def test_procedural_phrase_in_text():
    text = "The provider shall undergo the conformity assessment procedure as set out in Annex VII."
    # OBLIGATION will win here because order matters: PROCEDURAL phrase needs
    # to lose to OBLIGATION when an actor+shall is present. This is intended:
    # the text imposes a duty, the procedure name is incidental.
    result = _classify(text)
    assert result.role == "OBLIGATION"


# ---------------------------------------------------------------------------
# PENALTY
# ---------------------------------------------------------------------------

def test_penalty_title():
    result = _classify("Some neutral text.", title="Penalties")
    assert result.role == "PENALTY"


def test_penalty_administrative_fines():
    text = "Non-compliance with this Article shall be subject to administrative fines of up to EUR 35 000 000."
    result = _classify(text)
    assert result.role == "PENALTY"


def test_penalty_member_states_lay_down():
    text = "Member States shall lay down the rules on penalties applicable to infringements of this Regulation."
    result = _classify(text)
    assert result.role == "PENALTY"  # PENALTY runs before OBLIGATION


# ---------------------------------------------------------------------------
# Kind-based pre-classification
# ---------------------------------------------------------------------------

def test_structural_kind_chapter():
    result = _classify("Anything.", kind="chapter")
    assert result.role == "STRUCTURAL"


def test_structural_kind_annex_section():
    result = _classify("Anything.", kind="annex_section")
    assert result.role == "STRUCTURAL"


def test_interpretive_kind_recital():
    text = "Whereas providers should be considered responsible for ensuring compliance."
    result = _classify(text, kind="recital")
    assert result.role == "INTERPRETIVE"


def test_interpretive_kind_citation():
    text = "Having regard to the proposal from the European Commission,"
    result = _classify(text, kind="citation")
    assert result.role == "INTERPRETIVE"


# ---------------------------------------------------------------------------
# UNCLASSIFIED fallback
# ---------------------------------------------------------------------------

def test_unclassified_when_no_rule_matches():
    text = "Annex IV contains the technical documentation requirements."
    result = _classify(text)
    assert result.role == "UNCLASSIFIED"


def test_unclassified_empty_text():
    result = _classify("")
    assert result.role == "UNCLASSIFIED"


# ---------------------------------------------------------------------------
# Taxonomy invariant
# ---------------------------------------------------------------------------

def test_classifier_always_returns_role_from_taxonomy():
    samples = [
        "Hello world.",
        "'X' means Y.",
        "shall not apply to research devices",
        "shall be considered to be a provider",
        "is prohibited",
        "providers shall comply with Annex IV",
        "is classified as class IIb",
        "Member States shall lay down penalties",
    ]
    for text in samples:
        result = _classify(text)
        assert result.role in PROVISION_ROLE_TAXONOMY


def test_classifier_always_returns_confidence_in_range():
    samples = [
        ("'X' means Y.", "paragraph"),
        ("shall not apply to research", "paragraph"),
        ("anything", "recital"),
        ("anything", "chapter"),
        ("", "paragraph"),
    ]
    for text, kind in samples:
        result = _classify(text, kind=kind)
        assert 0.0 <= result.confidence <= 1.0
