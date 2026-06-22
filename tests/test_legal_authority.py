"""Tests for the interpretive-authority taxonomy (#7b)."""
from domain.ontology.legal_authority import (
    AUTHORITY_LEVELS,
    BINDING,
    PERSUASIVE,
    PRESUMPTION_OF_CONFORMITY,
    authority_for_source,
)


def test_guidance_is_persuasive():
    assert authority_for_source("guidance") == PERSUASIVE


def test_harmonised_standard_confers_presumption_of_conformity():
    assert authority_for_source("harmonised_standard") == PRESUMPTION_OF_CONFORMITY


def test_implementing_and_delegated_acts_are_binding():
    assert authority_for_source("implementing_act") == BINDING
    assert authority_for_source("delegated_act") == BINDING
    assert authority_for_source("common_specification") == BINDING


def test_unknown_or_missing_defaults_to_persuasive():
    # Conservative default — never over-state an unknown source's weight.
    assert authority_for_source(None) == PERSUASIVE
    assert authority_for_source("") == PERSUASIVE
    assert authority_for_source("something_new") == PERSUASIVE


def test_case_insensitive():
    assert authority_for_source("Harmonised_Standard") == PRESUMPTION_OF_CONFORMITY


def test_levels_are_distinct_and_registered():
    assert len(set(AUTHORITY_LEVELS)) == 3
    assert PERSUASIVE in AUTHORITY_LEVELS
