"""Drift tests reconciling the actor lexicons against one source of truth (#5).

``actor_roles.CANONICAL_ACTOR_ROLES`` is authoritative. These tests fail loudly
if either dependent lexicon drifts from it — preventing the class of silent
inconsistency that left GDPR ``supervisory authority`` routable (Tier 2) but
undetectable by the obligation rule.
"""
from domain.ontology.actor_roles import (
    CANONICAL_ACTOR_ROLES,
    ENTITY_SYNONYMS,
    is_known_actor_role,
    normalize_role_term,
)
from domain.ontology.provision_roles import _ACTOR_SUBJECT_RE


def test_entity_synonym_targets_are_all_registered_roles():
    """Every role ENTITY_SYNONYMS routes to must exist in the registry."""
    offenders = []
    for phrase, specs in ENTITY_SYNONYMS.items():
        for role, celex in specs:
            if not is_known_actor_role(role, celex):
                offenders.append((phrase, role, celex))
    assert not offenders, (
        "ENTITY_SYNONYMS routes to roles absent from CANONICAL_ACTOR_ROLES: "
        f"{offenders}"
    )


def test_every_canonical_role_is_an_obligation_subject():
    """Every registered role must be recognized by the obligation-rule regex,
    so a provision imposing a duty on it can classify as OBLIGATION."""
    unmatched = [
        role for role in CANONICAL_ACTOR_ROLES
        if not _ACTOR_SUBJECT_RE.search(role)
    ]
    assert not unmatched, (
        "Canonical roles not matched by provision_roles._ACTOR_SUBJECT_RE "
        f"(add them to _ACTOR_SUBJECTS): {unmatched}"
    )


def test_canonical_role_keys_are_normalized():
    """Registry keys must be in the normalized (lowercase) comparison form."""
    for role in CANONICAL_ACTOR_ROLES:
        assert role == role.lower().strip()


def test_gdpr_roles_are_registered_and_detectable():
    # Regression guard for the specific Tier-2 / Tier-3 gap.
    for role in ("controller", "processor", "supervisory authority"):
        assert is_known_actor_role(role, "32016R0679")
        assert _ACTOR_SUBJECT_RE.search(role)
    # normalize_role_term keeps multi-word roles consistent with ActorRole keys
    assert normalize_role_term("supervisory authority") == "supervisory_authority"
