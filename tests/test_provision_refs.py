"""Provision-reference extraction + normalisation (application/_config).

Guards the direct-ref lookup contract: a user writes a bare-parenthesis ref
("Article 2(65)") while the graph stores the qualified display_ref
("Article 2, point (65)").  ``_normalize_ref_key`` folds both to one key so the
lookup (and the sufficiency coverage check) can match them.  This mirrors the
``_REF_NORM`` fold applied inside ``retrieval._cypher._DIRECT_REF_CYPHER``.

Deterministic, no Neo4j / no LLM.
"""
import pytest

from application._config import _extract_provision_refs, _normalize_ref_key


def test_extract_keeps_point_number():
    # Regression: the definition-point number must survive extraction, not be
    # dropped down to "Article 2" (which would silently widen to the article).
    assert _extract_provision_refs("Show me article 2(65) of the MDR") == ["Article 2(65)"]
    assert _extract_provision_refs("What does Article 53(1) say?") == ["Article 53(1)"]
    assert _extract_provision_refs("Annex I of the AI Act") == ["Annex I"]


@pytest.mark.parametrize(
    "user_ref, stored_display_ref",
    [
        ("Article 2(65)", "Article 2, point (65)"),        # MDR definition point
        ("Article 2(58)(b)", "Article 2, point (58)(b)"),  # lettered sub-point
        ("Article 3(65)", "Article 3, point (65)"),        # AI Act definition point
        ("Article 53(1)(a)", "Article 53(1), point (a)"),  # point under a paragraph
        ("Article 4(7)", "Article 4, point (7)"),          # GDPR definition
        ("Article 10(1)", "Article 10(1)"),                # paragraph — already aligned
        ("Article 10", "Article 10"),                       # bare article
        ("Annex I", "Annex I"),
    ],
)
def test_user_ref_and_stored_ref_share_key(user_ref, stored_display_ref):
    """The bare user form and the qualified stored form fold to the same key."""
    assert _normalize_ref_key(user_ref) == _normalize_ref_key(stored_display_ref)


def test_distinct_refs_do_not_collide():
    # Different provisions must NOT collapse together.
    assert _normalize_ref_key("Article 2(65)") != _normalize_ref_key("Article 2(66)")
    assert _normalize_ref_key("Article 2") != _normalize_ref_key("Article 2(1)")
    assert _normalize_ref_key("Article 2(58)") != _normalize_ref_key("Article 2(58)(b)")