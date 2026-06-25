"""Unit tests for the typed read-path contracts (Phase 1, zero behaviour change).

The load-bearing tests are the *equivalence* ones: they assert the new typed
view produces exactly what the existing read path already computes, so adopting
the contract cannot drift behaviour.
"""
from __future__ import annotations

from application.contracts import Definition, Evidence, Provision, Scenario
from application._faithfulness import _provision_text, _definition_text, _build_corpus
from domain.legislation_catalog import AI_ACT_CELEX as _AI_ACT, MDR_CELEX as _MDR


_PROVISION = {
    "article_id": f"{_AI_ACT}_article_43",
    "celex": _AI_ACT,
    "article_ref": "Article 43",
    "article_text": "Where a high-risk AI system undergoes a substantial modification …",
    "binding_force": "binding",
    "provision_role": "obligation",
    "matched_leaf_id": f"{_AI_ACT}_article_43_para_4",
    "children": [
        {"id": f"{_AI_ACT}_article_43_para_4", "ref": "Article 43(4)",
         "kind": "paragraph", "raw_text": "A new conformity assessment is required."},
        {"id": "x", "ref": "Article 43(1)", "kind": "paragraph", "text": "fallback text"},
    ],
    "interpreting_guidance": [{"text": "MDCG note on conformity."}],
    "interpreted_provisions": [],
    "score": 0.91,
    "_pointer_expansion": True,
}

_DEFINITION = {
    "term": "ai system",
    "term_normalized": "ai_system",
    "celex": _AI_ACT,
    "article_ref": "Article 3(1)",
    "regulation": "EU AI Act",
    "definition_type": "formal",
    "definition_text": "'AI system' means a machine-based system …",
    "source_provision_id": f"{_AI_ACT}_article_3",
}


# ---------------------------------------------------------------------------
# Provision — typed view + lossless round-trip + equivalence
# ---------------------------------------------------------------------------


def test_provision_typed_accessors_match_dict():
    p = Provision.from_dict(_PROVISION)
    assert p.article_id == "32024R1689_article_43"
    assert p.celex == "32024R1689"
    assert p.article_ref == "Article 43"
    assert p.binding_force == "binding"
    assert p.provision_role == "obligation"
    assert p.matched_leaf_id == "32024R1689_article_43_para_4"
    assert len(p.children) == 2


def test_provision_to_dict_is_lossless():
    p = Provision.from_dict(_PROVISION)
    # The dict is the source of truth; round-trip preserves every key, including
    # internal provenance flags the typed view does not model.
    assert p.to_dict() is _PROVISION
    assert p.to_dict()["_pointer_expansion"] is True
    assert p.to_dict()["score"] == 0.91


def test_provision_text_payload_is_the_full_quotable_corpus_text():
    # The contract is now the single implementation (the _faithfulness helper
    # delegates to it). Pin the exact bytes: body → children (raw_text, else
    # text) → interpretive-link lines, joined by newlines.
    p = Provision.from_dict(_PROVISION)
    expected = (
        "Where a high-risk AI system undergoes a substantial modification …\n"
        "A new conformity assessment is required.\n"
        "fallback text\n"
        "MDCG note on conformity."
    )
    assert p.text_payload() == expected


def test_faithfulness_helpers_delegate_to_the_contract():
    # _faithfulness is the contract's first real consumer: its corpus-building
    # helpers route through text_payload(), so the two cannot drift.
    assert _provision_text(_PROVISION) == Provision.from_dict(_PROVISION).text_payload()
    assert _definition_text(_DEFINITION) == Definition.from_dict(_DEFINITION).text_payload()
    # And the assembled corpus contains the contract's payload for each item.
    corpus = _build_corpus([_PROVISION], [_DEFINITION])
    assert "a new conformity assessment is required." in corpus  # normalized child
    assert "mdcg note on conformity." in corpus                  # interpretive link


def test_provision_identity_prefers_node_id_not_display_ref():
    assert Provision.from_dict({"article_id": f"{_MDR}_annex_1"}).identity == f"{_MDR}_annex_1"
    # No id -> celex|article_ref (never a bare, non-unique display_ref)
    assert Provision.from_dict(
        {"celex": _MDR, "article_ref": "Annex I"}
    ).identity == f"{_MDR}|Annex I"


# ---------------------------------------------------------------------------
# Definition
# ---------------------------------------------------------------------------


def test_definition_typed_accessors_and_text_payload():
    d = Definition.from_dict(_DEFINITION)
    assert d.term == "ai system"
    assert d.celex == "32024R1689"
    assert d.definition_type == "formal"
    assert d.text_payload() == "'AI system' means a machine-based system …"
    assert d.to_dict() is _DEFINITION


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


def test_scenario_construction_and_helpers():
    s = Scenario(
        question="What obligations apply?",
        mentioned_regs=frozenset({"EU AI Act", "MDR 2017/745"}),
        target_celexes=frozenset({"32024R1689", "32017R0745"}),
        role_specs=(("provider", "32024R1689"),),
        explicit_refs=(),
        route_id="classification_chain",
        is_definition_question=False,
    )
    assert s.has_role is True
    assert s.is_cross_regulation is True
    assert s.in_scope("32024R1689") is True
    assert s.in_scope("32016R0679") is False


def test_scenario_defaults_are_empty_not_none():
    s = Scenario(question="x")
    assert s.role_specs == ()
    assert s.has_role is False
    assert s.is_cross_regulation is False


# ---------------------------------------------------------------------------
# Evidence — single dedup point
# ---------------------------------------------------------------------------


def test_evidence_dedup_by_identity():
    dup = dict(_PROVISION)  # same article_id -> same identity
    ev = Evidence.from_dicts([_PROVISION, dup], [_DEFINITION])
    assert len(ev.provisions) == 2
    assert len(ev.unique_provisions()) == 1


def test_evidence_extend_merges_without_duplicates():
    ev = Evidence.from_dicts([_PROVISION], [_DEFINITION])
    other = Evidence.from_dicts(
        [_PROVISION, {"article_id": "32017R0745_article_10", "celex": "32017R0745"}],
        [_DEFINITION],
    )
    ev.extend(other)
    assert sorted(ev.provision_ids()) == ["32017R0745_article_10", "32024R1689_article_43"]
    assert len(ev.definitions) == 1  # same (term, celex) not duplicated


def test_evidence_provision_dicts_round_trip_to_legacy_hot_path():
    ev = Evidence.from_dicts([_PROVISION], [])
    assert ev.provision_dicts() == [_PROVISION]
