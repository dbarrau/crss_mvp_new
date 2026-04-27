"""Unit tests for INTERPRETS-expansion and guidance anchor changes in GraphRetriever.

These tests are purely structural — they verify that the Cypher queries and
kind sets contain the right constructs without touching Neo4j.
"""
from retrieval.graph_retriever import (
    _EXPAND_CYPHER,
    _PARENT_KINDS,
)
from application.agent import _format_context


# ── _PARENT_KINDS ─────────────────────────────────────────────────────────

def test_parent_kinds_includes_guidance_paragraph():
    assert "guidance_paragraph" in _PARENT_KINDS


def test_parent_kinds_includes_guidance_chart():
    assert "guidance_chart" in _PARENT_KINDS


def test_parent_kinds_retains_existing_guidance_kinds():
    assert "guidance_section" in _PARENT_KINDS
    assert "guidance_subsection" in _PARENT_KINDS


# ── _EXPAND_CYPHER ────────────────────────────────────────────────────────

def test_expand_cypher_has_interprets_inbound():
    """Legislation provisions must pull in guidance that INTERPRETS them."""
    assert "(interp_g:Guidance)-[:INTERPRETS]->(art)" in _EXPAND_CYPHER
    assert "interpreting_guidance" in _EXPAND_CYPHER


def test_expand_cypher_has_interprets_outbound():
    """Guidance nodes must pull the legislation provisions they INTERPRETS."""
    assert "(art)-[:INTERPRETS]->(interp_p:Provision)" in _EXPAND_CYPHER
    assert "interpreted_provisions" in _EXPAND_CYPHER


def test_expand_cypher_null_filters_on_interprets_collections():
    """Null-guard list comprehensions must be present to avoid phantom rows."""
    # Both new collections must use the WHERE x.id IS NOT NULL guard.
    assert _EXPAND_CYPHER.count("WHERE x.id IS NOT NULL") >= 2


def test_expand_cypher_returns_both_interprets_columns():
    """RETURN clause must expose interpreting_guidance and interpreted_provisions."""
    assert "interpreting_guidance," in _EXPAND_CYPHER
    assert "interpreted_provisions" in _EXPAND_CYPHER


# ── _format_context ───────────────────────────────────────────────────────

def _make_provision(**overrides) -> dict:
    base = {
        "article_id": "test_id",
        "celex": "32017R0745",
        "regulation": "MDR",
        "article_ref": "Article 1",
        "article_path": "",
        "article_text": "Base article text.",
        "children": [],
        "cited_provisions": [],
        "cross_reg_cited": [],
        "interpreting_guidance": [],
        "interpreted_provisions": [],
        "score": 1.0,
        "matched_leaf_id": None,
    }
    base.update(overrides)
    return base


def test_format_context_renders_interpreting_guidance():
    prov = _make_provision(
        interpreting_guidance=[
            {
                "id": "MDCG_2020_3_sec_4",
                "ref": "Section 4",
                "text": "This guidance section interprets Article 120(3c).",
            }
        ]
    )
    output = _format_context([prov])
    assert "[GUIDANCE interprets this] Section 4:" in output
    assert "Article 120(3c)" in output


def test_format_context_renders_interpreted_provisions():
    prov = _make_provision(
        celex="MDCG_2020_3",
        regulation="MDCG 2020-3 Rev.1",
        article_ref="Section 4.3.2",
        interpreting_guidance=[],
        interpreted_provisions=[
            {
                "id": "32017R0745_art120",
                "ref": "Article 120",
                "text": "Article 120 provides the transitional provisions.",
            }
        ],
    )
    output = _format_context([prov])
    assert "[INTERPRETS legislation] Article 120:" in output
    assert "transitional provisions" in output


def test_format_context_omits_interprets_block_when_empty():
    prov = _make_provision()
    output = _format_context([prov])
    assert "Interpretive links" not in output


def test_format_context_guidance_tag_applied():
    """MDCG provisions must be tagged [GUIDANCE] in the formatted output."""
    prov = _make_provision(celex="MDCG_2020_3", regulation="MDCG 2020-3")
    output = _format_context([prov])
    assert "[GUIDANCE]" in output
