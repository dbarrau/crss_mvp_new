"""Tests for Phase 3 GraphRAG community integration.

Covers:
- _routing.py: community_summary_search route selection
- _context.py: _community_summary_header builder
- retrieval/graph_retriever.py: retrieve_by_communities_hierarchical
- _retrieval.py: community route wiring and sufficiency checks
"""
from __future__ import annotations

import numpy as np
import pytest

from application.agent import (
    _select_question_route,
    _evaluate_route_sufficiency,
    _retrieve_route_provisions,
)
from application._context import _community_summary_header
from application._routing import _is_community_summary_question


# ---------------------------------------------------------------------------
# Helper: minimal provision dicts
# ---------------------------------------------------------------------------

def _make_provision(*, article_id="art-1", celex="32024R1689", community_id=None,
                    community_summary=None, community_retrieval=False):
    return {
        "article_id": article_id,
        "article_ref": "Article 1",
        "article_text": "Some text.",
        "article_path": "",
        "celex": celex,
        "regulation": "EU AI Act",
        "score": 0.9,
        "children": [],
        "cited_provisions": [],
        "cross_reg_cited": [],
        "matched_leaf_id": None,
        "community_id": community_id,
        "community_summary": community_summary,
        "_community_retrieval": community_retrieval,
    }


# ---------------------------------------------------------------------------
# 1. Routing: _is_community_summary_question
# ---------------------------------------------------------------------------

class TestIsCommunityQuestion:
    def test_all_obligations_triggers(self):
        assert _is_community_summary_question(
            "What are all obligations of providers under the AI Act?",
            mentioned_regs={"EU AI Act"},
            role_specs=[],
        )

    def test_comprehensive_with_reg_triggers(self):
        assert _is_community_summary_question(
            "Give me a comprehensive overview of the AI Act.",
            mentioned_regs={"EU AI Act"},
            role_specs=[],
        )

    def test_enumerate_with_role_triggers(self):
        assert _is_community_summary_question(
            "Enumerate all duties of a deployer.",
            mentioned_regs=set(),
            role_specs=[("deployer", "32024R1689")],
        )

    def test_no_signal_returns_false(self):
        assert not _is_community_summary_question(
            "What does Article 6 require?",
            mentioned_regs={"EU AI Act"},
            role_specs=[],
        )

    def test_broad_language_without_reg_or_role_returns_false(self):
        # "all" present but no concrete scope
        assert not _is_community_summary_question(
            "Tell me all the things.",
            mentioned_regs=set(),
            role_specs=[],
        )


# ---------------------------------------------------------------------------
# 2. Routing: _select_question_route returns community_summary_search
# ---------------------------------------------------------------------------

class TestSelectRouteCommunitySummary:
    def test_route_selected_for_comprehensive_question(self):
        route = _select_question_route(
            "List all obligations under the EU AI Act.",
            explicit_refs=[],
            mentioned_regs={"EU AI Act"},
            role_specs=[],
            is_definition_question=False,
        )
        assert route.id == "community_summary_search"

    def test_explicit_ref_takes_precedence(self):
        """provision_lookup must win even when broad language is present."""
        route = _select_question_route(
            "Give a comprehensive overview of all requirements in Article 10.",
            explicit_refs=["Article 10"],
            mentioned_regs={"EU AI Act"},
            role_specs=[],
            is_definition_question=False,
        )
        assert route.id == "provision_lookup"

    def test_multi_reg_cross_regulation_takes_precedence(self):
        """cross_regulation wins when 2+ regs are in scope before community check."""
        route = _select_question_route(
            "Compare all obligations under MDR and AI Act.",
            explicit_refs=[],
            mentioned_regs={"EU AI Act", "MDR 2017/745"},
            role_specs=[],
            is_definition_question=False,
        )
        assert route.id == "cross_regulation"


# ---------------------------------------------------------------------------
# 3. Context: _community_summary_header
# ---------------------------------------------------------------------------

class TestCommunitySummaryHeader:
    def test_returns_empty_when_no_community_data(self):
        provisions = [_make_provision()]
        assert _community_summary_header(provisions) == ""

    def test_returns_header_with_summaries(self):
        provisions = [
            _make_provision(
                community_id="community::0001",
                community_summary="This community covers conformity assessment.",
            ),
            _make_provision(
                article_id="art-2",
                community_id="community::0002",
                community_summary="This community covers post-market surveillance.",
            ),
        ]
        header = _community_summary_header(provisions)
        assert "[Community Overview]" in header
        assert "conformity assessment" in header
        assert "post-market surveillance" in header

    def test_deduplicates_communities(self):
        summary = "Conformity assessment obligations."
        provisions = [
            _make_provision(
                article_id="art-1",
                community_id="community::0001",
                community_summary=summary,
            ),
            _make_provision(
                article_id="art-2",
                community_id="community::0001",
                community_summary=summary,
            ),
        ]
        header = _community_summary_header(provisions)
        # summary should appear exactly once
        assert header.count("Conformity assessment") == 1

    def test_returns_empty_when_summary_is_missing(self):
        provisions = [_make_provision(community_id="community::0001", community_summary="")]
        assert _community_summary_header(provisions) == ""


# ---------------------------------------------------------------------------
# 4. Fake retriever with community support
# ---------------------------------------------------------------------------

class _FakeCommunityRetriever:
    """Minimal fake retriever for community route tests."""

    def __init__(self, *, community_provisions=None, hybrid=None, fail_community=False):
        self._community_provisions = community_provisions or []
        self._hybrid = hybrid or []
        self._fail_community = fail_community
        self.calls: list[str] = []

    def retrieve_by_communities_hierarchical(
        self, question, *, k_communities=5, k_provisions=20,
        target_celexes=None, query_vec=None,
    ):
        self.calls.append("community")
        if self._fail_community:
            return []
        return list(self._community_provisions)

    def encode_as_passage(self, text):
        self.calls.append("encode")
        return np.zeros(768, dtype=np.float32)

    def encode_as_query(self, text):
        self.calls.append("encode_query")
        return np.zeros(768, dtype=np.float32)

    def retrieve(self, question, k=20, target_celexes=None, query_vec=None):
        self.calls.append("retrieve")
        return list(self._hybrid)

    def retrieve_by_refs(self, refs, celex_filter=None):
        self.calls.append("direct")
        return []

    def retrieve_by_roles(self, role_specs, k=8):
        self.calls.append("roles")
        return []

    def get_all_community_summaries(self, *, level=1):
        # Return empty list in tests — map-reduce is optional
        return []


# ---------------------------------------------------------------------------
# 5. Retrieval wiring: _retrieve_route_provisions for community route
# ---------------------------------------------------------------------------

_DUMMY_ROUTE_COMMUNITY = type("R", (), {"id": "community_summary_search"})()


class _FakeDecomposeClient:
    """Stub Mistral client that makes _decompose_question fall back to the
    original question (returns only 1 numbered item, below the 2-item
    threshold required to activate multi-sub-question decomposition)."""
    class _Chat:
        def complete(self, *, model, messages, temperature, max_tokens):
            return type("R", (), {
                "choices": [type("C", (), {
                    "message": type("M", (), {"content": "1. sub-question placeholder"})(),
                })()],
            })()
    chat = _Chat()


_FAKE_DECOMPOSE_CLIENT = _FakeDecomposeClient()


def _noop_hyde(question, client):
    return "hyde text"


class TestRetrieveRouteProvisionsCommunity:
    def test_calls_community_method(self):
        community_prov = _make_provision(
            community_id="community::0001",
            community_summary="summary",
            community_retrieval=True,
        )
        retriever = _FakeCommunityRetriever(community_provisions=[community_prov])

        result = _retrieve_route_provisions(
            "List all obligations under the AI Act.",
            retriever,
            client=_FAKE_DECOMPOSE_CLIENT,
            k=20,
            route=_DUMMY_ROUTE_COMMUNITY,
            target_celexes={"32024R1689"},
            explicit_refs=[],
            role_specs=[],
            has_definitions=False,
            hyde_builder=_noop_hyde,
        )

        # Raw question encoding is used for community matching (no HyDE bias)
        assert "encode_query" in retriever.calls
        assert "community" in retriever.calls
        # No HyDE provision merge — community retrieval is standalone
        assert "retrieve" not in retriever.calls
        assert community_prov in result["provisions"]

    def test_community_provisions_are_sole_result(self):
        """Community provisions are returned as-is; no HyDE merge."""
        community_prov = _make_provision(article_id="art-community")
        retriever = _FakeCommunityRetriever(
            community_provisions=[community_prov],
        )

        result = _retrieve_route_provisions(
            "List all obligations under the AI Act.",
            retriever,
            client=_FAKE_DECOMPOSE_CLIENT,
            k=20,
            route=_DUMMY_ROUTE_COMMUNITY,
            target_celexes=None,
            explicit_refs=[],
            role_specs=[],
            has_definitions=False,
            hyde_builder=_noop_hyde,
        )

        article_ids = [p["article_id"] for p in result["provisions"]]
        assert "art-community" in article_ids
        assert "retrieve" not in retriever.calls

    def test_returns_empty_when_community_finds_nothing(self):
        """When community retrieval returns empty, provisions is empty — no HyDE fallback."""
        retriever = _FakeCommunityRetriever(fail_community=True)

        result = _retrieve_route_provisions(
            "List all obligations under the AI Act.",
            retriever,
            client=_FAKE_DECOMPOSE_CLIENT,
            k=20,
            route=_DUMMY_ROUTE_COMMUNITY,
            target_celexes=None,
            explicit_refs=[],
            role_specs=[],
            has_definitions=False,
            hyde_builder=_noop_hyde,
        )

        assert "community" in retriever.calls
        assert "retrieve" not in retriever.calls
        assert result["provisions"] == []

    def test_map_results_key_present_in_return(self):
        """map_results key must always exist in the return dict."""
        retriever = _FakeCommunityRetriever()
        result = _retrieve_route_provisions(
            "List all obligations under the AI Act.",
            retriever,
            client=_FAKE_DECOMPOSE_CLIENT,
            k=20,
            route=_DUMMY_ROUTE_COMMUNITY,
            target_celexes=None,
            explicit_refs=[],
            role_specs=[],
            has_definitions=False,
            hyde_builder=_noop_hyde,
        )
        assert "map_results" in result
        assert isinstance(result["map_results"], list)


# ---------------------------------------------------------------------------
# 6. Sufficiency: community_summary_search route
# ---------------------------------------------------------------------------

_DUMMY_ROUTE_OBJ = type("R", (), {"id": "community_summary_search"})()


class TestSufficiencyCommunitySummaryRoute:
    def test_passes_with_provisions_from_multiple_communities(self):
        provisions = [
            _make_provision(
                article_id="art-1",
                community_id="community::0001",
                community_retrieval=True,
            ),
            _make_provision(
                article_id="art-2",
                community_id="community::0002",
                community_retrieval=True,
            ),
        ]
        result = _evaluate_route_sufficiency(
            route=_DUMMY_ROUTE_OBJ,
            question="All obligations under AI Act?",
            explicit_refs=[],
            target_celexes=None,
            role_specs=[],
            provisions=provisions,
            definitions=[],
            direct_provisions=[],
            role_provisions=[],
            legal_qualification_targets=[],
        )
        assert result["ok"] is True

    def test_fails_when_no_provisions_returned(self):
        result = _evaluate_route_sufficiency(
            route=_DUMMY_ROUTE_OBJ,
            question="All obligations under AI Act?",
            explicit_refs=[],
            target_celexes=None,
            role_specs=[],
            provisions=[],
            definitions=[],
            direct_provisions=[],
            role_provisions=[],
            legal_qualification_targets=[],
        )
        assert result["ok"] is False
        names = {c["name"] for c in result["checks"]}
        assert "community_coverage" in names

    def test_flags_hyde_fallback(self):
        """When community index is absent and HyDE was used, report it."""
        provisions = [
            _make_provision(
                article_id="art-hyde-1",
                community_id=None,
                community_retrieval=False,
            ),
        ]
        result = _evaluate_route_sufficiency(
            route=_DUMMY_ROUTE_OBJ,
            question="All obligations?",
            explicit_refs=[],
            target_celexes=None,
            role_specs=[],
            provisions=provisions,
            definitions=[],
            direct_provisions=[],
            role_provisions=[],
            legal_qualification_targets=[],
        )
        names = {c["name"] for c in result["checks"]}
        assert "community_index_present" in names
        idx_check = next(c for c in result["checks"] if c["name"] == "community_index_present")
        assert idx_check["passed"] is False

    def test_context_communities_key_present(self):
        provisions = [
            _make_provision(
                article_id="art-1",
                community_id="community::0001",
                community_retrieval=True,
            ),
        ]
        result = _evaluate_route_sufficiency(
            route=_DUMMY_ROUTE_OBJ,
            question="All obligations?",
            explicit_refs=[],
            target_celexes=None,
            role_specs=[],
            provisions=provisions,
            definitions=[],
            direct_provisions=[],
            role_provisions=[],
            legal_qualification_targets=[],
        )
        assert "context_communities" in result
        assert "community::0001" in result["context_communities"]
