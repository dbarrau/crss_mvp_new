"""Graph-aware vector retrieval over Neo4j provision embeddings.

Performs hybrid retrieval:
1. In-memory cosine similarity → top-k parent provisions (numpy, fast for <10k nodes)
2. Graph traversal → expand children via HAS_PART
3. Cross-reference expansion → follow CITES edges

No Neo4j Vector Index plugin required — embeddings are loaded into
memory at startup and similarity is computed with numpy.

This module is the public facade.  The subsystems live in private
submodules, each owning its own state:

- :mod:`retrieval._config`      — tuning constants (RRF, reranker, budgets)
- :mod:`retrieval._cypher`      — Cypher queries + kind sets (pure data)
- :mod:`retrieval._dense`       — embedding matrix, encoder, anchor mapping
- :mod:`retrieval._lexical`     — BM25 full-text channel
- :mod:`retrieval._reranking`   — cross-encoder blended reranking
- :mod:`retrieval._communities` — community-summary index
- :mod:`retrieval._traversal`   — ref/id/role/chain lookups + drilldown

Import :class:`GraphRetriever` from here; the private submodules are an
implementation detail.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase

from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri
from retrieval import _traversal
from retrieval._communities import CommunityIndex
from retrieval._config import (  # noqa: F401  (prefixes re-exported for back-compat)
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    _CANDIDATE_CAP,
    _CANDIDATE_MULTIPLIER,
    _LEXICAL_WEIGHT,
    _RRF_K,
)
from retrieval._cypher import (  # noqa: F401  (re-exported for structural tests)
    _EXPAND_CYPHER,
    _PARENT_KINDS,
    _REVERSE_XREF_CYPHER,
)
from retrieval._dense import DenseIndex
from retrieval._lexical import LexicalChannel
from retrieval._reranking import Reranker

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
logger = logging.getLogger(__name__)


class GraphRetriever:
    """Hybrid vector + graph retriever backed by Neo4j.

    On init, loads all embedded provisions into a numpy matrix.
    Queries compute cosine similarity in-memory (~1ms for 6k vectors),
    then expand top-k results via Cypher graph traversal.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        uri = _normalize_neo4j_uri(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        )
        self._driver = GraphDatabase.driver(
            uri,
            auth=(
                os.environ.get(
                    "NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j")
                ),
                os.environ.get("NEO4J_PASSWORD", "password"),
            ),
        )
        self._db = os.environ.get("NEO4J_DATABASE", "neo4j")

        # Dense channel: the embedding index, loaded into memory now.
        self._dense = DenseIndex(self._driver, self._db, model_name)

        # Lexical (BM25) channel via a Neo4j full-text index.  Disabled by
        # setting CRSS_LEXICAL=0.  Created idempotently on startup.
        self._lexical = LexicalChannel(self._driver, self._db)

        # Cross-encoder reranker (optional).  Disabled by setting
        # CRSS_RERANKER=0.  Model overrideable via CRSS_RERANKER_MODEL.
        self._reranker = Reranker.load_if_enabled()

        # Community-summary index (lazy — loaded on first community query).
        self._communities = CommunityIndex(self._driver, self._db)

        # Whole-graph lookup caches (fetched once, on first use).
        self._reference_index_cache: dict[str, tuple[str, str]] | None = None
        self._term_index_cache: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_as_passage(self, text: str) -> np.ndarray:
        """Encode *text* with the passage prefix, matching the provision embedding space.

        Use this for HyDE (Hypothetical Document Embedding): the generated
        hypothetical answer should be encoded as a passage so it sits in the
        same embedding space as the stored provision embeddings.
        """
        return self._dense.encode_passage(text)

    def encode_as_query(self, text: str) -> np.ndarray:
        """Encode *text* with the query prefix.

        Use this when you want to embed the raw question for community-level
        retrieval, bypassing HyDE.  Community summaries are semantically rich
        enough to match a plain question vector; using a query prefix (rather
        than a passage prefix) aligns with how the model was trained.
        """
        return self._dense.encode_query(text)

    # ------------------------------------------------------------------
    # Hybrid retrieval (dense + lexical fusion → expansion → rerank)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        k: int = 5,
        target_celexes: set[str] | None = None,
        query_vec: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k provisions with children and cross-references.

        Parameters
        ----------
        target_celexes:
            If provided, filters results to the given CELEX(es).  When more
            than one CELEX is given, allocates slots per regulation so each
            regulation gets adequate coverage.
        query_vec:
            Optional pre-computed query vector (e.g. from HyDE).  When
            provided the *question* string is used only for logging; encoding
            is skipped.
        """
        dense = self._dense
        if dense.matrix is None or len(dense.ids) == 0:
            return []

        # Widen the cosine candidate pool when a reranker is active so the
        # cross-encoder gets enough candidates to discriminate.
        candidate_k = (
            min(k * _CANDIDATE_MULTIPLIER, _CANDIDATE_CAP)
            if self._reranker is not None
            else k
        )

        # Encode query (or use a pre-computed vector, e.g. from HyDE)
        if query_vec is not None:
            q_vec = query_vec
        else:
            q_vec = dense.encode_query(question)

        # Cosine similarity (embeddings are already L2-normalized)
        scores = dense.matrix @ q_vec

        # ------------------------------------------------------------------
        # Reciprocal Rank Fusion of the dense (cosine) and lexical (BM25)
        # channels.  Dense cosine alone smears legally-distinct but
        # vocabulary-similar provisions into a near-tie (e.g. Annex II
        # "Technical Documentation" vs Annex IX "assessment of technical
        # documentation").  A BM25 ranking lets an exact heading/term match
        # break the tie.  RRF fuses *ranks*, not scores, so the two channels'
        # incompatible score scales need no calibration.
        # ------------------------------------------------------------------
        dense_order = sorted(
            range(len(dense.ids)), key=lambda i: scores[i], reverse=True,
        )
        dense_rank = {idx: rank for rank, idx in enumerate(dense_order)}

        # Pull a generous lexical pool so BM25 hits outside the dense top
        # still contribute their rank to the fusion.  Filter to the in-scope
        # regulation(s) so cross-reg guidance can't starve the budget.
        lexical_rank = self._lexical.search(
            question, candidate_k * 8, celex_filter=target_celexes,
        )

        def _fused_score(idx: int) -> float:
            f = 1.0 / (_RRF_K + dense_rank[idx])
            nid = dense.ids[idx]
            if nid in lexical_rank:
                f += _LEXICAL_WEIGHT / (_RRF_K + lexical_rank[nid])
            return f

        fused_order = sorted(
            range(len(dense.ids)), key=_fused_score, reverse=True,
        )

        # Walk the fused ranking, mapping each embedded node to its nearest
        # parent-kind ancestor. This lets deep leaf nodes (point, roman_item)
        # surface their parent article/section with the most specific match.
        score_map: dict[str, float] = {}
        leaf_map: dict[str, str] = {}

        if target_celexes and len(target_celexes) > 1:
            # Multi-regulation mode: allocate slots per regulation to
            # guarantee coverage of each mentioned regulation.
            # Use ceiling division so k=6 across 2 regs → 3 each (not 2).
            per_reg = max(3, -(-candidate_k // len(target_celexes)))  # ceiling division
            per_reg_count: dict[str, int] = {c: 0 for c in target_celexes}

            for idx in fused_order:
                anchor_id = dense.find_anchor(
                    dense.ids[idx], dense.kinds[idx], dense.path_strings[idx],
                )
                if anchor_id in score_map:
                    continue
                # Determine celex of the anchor
                anchor_celex = dense.celexes[idx]
                for c in target_celexes:
                    if anchor_id.startswith(c):
                        anchor_celex = c
                        break

                # Only include results from target regulations
                if anchor_celex not in per_reg_count:
                    continue
                if per_reg_count[anchor_celex] >= per_reg:
                    continue
                per_reg_count[anchor_celex] += 1

                score_map[anchor_id] = _fused_score(idx)
                leaf_map[anchor_id] = dense.ids[idx]

                if sum(per_reg_count.values()) >= candidate_k:
                    break
        else:
            # Standard mode (single regulation or no filter)
            for idx in fused_order:
                # If a celex filter is set (single-regulation question),
                # skip embeddings from other regulations so they don't consume
                # top-k slots.
                if target_celexes and dense.celexes[idx] not in target_celexes:
                    continue
                anchor_id = dense.find_anchor(
                    dense.ids[idx], dense.kinds[idx], dense.path_strings[idx],
                )
                if anchor_id not in score_map:
                    score_map[anchor_id] = _fused_score(idx)
                    leaf_map[anchor_id] = dense.ids[idx]
                if len(score_map) >= candidate_k:
                    break

        top_ids = list(score_map.keys())

        # Graph expansion via Cypher
        results = _traversal.expand(self._driver, self._db, top_ids)

        # Attach cosine scores and matched leaf IDs
        for r in results:
            r["score"] = score_map.get(r["article_id"], 0.0)
            r["matched_leaf_id"] = leaf_map.get(r["article_id"])

        # Rerank the widened candidate pool to the final top-k, or fall back
        # to cosine ordering when no reranker is loaded.
        if self._reranker is not None and len(results) > k:
            if target_celexes and len(target_celexes) > 1:
                # Multi-regulation: rerank within each regulation's slot so the
                # cross-encoder cannot make one regulation crowd out another.
                per_reg = max(1, -(-k // len(target_celexes)))  # ceiling div
                reranked: list[dict] = []
                for celex in target_celexes:
                    bucket = [r for r in results if r.get("celex") == celex]
                    if len(bucket) > per_reg:
                        bucket = self._reranker.rerank(question, bucket, per_reg)
                    else:
                        bucket.sort(
                            key=lambda r: r.get("rerank_score", r.get("score", 0.0)),
                            reverse=True,
                        )
                    reranked.extend(bucket)
                results = reranked
            else:
                results = self._reranker.rerank(question, results, k)
        else:
            results.sort(key=lambda r: r["score"], reverse=True)

        # Use only the reranked top-k IDs for cross-reg expansion, so the
        # widened pool doesn't pollute the reverse-xref budget.
        reranked_ids = [r["article_id"] for r in results]

        # Drill into cited container nodes (e.g. "Annex XIV" headings)
        # to surface their children's text for the LLM.
        _traversal.expand_cited_containers(self._driver, self._db, results)

        # Cross-regulation expansion: if the retrieved provisions have
        # cross-regulation CITES links, also retrieve the "other side"
        # provisions from the other regulation(s).  This ensures that
        # when a question spans multiple regulations, both sides of the
        # cross-reference chain appear in context.
        if not target_celexes:
            with self._driver.session(database=self._db) as s:
                reverse = s.run(
                    _REVERSE_XREF_CYPHER, ids=reranked_ids
                ).data()
            if reverse:
                rev_ids = [r["article_id"] for r in reverse]
                rev_expanded = _traversal.expand(self._driver, self._db, rev_ids)
                for r in rev_expanded:
                    r["score"] = 0.0
                    r["matched_leaf_id"] = None
                    r["_cross_reg_expansion"] = True
                results.extend(rev_expanded)

        return results

    # ------------------------------------------------------------------
    # Whole-graph lookup indexes (cached)
    # ------------------------------------------------------------------

    def reference_index(self) -> dict[str, tuple[str, str]]:
        """Return ``{node_id: (display_ref, regulation)}`` for **every** provision.

        Unlike the per-query retrieved bag, this covers the entire graph, so the
        citation resolver can render a human-readable reference for a real
        provision the model cited but that retrieval did not surface (e.g. AI Act
        Article 25 in an importer-obligations answer).  Ids the model *invents*
        (a paragraph/point that does not exist) are simply absent here, so they
        are still dropped rather than laundered into a clean-looking citation.

        Fetched once from Neo4j and cached for the retriever's lifetime.  Node
        ids are stable internal keys; only the ``(display_ref, regulation)`` pair
        is ever shown to a reader.
        """
        if self._reference_index_cache is None:
            self._reference_index_cache = _traversal.load_reference_index(
                self._driver, self._db,
            )
        return self._reference_index_cache

    def get_defined_terms_index(self) -> dict[str, str]:
        """Return ``{lowercase_term: term_normalized}`` for all DefinedTerm nodes.

        The index is fetched once from Neo4j and cached for the lifetime of
        the retriever instance.  Used by the agent to detect which regulatory
        terms appear in a user question so it can enrich context with their
        legal definitions.
        """
        if self._term_index_cache is None:
            self._term_index_cache = _traversal.load_defined_terms_index(
                self._driver, self._db,
            )
        return self._term_index_cache

    # ------------------------------------------------------------------
    # DefinedTerm lookups
    # ------------------------------------------------------------------

    def find_by_term(self, term: str) -> list[dict[str, Any]]:
        """Exact-match lookup for a :DefinedTerm by its normalized term name.

        Normalises *term* the same way the extraction pipeline does
        (lowercased, whitespace → underscore) and performs a direct Neo4j
        lookup — no embedding needed.

        For each match the full :Point provision that defines the term is
        returned alongside its parent article for context.

        Parameters
        ----------
        term:
            The term to look up, e.g. ``"provider"``, ``"AI system"``, or
            ``"high-risk AI system"``.

        Returns
        -------
        list of result dicts, each containing:
          ``term``, ``term_normalized``, ``category``, ``regulation``,
          ``celex``, ``definition_text``, ``source_provision_id``,
          ``article_ref``, ``article_path``
        """
        return _traversal.find_by_term(self._driver, self._db, term)

    def find_by_category(self, category: str, celex: str | None = None) -> list[dict[str, Any]]:
        """Return all :DefinedTerm nodes for a given semantic category.

        Parameters
        ----------
        category:
            One of ``"actor"``, ``"system"``, ``"data"``, ``"document"``,
            ``"process"``, ``"concept"``, ``"body"``, or ``"other"``.
        celex:
            Optional CELEX filter (e.g. ``"32024R1689"`` for AI Act only).

        Returns
        -------
        list of dicts with ``term``, ``category``, ``regulation``, ``celex``,
        ``source_provision_id``.
        """
        return _traversal.find_by_category(self._driver, self._db, category, celex)

    # ------------------------------------------------------------------
    # Direct lookups (refs / ids / roles)
    # ------------------------------------------------------------------

    def retrieve_by_refs(
        self,
        refs: list[str],
        celex_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Direct lookup of provisions by their display_ref (e.g. 'Annex I', 'Article 26').

        Performs an exact case-insensitive match against ``display_ref``,
        completely bypassing vector similarity.  Ideal for structural /
        navigational questions that explicitly name a provision.

        The matched provisions are then expanded via the same Cypher graph
        traversal used by :meth:`retrieve`, so children and cross-references
        are included in the result.

        Parameters
        ----------
        refs:
            Normalised provision references extracted from the question,
            e.g. ``['Annex I', 'Article 26']``.
        celex_filter:
            Optional set of CELEX codes to restrict results to.
        """
        return _traversal.retrieve_by_refs(self._driver, self._db, refs, celex_filter)

    def retrieve_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Direct lookup + graph expansion of provisions by exact node id.

        Unlike :meth:`retrieve_by_refs`, which matches the non-unique
        ``display_ref``, this resolves the stable, unique node ``id``.  It is
        therefore the safe way to promote CITES-edge targets — whose ids the
        graph has already resolved into ``cited_provisions`` — into first-class
        citable provisions, without risking the display_ref ambiguity that
        would let a promotion land on the wrong node.

        Returns the same expanded shape as :meth:`retrieve` (children,
        cross-references, interpretive links included), de-duplicated and with a
        neutral score so callers can order/append them as low-priority context.
        """
        return _traversal.retrieve_by_ids(self._driver, self._db, ids)

    def retrieve_by_roles(
        self,
        role_specs: list[tuple[str, str]],
        *,
        k: int = 8,
        query_vec: np.ndarray | None = None,
        target_celexes: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the obligations of one or more resolved actor roles.

        The role's ``OBLIGATION_OF`` set *is* the statutory answer to a
        role-obligation question, so it is treated as a guaranteed candidate set
        rather than a vector top-k: article-grained obligations are preferred
        over annex/recital fragments, and — when ``query_vec`` is supplied — the
        set is ranked by cosine relevance to the question so the on-point
        articles (e.g. Article 53 for "GPAI provider obligations") lead and
        survive trimming. This replaces the previous "first ``k`` in arbitrary
        Neo4j order" selection, which silently dropped relevant obligations.

        Parameters
        ----------
        role_specs:
            ``[(term_normalized, celex), ...]`` pairs resolved by the agent.
        k:
            Lower bound on the obligation budget. The effective cap is
            ``max(k, _ROLE_OBLIGATION_CAP)`` so a small caller ``k`` (e.g. the
            eval's k=8) never truncates a role's complete article set.
        query_vec:
            Optional query embedding (``encode_as_query``). When given, ranks
            obligations by cosine relevance; otherwise falls back to the
            Cypher's article-first ordering.
        target_celexes:
            When provided, drops obligations outside these regulations. The
            Cypher expands ``EQUIVALENT_ROLE`` / ``INCLUDES_ROLE`` across
            regulations (e.g. AI-Act ``provider`` ≡ MDR ``manufacturer``), which
            is wanted only when the question actually spans them; scoping keeps a
            single-regulation question from inheriting another reg's obligations.
            The filter is applied *inside* the Cypher (before its ``LIMIT``) so
            in-scope obligations are never crowded out of the row cap by
            cross-regulation ones (MDR ids sort ahead of AI-Act ids).
        """
        return _traversal.retrieve_by_roles(
            self._driver, self._db, self._dense, role_specs,
            k=k, query_vec=query_vec, target_celexes=target_celexes,
        )

    # ------------------------------------------------------------------
    # Community-based (summary-first) retrieval
    # ------------------------------------------------------------------

    def retrieve_by_communities_hierarchical(
        self,
        question: str,
        *,
        k_communities: int = 5,
        k_provisions: int = 20,
        target_celexes: set[str] | None = None,
        query_vec: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Community-first retrieval: search summaries, then fetch member provisions.

        Two-stage search:
        1. Encode question (or use *query_vec*) → cosine similarity against
           ~300 community summary embeddings → top-``k_communities``.
        2. For each matched community, fetch up to ``k_provisions`` member
           Provision nodes (anchored to parent kinds), then run the same
           graph expansion used by :meth:`retrieve`.

        Returns the same dict shape as :meth:`retrieve` so results can be
        merged transparently.  Each provision dict gains two extra keys:

        * ``community_id`` — the community it came from
        * ``_community_retrieval`` — ``True`` for attribution in audit traces
        * ``community_summary`` — the community's summary text

        Falls back to an empty list when no community embeddings are loaded.
        """
        communities = self._communities
        communities.ensure_loaded()
        if not communities.ids or communities.matrix is None:
            return []

        if query_vec is not None:
            q_vec = query_vec
        else:
            q_vec = self._dense.encode_query(question)

        # Step 1 — rank communities by summary similarity
        top_community_ids, top_community_meta = communities.rank(
            q_vec, k_communities, target_celexes,
        )
        if not top_community_ids:
            return []

        # Step 2 — fetch member provisions for matched communities
        provision_to_community = communities.fetch_members(
            top_community_ids, k_provisions,
        )
        if not provision_to_community:
            return []

        top_ids = list(dict.fromkeys(provision_to_community))  # preserves order, deduplicates

        # Step 3 — graph expansion (same as retrieve())
        results = _traversal.expand(self._driver, self._db, top_ids)

        # Attach community metadata to each result
        for r in results:
            cid = provision_to_community.get(r["article_id"], "")
            meta = top_community_meta.get(cid, {})
            r["community_id"] = cid
            r["community_summary"] = meta.get("summary_text", "")
            r["score"] = meta.get("score", 0.0)
            r["matched_leaf_id"] = None
            r["_community_retrieval"] = True

        _traversal.expand_cited_containers(self._driver, self._db, results)
        return results

    def get_all_community_summaries(self, *, level: int = 1) -> list[dict]:
        """Return all Community summaries for the given level.

        Used by the map-reduce pass in the community_summary_search route.
        Each dict has keys: ``id``, ``summary_text``, ``regulations``, ``label``.

        Level-0 results include summaries loaded at startup.
        Level-1 results are the chapter-level aggregation summaries.
        """
        return self._communities.summaries(level=level)

    # ------------------------------------------------------------------
    # Legal reasoning chain retrieval
    # ------------------------------------------------------------------

    def retrieve_by_chain(
        self,
        refs: list[str],
        celex: str,
        *,
        seed_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve provisions reachable via legal reasoning edges.

        Given one or more seed provision references (e.g. ``["Article 6"]``),
        this method:

        1. Resolves the seed nodes in Neo4j by ``display_ref`` + ``celex``.
        2. Traverses ``TRIGGERS_OBLIGATION_CLUSTER`` and
           ``IS_PREREQUISITE_FOR`` edges up to 2 hops.
        3. Returns the expanded graph context for all reachable provisions
           (same structure as :meth:`retrieve`).

        Falls back to :meth:`retrieve_by_refs` when no legal reasoning edges
        exist for the seed provisions (e.g. for newly loaded regulations
        before the edge loading script has run).

        Parameters
        ----------
        refs:
            List of ``display_ref`` strings to start from.
        celex:
            CELEX of the regulation.
        seed_only:
            When ``True``, skip the chain traversal and return only the seed
            provisions' expanded context.  Useful for provisions that are
            already a complete obligation cluster (e.g. ``Article 26``).
        """
        return _traversal.retrieve_by_chain(
            self._driver, self._db, refs, celex, seed_only=seed_only,
        )

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
