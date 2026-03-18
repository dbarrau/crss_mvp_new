"""Graph-aware vector retrieval over Neo4j provision embeddings.

Performs hybrid retrieval:
1. In-memory cosine similarity → top-k parent provisions (numpy, fast for <10k nodes)
2. Graph traversal → expand children via HAS_PART
3. Cross-reference expansion → follow CITES edges

No Neo4j Vector Index plugin required — embeddings are loaded into
memory at startup and similarity is computed with numpy.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
logger = logging.getLogger(__name__)

QUERY_PREFIX = "query: "

# Parent-level kinds we want the vector search to return.
_PARENT_KINDS = frozenset({
    "article", "annex_section", "recital", "section",
})

# Graph expansion: given top-k article IDs, fetch children + cross-refs.
_EXPAND_CYPHER = """\
UNWIND $ids AS aid
MATCH (art:Provision {id: aid})

OPTIONAL MATCH (art)-[:HAS_PART*1..5]->(leaf)
WHERE leaf.text_for_analysis IS NOT NULL

WITH art,
     collect(DISTINCT {
       id: leaf.id,
       kind: leaf.kind,
       text: leaf.text_for_analysis,
       raw_text: leaf.text,
       ref: leaf.display_ref
     })[..25] AS children

OPTIONAL MATCH (art)-[:HAS_PART*1..5]->()-[:CITES]->(cited:Provision)
WHERE cited.text_for_analysis IS NOT NULL

RETURN
  art.id              AS article_id,
  art.celex           AS celex,
  art.regulation_id   AS regulation,
  art.display_ref     AS article_ref,
  art.display_path    AS article_path,
  art.text_for_analysis AS article_text,
  children,
  collect(DISTINCT {
    id:   cited.id,
    ref:  cited.display_ref,
    text: cited.text_for_analysis
  })[..5] AS cited_provisions
"""


class GraphRetriever:
    """Hybrid vector + graph retriever backed by Neo4j.

    On init, loads all embedded provisions into a numpy matrix.
    Queries compute cosine similarity in-memory (~1ms for 6k vectors),
    then expand top-k results via Cypher graph traversal.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        self._model = SentenceTransformer(model_name)
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

        # Load the embedding index into memory
        self._ids: list[str] = []
        self._kinds: list[str] = []
        self._path_strings: list[str] = []
        self._id_to_kind: dict[str, str] = {}
        self._matrix: np.ndarray | None = None
        self._load_index()

    def _load_index(self) -> None:
        """Fetch all embedded provisions from Neo4j into a numpy matrix."""
        with self._driver.session(database=self._db) as s:
            rows = s.run(
                "MATCH (n:Provision) "
                "WHERE n.embedding IS NOT NULL "
                "RETURN n.id AS id, n.kind AS kind, "
                "n.path_string AS path_string, n.embedding AS emb"
            ).data()

        if not rows:
            logger.warning("No embeddings found in Neo4j. Run embed_provisions.py first.")
            return

        self._ids = [r["id"] for r in rows]
        self._kinds = [r["kind"] for r in rows]
        self._path_strings = [r["path_string"] or "" for r in rows]
        self._id_to_kind = dict(zip(self._ids, self._kinds))
        self._matrix = np.array([r["emb"] for r in rows], dtype=np.float32)
        logger.info(
            "Loaded %d provision embeddings (%d dims) into memory.",
            len(self._ids), self._matrix.shape[1],
        )

    def _find_anchor(self, node_id: str, node_kind: str, path_string: str) -> str:
        """Return the nearest parent-kind ancestor ID for the given node.

        If the node itself is a parent kind, returns it directly.
        Otherwise walks up path_string (root → immediate parent, "/"-separated)
        to find the closest ancestor whose kind is in _PARENT_KINDS.
        Falls back to node_id if no parent-kind ancestor is found.
        """
        if node_kind in _PARENT_KINDS:
            return node_id
        for ancestor_id in reversed(path_string.split("/") if path_string else []):
            if ancestor_id and self._id_to_kind.get(ancestor_id) in _PARENT_KINDS:
                return ancestor_id
        return node_id

    def retrieve(self, question: str, k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k provisions with children and cross-references."""
        if self._matrix is None or len(self._ids) == 0:
            return []

        # Encode query
        q_vec = self._model.encode(
            QUERY_PREFIX + question, normalize_embeddings=True
        ).astype(np.float32)

        # Cosine similarity (embeddings are already L2-normalized)
        scores = self._matrix @ q_vec

        # Score all embedded nodes, then map each to its nearest parent-kind
        # ancestor. This lets deep leaf nodes (point, roman_item) surface their
        # parent article/section with a score driven by the most specific match.
        score_map: dict[str, float] = {}
        leaf_map: dict[str, str] = {}
        for idx, sc in sorted(
            enumerate(scores.tolist()), key=lambda x: x[1], reverse=True
        ):
            anchor_id = self._find_anchor(
                self._ids[idx], self._kinds[idx], self._path_strings[idx]
            )
            if anchor_id not in score_map:
                score_map[anchor_id] = float(sc)
                leaf_map[anchor_id] = self._ids[idx]
            if len(score_map) >= k:
                break

        top_ids = list(score_map.keys())

        # Graph expansion via Cypher
        with self._driver.session(database=self._db) as s:
            results = s.run(_EXPAND_CYPHER, ids=top_ids).data()

        # Attach scores, matched leaf, and sort
        for r in results:
            r["score"] = score_map.get(r["article_id"], 0.0)
            r["matched_leaf_id"] = leaf_map.get(r["article_id"])
        results.sort(key=lambda r: r["score"], reverse=True)

        return results

    def get_defined_terms_index(self) -> dict[str, str]:
        """Return ``{lowercase_term: term_normalized}`` for all DefinedTerm nodes.

        The index is fetched once from Neo4j and cached for the lifetime of
        the retriever instance.  Used by the agent to detect which regulatory
        terms appear in a user question so it can enrich context with their
        legal definitions.
        """
        if not hasattr(self, "_term_index"):
            with self._driver.session(database=self._db) as s:
                rows = s.run(
                    "MATCH (d:DefinedTerm) "
                    "RETURN d.term AS term, d.term_normalized AS tn"
                ).data()
            self._term_index: dict[str, str] = {
                r["term"].lower(): r["tn"] for r in rows
            }
        return self._term_index

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
        # Mirror the normalisation used in definitions.py
        import re as _re
        term_normalized = _re.sub(r"\s+", "_", term.strip().lower())

        cypher = """\
MATCH (d:DefinedTerm {term_normalized: $term_normalized})
MATCH (d)-[:DEFINED_BY]->(p:Provision)
OPTIONAL MATCH (p)<-[:HAS_PART]-(art:Provision)
RETURN
    d.term                AS term,
    d.term_normalized     AS term_normalized,
    d.category            AS category,
    d.regulation          AS regulation,
    d.celex               AS celex,
    p.text                AS definition_text,
    p.id                  AS source_provision_id,
    art.display_ref       AS article_ref,
    art.display_path      AS article_path
"""
        with self._driver.session(database=self._db) as s:
            return s.run(cypher, term_normalized=term_normalized).data()

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
        if celex:
            cypher = """\
MATCH (d:DefinedTerm {category: $category, celex: $celex})
RETURN d.term AS term, d.term_normalized AS term_normalized,
       d.category AS category, d.regulation AS regulation,
       d.celex AS celex, d.source_provision_id AS source_provision_id
ORDER BY d.term_normalized
"""
            with self._driver.session(database=self._db) as s:
                return s.run(cypher, category=category, celex=celex).data()
        else:
            cypher = """\
MATCH (d:DefinedTerm {category: $category})
RETURN d.term AS term, d.term_normalized AS term_normalized,
       d.category AS category, d.regulation AS regulation,
       d.celex AS celex, d.source_provision_id AS source_provision_id
ORDER BY d.celex, d.term_normalized
"""
            with self._driver.session(database=self._db) as s:
                return s.run(cypher, category=category).data()

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
