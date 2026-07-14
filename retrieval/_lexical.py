"""Lexical (BM25) retrieval channel via a Neo4j full-text index.

Owns the index lifecycle (created idempotently on startup) and the sanitised
BM25 search.  Disabled entirely with ``CRSS_LEXICAL=0``; degrades gracefully
to dense-only retrieval when the index cannot be created or queried.
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

# Name of the Neo4j full-text (Lucene/BM25) index over provision text.
_FULLTEXT_INDEX = "provision_fulltext"

# Lucene query-syntax special characters.  These are stripped from the user
# question before it is passed to the full-text index so that punctuation
# (e.g. "?", "/", parentheses in "Article 6(3)") cannot raise a query-parse
# error or be misinterpreted as Lucene operators.
_LUCENE_SPECIALS = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')


def _sanitise_lucene(text: str) -> str:
    """Strip Lucene operator characters and collapse whitespace.

    The full-text index uses OR semantics across the remaining terms, so a
    cleaned bag-of-words query is sufficient for BM25 scoring.
    """
    cleaned = _LUCENE_SPECIALS.sub(" ", text)
    return " ".join(cleaned.split())


class LexicalChannel:
    """BM25 full-text search over :Provision and :Guidance nodes."""

    def __init__(self, driver, db: str):
        self._driver = driver
        self._db = db
        self.enabled = False
        if os.environ.get("CRSS_LEXICAL", "1") != "0":
            self._ensure_fulltext_index()

    def _ensure_fulltext_index(self) -> None:
        """Create the full-text (BM25) index over provision text if absent.

        Idempotent — ``IF NOT EXISTS`` makes re-runs cheap.  The index spans
        both :Provision and :Guidance on ``text_for_analysis`` (which carries
        the context-prefixed ancestor headings, so an annex's title term like
        "TECHNICAL DOCUMENTATION" is searchable) and ``display_ref``.
        """
        try:
            with self._driver.session(database=self._db) as s:
                s.run(
                    f"CREATE FULLTEXT INDEX {_FULLTEXT_INDEX} IF NOT EXISTS "
                    "FOR (n:Provision|Guidance) "
                    "ON EACH [n.text_for_analysis, n.display_ref]"
                )
                # Index population is async; wait so the first query sees it.
                s.run("CALL db.awaitIndexes(30000)")
            self.enabled = True
            logger.info("Full-text BM25 index ready: %s", _FULLTEXT_INDEX)
        except Exception as exc:
            logger.warning(
                "Full-text index unavailable (%s) — dense-only retrieval.", exc
            )
            self.enabled = False

    def search(
        self,
        question: str,
        limit: int,
        celex_filter: set[str] | None = None,
    ) -> dict[str, int]:
        """Return ``{node_id: rank}`` from the BM25 full-text index (0-based).

        Returns an empty mapping when the lexical channel is disabled or the
        sanitised query is empty, so callers degrade gracefully to dense-only.

        ``celex_filter`` restricts hits to the in-scope regulation(s).  Without
        it, a cross-regulation query's lexically-verbose guidance docs (MDCG)
        can flood the result budget and starve the operative provision the
        query targets.
        """
        if not self.enabled:
            return {}
        lucene_q = _sanitise_lucene(question)
        if not lucene_q:
            return {}
        try:
            with self._driver.session(database=self._db) as s:
                # Exclude recitals/citations: they are verbose and repeat
                # operative vocabulary (e.g. "AI system"), so BM25 term-density
                # over-promotes them and displaces the terse normative
                # provision that actually answers the query.  The dense channel
                # still surfaces recitals when they are genuinely on-topic.
                rows = s.run(
                    f"CALL db.index.fulltext.queryNodes('{_FULLTEXT_INDEX}', $q) "
                    "YIELD node, score "
                    "WHERE NOT node.kind IN ['recital', 'citation', 'preamble'] "
                    "  AND ($celexes IS NULL OR node.celex IN $celexes) "
                    "RETURN node.id AS id ORDER BY score DESC LIMIT $lim",
                    q=lucene_q,
                    lim=limit,
                    celexes=list(celex_filter) if celex_filter else None,
                ).data()
            return {r["id"]: rank for rank, r in enumerate(rows)}
        except Exception as exc:
            logger.warning("Lexical search failed (%s) — dense-only.", exc)
            return {}
