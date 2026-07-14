"""Community-summary index for community-first (summary-first) retrieval.

Owns the lazily-loaded Level-0 community embedding matrix and the Level-1
chapter-summary list.  Ranking and member-fetching live here; the facade
(:mod:`retrieval.graph_retriever`) composes them with graph expansion.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Members of the matched communities, anchored to parent kinds, capped
# per community so one large community cannot starve the others.
_MEMBER_QUERY = """\
UNWIND $community_ids AS cid
MATCH (p:Provision)-[:MEMBER_OF]->(c:Community {id: cid})
WHERE p.kind IN ['article', 'annex_section', 'annex_subsection', 'annex_point',
                  'annex_part', 'annex_chapter', 'recital', 'section',
                  'guidance_section', 'guidance_subsection']
  AND p.text_for_analysis IS NOT NULL
WITH cid, p
ORDER BY p.hierarchy_depth ASC
WITH cid, collect(p.id)[..$per_community] AS ids
UNWIND ids AS pid
RETURN pid AS provision_id, cid AS community_id
"""


class CommunityIndex:
    """Lazy in-memory index of Community summary embeddings."""

    def __init__(self, driver, db: str):
        self._driver = driver
        self._db = db
        self._loaded = False
        self.ids: list[str] = []
        self.meta: list[dict] = []
        self.matrix: np.ndarray | None = None
        self.l1_summaries: list[dict] = []

    def ensure_loaded(self) -> None:
        """Lazily fetch Community summaries into an in-memory numpy matrix.

        Loads **Level-0** community embeddings into the cosine-similarity
        matrix used for community-first retrieval.  Level-1 (chapter-level)
        summaries are stored separately in ``l1_summaries`` and used for the
        map-reduce pass.
        """
        if self._loaded:
            return
        self._loaded = True
        with self._driver.session(database=self._db) as s:
            rows = s.run(
                "MATCH (c:Community) "
                "WHERE c.summary_embedding IS NOT NULL "
                "  AND (c.level IS NULL OR c.level = 0) "
                "RETURN c.id AS id, c.summary_text AS summary_text, "
                "c.member_count AS member_count, "
                "c.regulations AS regulations, "
                "c.summary_embedding AS emb"
            ).data()
            l1_rows = s.run(
                "MATCH (c:Community) "
                "WHERE c.level = 1 AND c.summary_text IS NOT NULL "
                "RETURN c.id AS id, c.summary_text AS summary_text, "
                "c.regulations AS regulations, c.label AS label, "
                "c.member_count AS member_count, "
                "c.summary_embedding AS emb"
            ).data()
        if not rows:
            logger.warning(
                "No community summary embeddings found. "
                "Run scripts/generate_community_summaries.py first."
            )
        else:
            self.ids = [r["id"] for r in rows]
            self.meta = [
                {
                    "id": r["id"],
                    "summary_text": r.get("summary_text") or "",
                    "member_count": r.get("member_count") or 0,
                    "regulations": r.get("regulations") or [],
                }
                for r in rows
            ]
            self.matrix = np.array(
                [r["emb"] for r in rows], dtype=np.float32
            )
            logger.info(
                "Loaded %d Level-0 community summary embeddings into memory.",
                len(self.ids),
            )

        # Level-1 summaries (no embedding search needed — map-reduce uses all)
        self.l1_summaries = [
            {
                "id": r["id"],
                "summary_text": r.get("summary_text") or "",
                "regulations": r.get("regulations") or [],
                "label": r.get("label") or r["id"],
                "member_count": r.get("member_count") or 0,
            }
            for r in l1_rows
        ]
        if l1_rows:
            logger.info(
                "Loaded %d Level-1 chapter-community summaries.",
                len(l1_rows),
            )

    def rank(
        self,
        q_vec: np.ndarray,
        k_communities: int,
        target_celexes: set[str] | None = None,
    ) -> tuple[list[str], dict[str, dict]]:
        """Rank communities by summary similarity; return (ids, meta-by-id).

        The per-community meta dict gains a ``score`` key.  Communities whose
        regulation tagging misses every target CELEX are skipped; untagged
        communities always pass the filter.
        """
        assert self.matrix is not None
        scores = self.matrix @ q_vec
        sorted_indices = scores.argsort()[::-1]

        top_community_ids: list[str] = []
        top_community_meta: dict[str, dict] = {}
        for idx in sorted_indices:
            cid = self.ids[idx]
            meta = self.meta[idx]
            # Optional CELEX filter: include community if any of its
            # regulations match, or if it has no regulation tagging.
            if target_celexes:
                regs = set(meta.get("regulations") or [])
                if regs and not regs.intersection(target_celexes):
                    continue
            top_community_ids.append(cid)
            top_community_meta[cid] = {**meta, "score": float(scores[idx])}
            if len(top_community_ids) >= k_communities:
                break
        return top_community_ids, top_community_meta

    def fetch_members(
        self,
        community_ids: list[str],
        k_provisions: int,
    ) -> dict[str, str]:
        """Return ``{provision_id: community_id}`` for the matched communities."""
        per_community = max(1, k_provisions // len(community_ids))
        with self._driver.session(database=self._db) as s:
            member_rows = s.run(
                _MEMBER_QUERY,
                community_ids=community_ids,
                per_community=per_community,
            ).data()
        return {
            row["provision_id"]: row["community_id"] for row in member_rows
        }

    def summaries(self, *, level: int = 1) -> list[dict]:
        """Return all Community summaries for the given level.

        Each dict has keys: ``id``, ``summary_text``, ``regulations``,
        ``label``.  Level-1 results are the chapter-level aggregation
        summaries; level-0 results come from the standard community meta list.
        """
        self.ensure_loaded()
        if level == 1:
            return list(self.l1_summaries)
        return [
            {
                "id": m["id"],
                "summary_text": m["summary_text"],
                "regulations": m["regulations"],
                "label": m["id"],
            }
            for m in self.meta
        ]
