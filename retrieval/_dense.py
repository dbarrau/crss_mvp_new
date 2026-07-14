"""Dense (embedding) index: the in-memory cosine-similarity channel.

Owns the SentenceTransformer encoder and the numpy matrix of all provision
and guidance embeddings, loaded from Neo4j once at startup.  No Neo4j Vector
Index plugin required — similarity is computed with numpy (~1ms for 6k
vectors).
"""
from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval._config import PASSAGE_PREFIX, QUERY_PREFIX
from retrieval._cypher import _PARENT_KINDS

logger = logging.getLogger(__name__)


class DenseIndex:
    """Embedding matrix + encoder + leaf→anchor resolution.

    Attributes are deliberately public within the package: the facade's
    ``retrieve()`` orchestrates the RRF fusion directly over ``ids`` /
    ``celexes`` / ``matrix``.
    """

    def __init__(self, driver, db: str, model_name: str):
        self._model = SentenceTransformer(model_name)
        self._driver = driver
        self._db = db

        self.ids: list[str] = []
        self.kinds: list[str] = []
        self.path_strings: list[str] = []
        self.celexes: list[str] = []
        self.id_to_kind: dict[str, str] = {}
        self.id_index: dict[str, int] = {}
        self.matrix: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        """Fetch all embedded provisions and guidance nodes from Neo4j into a numpy matrix."""
        with self._driver.session(database=self._db) as s:
            rows = s.run(
                "MATCH (n:Provision) "
                "WHERE n.embedding IS NOT NULL "
                "RETURN n.id AS id, n.kind AS kind, "
                "n.path_string AS path_string, n.celex AS celex, "
                "n.embedding AS emb "
                "UNION ALL "
                "MATCH (n:Guidance) "
                "WHERE n.embedding IS NOT NULL "
                "RETURN n.id AS id, n.kind AS kind, "
                "n.path_string AS path_string, n.celex AS celex, "
                "n.embedding AS emb"
            ).data()

        if not rows:
            logger.warning("No embeddings found in Neo4j. Run embed_provisions.py first.")
            return

        self.ids = [r["id"] for r in rows]
        self.kinds = [r["kind"] for r in rows]
        self.path_strings = [r["path_string"] or "" for r in rows]
        self.celexes = [r["celex"] or "" for r in rows]
        self.id_to_kind = dict(zip(self.ids, self.kinds))
        self.id_index = {pid: i for i, pid in enumerate(self.ids)}
        self.matrix = np.array([r["emb"] for r in rows], dtype=np.float32)
        logger.info(
            "Loaded %d provision embeddings (%d dims) into memory.",
            len(self.ids), self.matrix.shape[1],
        )

    def encode_passage(self, text: str) -> np.ndarray:
        """Encode *text* with the passage prefix, matching the provision embedding space.

        Use this for HyDE (Hypothetical Document Embedding): the generated
        hypothetical answer should be encoded as a passage so it sits in the
        same embedding space as the stored provision embeddings.
        """
        return self._model.encode(
            PASSAGE_PREFIX + text, normalize_embeddings=True,
        ).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode *text* with the query prefix.

        Use this when you want to embed the raw question for community-level
        retrieval, bypassing HyDE.  Community summaries are semantically rich
        enough to match a plain question vector; using a query prefix (rather
        than a passage prefix) aligns with how the model was trained.
        """
        return self._model.encode(
            QUERY_PREFIX + text, normalize_embeddings=True,
        ).astype(np.float32)

    def find_anchor(self, node_id: str, node_kind: str, path_string: str) -> str:
        """Return the nearest parent-kind ancestor ID for the given node.

        If the node itself is a parent kind, returns it directly.
        Otherwise walks up path_string (root → immediate parent, "/"-separated)
        to find the closest ancestor whose kind is in _PARENT_KINDS.
        Falls back to node_id if no parent-kind ancestor is found.
        """
        if node_kind in _PARENT_KINDS:
            return node_id
        for ancestor_id in reversed(path_string.split("/") if path_string else []):
            if ancestor_id and self.id_to_kind.get(ancestor_id) in _PARENT_KINDS:
                return ancestor_id
        return node_id
