"""Cross-encoder reranking of the widened cosine candidate pool.

Owns the CrossEncoder model (loaded once, device auto-picked) and the
blended re-scoring.  Disabled entirely with ``CRSS_RERANKER=0``; model
overrideable via ``CRSS_RERANKER_MODEL``.
"""
from __future__ import annotations

import logging
import os

from retrieval._config import (
    _RERANK_MAX_CHARS,
    _RERANK_MAX_TOKENS,
    _RERANK_WEIGHT,
)

logger = logging.getLogger(__name__)


class Reranker:
    """Blended cross-encoder + cosine re-scoring of retrieval candidates."""

    def __init__(self, cross_encoder) -> None:
        self._model = cross_encoder

    @classmethod
    def load_if_enabled(cls) -> "Reranker | None":
        """Load the cross-encoder unless disabled or unavailable.

        Returns ``None`` (→ cosine-only retrieval) when ``CRSS_RERANKER=0``
        or the model cannot be loaded, mirroring the previous inline
        try/except in ``GraphRetriever.__init__``.
        """
        if os.environ.get("CRSS_RERANKER", "1") == "0":
            return None
        _rr_model = os.environ.get(
            "CRSS_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        try:
            import torch
            from sentence_transformers import CrossEncoder as _CrossEncoder
            _device = (
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
            reranker = _CrossEncoder(
                _rr_model, max_length=_RERANK_MAX_TOKENS, device=_device
            )
            logger.info(
                "Cross-encoder reranker loaded: %s (device=%s)",
                _rr_model, _device,
            )
            return cls(reranker)
        except Exception as exc:
            logger.warning(
                "Reranker unavailable (%s) — cosine-only retrieval.", exc
            )
            return None

    def rerank(
        self,
        question: str,
        results: list[dict],
        k: int,
    ) -> list[dict]:
        """Refine ranking with the cross-encoder, then return the top-k.

        The cross-encoder score is *blended* with the original cosine score
        rather than replacing it.  Pure cross-encoder ordering tends to bury
        container nodes (e.g. "Annex II") and gate articles (e.g. "Article 43")
        that share vocabulary with their neighbours; blending lets a strong
        cosine match survive while still benefiting from cross-encoder
        precision.  Both scores are min-max normalised within the candidate
        set before blending so they are on a comparable scale.
        """
        pairs: list[tuple[str, str]] = []
        for r in results:
            text = r.get("article_text") or ""
            if len(text) < 200:
                # Container or thin node — enrich with children text so the
                # cross-encoder has enough signal (e.g. Annex II title alone
                # is unscoreable; its sub-items carry the actual content).
                children_text = " ".join(
                    (c.get("text") or "")[:400]
                    for c in (r.get("children") or [])[:8]
                ).strip()
                text = (text + " " + children_text).strip() or text
            pairs.append((question, text[:_RERANK_MAX_CHARS]))

        rr_scores = [
            float(s) for s in self._model.predict(pairs, show_progress_bar=False)
        ]
        cos_scores = [float(r.get("score", 0.0)) for r in results]

        def _normalise(xs: list[float]) -> list[float]:
            lo, hi = min(xs), max(xs)
            span = hi - lo
            if span < 1e-9:
                return [1.0 for _ in xs]
            return [(x - lo) / span for x in xs]

        rr_norm = _normalise(rr_scores)
        cos_norm = _normalise(cos_scores)
        for r, rr_raw, rr_n, cos_n in zip(results, rr_scores, rr_norm, cos_norm):
            r["rerank_score"] = rr_raw
            r["_blended_score"] = (
                _RERANK_WEIGHT * rr_n + (1.0 - _RERANK_WEIGHT) * cos_n
            )
        results.sort(key=lambda r: r.get("_blended_score", 0.0), reverse=True)
        return results[:k]
