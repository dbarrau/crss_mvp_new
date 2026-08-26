"""
Guidance Catalog — merged view
===============================

Single family-agnostic registry over every guidance document CRSS can
ingest, regardless of publisher: MDCG (:mod:`domain.mdcg_catalog`) and the
AI Office (:mod:`domain.ai_office_catalog`). The per-publisher dicts stay
separate (their *data* differs, not their handling), but routing, the build
DAG, and every "which guidance do we have?" consumer should read this merged
view so adding a new family is a one-line import here — never a new ``if``
branch downstream.

Each entry carries a ``parse_profile`` (defaulting to ``"mdcg"``) that selects
the LlamaParse prompt + post-processing profile in
:mod:`ingestion.parse.guidance.profiles`.
"""
from __future__ import annotations

from domain.mdcg_catalog import (
    MDCG_DOCUMENTS,
    DEFAULT_INGEST_TIER,  # re-exported so consumers have one import site
)
from domain.ai_office_catalog import AI_OFFICE_DOCUMENTS

# The merged registry. Later families are added by importing their dict above
# and unpacking it here — nothing else changes.
GUIDANCE_DOCUMENTS: dict[str, dict] = {**MDCG_DOCUMENTS, **AI_OFFICE_DOCUMENTS}

# Guard against ID collisions between families (would silently shadow an entry).
_overlap = set(MDCG_DOCUMENTS) & set(AI_OFFICE_DOCUMENTS)
if _overlap:  # pragma: no cover - defensive
    raise ValueError(f"Guidance catalog ID collision across families: {sorted(_overlap)}")

# Default parse profile when an entry does not name one (MDCG predates the field).
DEFAULT_PARSE_PROFILE = "mdcg"

__all__ = [
    "GUIDANCE_DOCUMENTS",
    "DEFAULT_INGEST_TIER",
    "DEFAULT_PARSE_PROFILE",
    "default_doc_ids",
    "get_parse_profile",
]


def default_doc_ids(max_tier: int = DEFAULT_INGEST_TIER) -> list[str]:
    """Guidance doc IDs to ingest by default — those with ``tier <= max_tier``,
    in catalog order (MDCG first, then AI Office). Untagged entries are treated
    as lowest priority."""
    return [k for k, v in GUIDANCE_DOCUMENTS.items() if v.get("tier", 99) <= max_tier]


def get_parse_profile(doc_id: str) -> str:
    """Return the parse-profile name for a guidance doc (``"mdcg"`` default)."""
    meta = GUIDANCE_DOCUMENTS.get(doc_id, {})
    return meta.get("parse_profile", DEFAULT_PARSE_PROFILE)
