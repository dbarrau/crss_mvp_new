"""Semantic enrichment helpers for parsed provisions.

This module centralises logic that interprets provision text to derive
obligations, obligation types and actor roles. It is currently used by
the ingestion parser but is placed under :mod:`canonicalization` to
respect the architectural separation between raw parsing and
canonical/semantic interpretation.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from domain.regulations_catalog import REGULATIONS
from domain.ontology.actor_roles import eu_ai_role_detector, mdr_role_detector
from ingestion.parse.semantic_layer.requirement_patterns import (
    is_requirement_text,
    classify_requirement_type,
)


def enrich_semantics(text: str, kind: str, celex: str, lang: str) -> Dict[str, Any]:
    """Return semantic enrichment for a provision's plain text.

    Args:
        text: Cleaned provision text (numbering stripped where possible).
        kind: Structural kind (article, paragraph, point, roman_item, annex, ...).
        celex: Regulation CELEX identifier.
        lang: Language code (EN/DE/FR).

    Returns:
        Dict with keys:
            - is_obligation (bool)
            - obligation_type (str | None)
            - roles (List[str])
            - semantic_role (str | None)
    """

    text_clean = text or ""
    lang_norm = (lang or "EN").upper()

    # Requirement / obligation cues via shared patterns
    # Use classify_requirement_type as the primary source of truth and fall back
    # to legacy obligation_patterns only when no requirement pattern matches.
    requirement_type = classify_requirement_type(text_clean, lang_norm)
    has_requirement = requirement_type != "other"

    # Legacy basic obligation cues kept as conservative fallback
    obligation_patterns = [
        r"\bshall\b",
        r"\bmust\b",
        r"\bare required to\b",
        r"\bis required to\b",
        r"\bresponsible for\b",
        r"\bensure\b",
        r"\bprohibited\b",
        r"\bnot permitted\b",
    ]
    legacy_obligation = any(re.search(p, text_clean, re.IGNORECASE) for p in obligation_patterns)

    # Definitions are not treated as obligations; other requirement types are.
    if has_requirement:
        if requirement_type in {"obligation", "prohibition", "permission"}:
            is_obligation = True
        else:  # "definition" or any future non-obligatory type
            is_obligation = False
    else:
        is_obligation = legacy_obligation

    # Regulation-specific role detection
    regulation_meta = REGULATIONS.get(celex, {})
    reg_type = regulation_meta.get("type")
    detected_roles: List[str] = []
    if reg_type == "medical_device_regulation":
        detected_roles = mdr_role_detector(text_clean, lang)
    elif reg_type == "ai_regulation":
        detected_roles = eu_ai_role_detector(text_clean, lang)

    # Coarse obligation_type classification based on keywords
    obligation_type = None
    if is_obligation:
        if re.search(r"vigilance", text_clean, re.IGNORECASE):
            obligation_type = "vigilance"
        elif re.search(r"post-market surveillance", text_clean, re.IGNORECASE):
            obligation_type = "post_market_surveillance"
        elif re.search(r"classification", text_clean, re.IGNORECASE):
            obligation_type = "classification_rule"
        elif re.search(r"risk management", text_clean, re.IGNORECASE):
            obligation_type = "risk_management"
        elif re.search(r"conformity assessment", text_clean, re.IGNORECASE):
            obligation_type = "conformity_assessment"
        elif re.search(r"clinical investigation", text_clean, re.IGNORECASE):
            obligation_type = "clinical_investigation"
        elif has_requirement and requirement_type in {"obligation", "prohibition", "permission", "definition"}:
            # Fall back to the coarse-grained requirement type when no more
            # specific topical category is detected.
            obligation_type = requirement_type
        else:
            obligation_type = "general_obligation"

    semantic_role = detected_roles[0] if detected_roles else None

    return {
        "is_obligation": is_obligation,
        "obligation_type": obligation_type,
        "roles": detected_roles,
        "semantic_role": semantic_role,
    }
