"""Parsing package dispatcher for ingestion.parse

Provides a thin adapter `parse_document` used by the pipeline. The
function discovers a parser via :data:`PARSER_REGISTRY` and normalises
its output to a single JSON file written to `out_dir/parsed.json`.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from domain.regulations_catalog import REGULATIONS
from .registry import PARSER_REGISTRY


def parse_document(html_file: Path, lang: str, celex: str, out_dir: Path) -> Path:
    """Dispatch to the appropriate regulation parser and write JSON output.

    Args:
        html_file: Path to the raw HTML file.
        lang: Language code (EN/DE/FR).
        celex: CELEX identifier used to select parser.
        out_dir: Directory where parsed JSON will be written.

    Returns:
        Path to the written JSON file.
    """
    parser = PARSER_REGISTRY.get(celex)
    if not parser:
        raise KeyError(f"No parser registered for CELEX {celex}")

    result = parser(html_file, lang)

    # Normalise parser output to dict with provisions + relations
    provisions: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []

    if isinstance(result, tuple) and len(result) == 2:
        provisions, relations = result
    elif isinstance(result, list):
        provisions = result
    else:
        # Unknown return type – attempt to coerce
        try:
            provisions = list(result)  # type: ignore
        except Exception:
            raise TypeError("Parser returned unexpected type; expected List or (List, List)")

    regulation_name = REGULATIONS.get(celex, {}).get("name")
    out = {
        "graph_version": "0.1",
        "celex_id": celex,
        "regulation_id": regulation_name or celex,
        "source_name": regulation_name or "unknown",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "provisions": provisions,
        "relations": relations,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "parsed.json"
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    return out_file
