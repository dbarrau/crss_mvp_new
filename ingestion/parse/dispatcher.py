"""Dispatcher for regulation parsers.

Holds the implementation of :func:`parse_document`, which looks up the
appropriate parser in :data:`PARSER_REGISTRY`, normalises its output and
writes ``parsed.json`` into the target directory. This keeps
``ingestion.parse.__init__`` lightweight while preserving the public API.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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

    # Read HTML content and invoke parser
    html_content = Path(html_file).read_text(encoding="utf-8")
    regulation_id = REGULATIONS.get(celex, {}).get("name", celex)

    # The universal parser returns a dict with 'provisions' and 'relations'.
    # Older parsers returned (provisions, relations). Support both.
    result = parser(html_content, celex, regulation_id, lang=lang)

    # Normalise parser output to dict with provisions + relations
    provisions: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []
    debug_roman_stats: Dict[str, Any] | None = None

    # Normalise possible return shapes
    if isinstance(result, dict):
        provisions = result.get("provisions", [])
        relations = result.get("relations", [])
        # Optional debug payload from universal_eurlex_parser (Phase 2)
        if "debug_roman_stats" in result:
            debug_roman_stats = result["debug_roman_stats"]
    elif isinstance(result, tuple) and len(result) == 2:
        provisions, relations = result
    elif isinstance(result, list):
        provisions = result
    else:
        # Unknown return type – attempt to coerce to list
        try:
            provisions = list(result)  # type: ignore
        except Exception:
            raise TypeError("Parser returned unexpected type; expected dict, List or (List, List)")

    regulation_name = REGULATIONS.get(celex, {}).get("name")
    out: Dict[str, Any] = {
        "graph_version": "0.1",
        "celex_id": celex,
        "regulation_id": regulation_name or celex,
        "source_name": regulation_name or "unknown",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "provisions": provisions,
        "relations": relations,
    }

    if debug_roman_stats is not None:
        out["debug_roman_stats"] = debug_roman_stats

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "parsed.json"
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    return out_file
