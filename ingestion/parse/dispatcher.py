"""Dispatcher for regulation parsers.

Holds the implementation of :func:`parse_document`, which looks up the
appropriate parser in :data:`PARSER_REGISTRY`, normalises its output and
writes ``parsed.json`` into the target directory. This keeps
``ingestion.parse.__init__`` lightweight while preserving the public API.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from domain.legislation_catalog import LEGISLATION
from canonicalization.text_enrichment import enrich_text_for_analysis
from .base.registry import PARSER_REGISTRY
from .normalizer import normalize_consolidated_html
from .semantic_layer.definitions import extract_defined_terms


def _supplement_preamble(
    provisions: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    html_file: Path,
    parser,
    celex: str,
    regulation_id: str,
    lang: str,
) -> int:
    """Graft the original act's preamble into a consolidated parse (in place).

    Consolidated EUR-Lex texts (CONSLEG) legally omit the preamble, so the
    consolidated parse of GDPR/MDR/IVDR carries zero recitals — while the AI
    Act (parsed from the original CELEX) carries all 180.  When the pipeline
    has cached the original act as ``raw_preamble.html`` next to the main
    document, this parses it with the same parser and grafts the preamble
    subtree (preamble → citations + recitals) under the consolidated document
    root, before the enacting terms.

    Safe by construction: both parses run under the same canonical CELEX, so
    the id schemes are identical (``{celex}_document``, ``{celex}_art_N``) and
    the preamble subtree's parent/path pointers already reference the
    consolidated root id.  Recital-sourced cross-references are carried over —
    their article targets resolve against the consolidated body because the
    ids match.

    Returns the number of recitals grafted (0 = nothing to do).
    """
    if any(p.get("kind") == "preamble" for p in provisions):
        return 0
    preamble_file = Path(html_file).parent / "raw_preamble.html"
    if not preamble_file.exists():
        return 0

    pre_html = normalize_consolidated_html(preamble_file.read_text(encoding="utf-8"))
    result = parser(pre_html, celex, regulation_id, lang=lang)
    if isinstance(result, dict):
        pre_provs = result.get("provisions", [])
        pre_rels = result.get("relations", [])
    else:
        pre_provs, pre_rels = result

    existing_ids = {p["id"] for p in provisions}
    subtree = [
        p for p in pre_provs
        if p.get("kind") in ("preamble", "citation", "recital")
        and p["id"] not in existing_ids
    ]
    recitals = sum(1 for p in subtree if p.get("kind") == "recital")
    if not recitals:
        return 0

    # Graft under the consolidated root, before the enacting terms, so
    # HAS_PART ordering reflects legal document order.
    root = next((p for p in provisions if p.get("kind") == "document"), None)
    preamble_node = next(p for p in subtree if p.get("kind") == "preamble")
    if root is not None:
        preamble_node["parent_id"] = root["id"]
        preamble_node["path"] = [root["id"]]
        root["children"].insert(0, preamble_node["id"])
        insert_at = provisions.index(root) + 1
    else:
        insert_at = 0
    provisions[insert_at:insert_at] = subtree

    # Recital/citation-sourced cross-references only; the original parse's
    # enacting-terms relations would duplicate the consolidated ones.
    subtree_ids = {p["id"] for p in subtree}
    relations.extend(r for r in pre_rels if r.get("source") in subtree_ids)
    return recitals


def _stamp_regulation_provenance(provisions: List[Dict[str, Any]]) -> None:
    """Stamp kind-aware ``binding_force`` + ``source_type`` (in place).

    Recitals/citations/preamble aid interpretation but impose no obligations
    (settled CJEU position: a recital cannot derogate from an operative
    provision). The previous blanket ``"binding"`` stamp silently defeated the
    loader's kind-default on every rebuild — the loader honours a parsed value
    over its own default — so a full re-ingest would have reverted the
    13 Jul 2026 interpretive-recitals migration.
    """
    for provision in provisions:
        provision["binding_force"] = (
            "interpretive"
            if provision.get("kind") in ("preamble", "citation", "recital")
            else "binding"
        )
        provision["source_type"] = "regulation"


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

    # Read HTML content and normalize if consolidated
    html_content = Path(html_file).read_text(encoding="utf-8")
    html_content = normalize_consolidated_html(html_content)
    regulation_id = LEGISLATION.get(celex, {}).get("name", celex)

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

    # Consolidated sources omit the preamble; graft it from the cached
    # original-act HTML when present (see _supplement_preamble).
    grafted = _supplement_preamble(
        provisions, relations, Path(html_file), parser, celex,
        regulation_id, lang,
    )
    if grafted:
        logging.getLogger(__name__).info(
            "Preamble supplement: grafted %d recital(s) into %s.", grafted, celex,
        )

    regulation_name = LEGISLATION.get(celex, {}).get("name")
    out: Dict[str, Any] = {
        "graph_version": "0.1",
        "celex_id": celex,
        "regulation_id": regulation_name or celex,
        "source_name": regulation_name or "unknown",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "provisions": provisions,
        "relations": relations,
    }

    _stamp_regulation_provenance(provisions)

    # Enrich text_for_analysis for multi-granularity embeddings
    enrich_text_for_analysis(provisions)

    # Extract DefinedTerm nodes and DEFINED_BY relations (Layer 1 semantic layer)
    defined_terms, dt_relations = extract_defined_terms(
        provisions, celex, regulation_name or celex
    )
    relations.extend(dt_relations)
    out["defined_terms"] = defined_terms

    if debug_roman_stats is not None:
        out["debug_roman_stats"] = debug_roman_stats

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "parsed.json"
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    return out_file
