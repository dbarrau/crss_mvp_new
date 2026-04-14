"""
canonicalization/crosslinker.py
===============================
Resolve CITES_EXTERNAL references whose target regulation is already loaded
in the Neo4j graph.  Creates concrete **CITES** edges between existing
Provision nodes — never creates new nodes.

After resolving, cleans up stale ExternalAct stub nodes and CITES_EXTERNAL
edges that correspond to loaded regulations.

Usage::

    python -m canonicalization.crosslinker          # reads .env for Neo4j creds
    python -m canonicalization.crosslinker --dry-run # preview without writing
    python -m canonicalization.crosslinker --cleanup # also remove stale ExternalAct nodes
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from domain.legislation_catalog import LEGISLATION
from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolution map: regulation "number" field → CELEX ID
# (derived from the single source of truth in domain/legislation_catalog.py)
# ---------------------------------------------------------------------------
CELEX_BY_NUMBER: dict[str, str] = {
    meta["number"]: celex for celex, meta in LEGISLATION.items()
}

_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "legislation"
_GUIDANCE_ROOT = Path(__file__).resolve().parent.parent / "data" / "guidance"

# ---------------------------------------------------------------------------
# 1. Discover resolvable CITES_EXTERNAL edges from parsed.json files
# ---------------------------------------------------------------------------

def discover_resolvable_refs() -> list[dict[str, Any]]:
    """Read all parsed.json files and return CITES_EXTERNAL relations
    whose ``number`` maps to a CELEX ID we have loaded.

    Scans both ``data/legislation/`` and ``data/guidance/`` directories.
    """
    results: list[dict[str, Any]] = []
    dirs_to_scan: list[Path] = []
    if _DATA_ROOT.is_dir():
        dirs_to_scan.extend(sorted(_DATA_ROOT.iterdir()))
    if _GUIDANCE_ROOT.is_dir():
        dirs_to_scan.extend(sorted(_GUIDANCE_ROOT.iterdir()))
    for celex_dir in dirs_to_scan:
        parsed = celex_dir / "EN" / "parsed.json"
        if not parsed.exists():
            continue
        with open(parsed, encoding="utf-8") as f:
            data = json.load(f)
        source_celex = data.get("celex_id", celex_dir.name)
        for rel in data.get("relations", []):
            if rel.get("type") != "CITES_EXTERNAL":
                continue
            props = rel.get("properties", {})
            number = props.get("number", "")
            target_celex = CELEX_BY_NUMBER.get(number)
            if target_celex is None:
                results.append({**rel, "_resolution": "skip"})
            elif target_celex == source_celex:
                # Self-reference — skip (already handled as internal CITES)
                continue
            else:
                results.append({**rel, "_resolution": "resolvable",
                                "_target_celex": target_celex})
    return results


# ---------------------------------------------------------------------------
# 2 & 3. Parse ref_text → provision-level target ID
# ---------------------------------------------------------------------------

_PROVISION_RE = re.compile(
    r"""
    (?:
      # "point (N) of Article M" (reversed order, e.g. definitions)
      point\s+\((?P<rev_point>[a-z0-9]+)\)
      (?:\((?P<rev_subpoint>[a-z0-9]+)\))?
      \s+of\s+Articles?\s+(?P<rev_article>\d+)
    |
      # "Article N(P), point (X)(Y)"
      Articles?\s+(?P<article>\d+)
      (?:\((?P<para>\d+)\))?
      (?:,?\s*point\s+\((?P<point>[a-z0-9]+)\)(?:\((?P<subpoint>[a-z0-9]+)\))?)?
    |
      Annex\s+(?P<annex>[IVX]+|\d+)
      (?:,?\s*Section\s+(?P<section>[A-Z]|\d+))?
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_ref_text(ref_text: str) -> dict[str, str]:
    """Extract article/paragraph/point/annex from a ref_text string."""
    m = _PROVISION_RE.search(ref_text)
    if not m:
        return {}
    groups = {k: v for k, v in m.groupdict().items() if v is not None}
    # Normalize reversed form: "point (N) of Article M"
    if "rev_article" in groups:
        groups["article"] = groups.pop("rev_article")
        if "rev_point" in groups:
            groups["point"] = groups.pop("rev_point")
        if "rev_subpoint" in groups:
            groups["subpoint"] = groups.pop("rev_subpoint")
    return groups


def build_target_id(celex: str, parts: dict[str, str]) -> str | None:
    """Construct a deterministic provision ID from parsed ref_text components.

    Returns None when no provision-level info was extracted (document-level).

    ID format mirrors the HTML-based IDs from the EUR-Lex parser:
      - Article only:           {celex}_art_{N}
      - Article + paragraph:    {celex}_{art:03d}.{para:03d}
      - Article + para + point: {celex}_{art:03d}.{para:03d}_pt_{letter}
      - Article + point (no para, e.g. definitions): {celex}_art_{N}_pt_{point}
      - Annex:                  {celex}_anx_{roman}
      - Annex + section:        {celex}_anx_{roman}_sec_{section}
    """
    if not parts:
        return None

    annex = parts.get("annex")
    if annex:
        base = f"{celex}_anx_{annex}"
        section = parts.get("section")
        if section:
            return f"{base}_sec_{section}"
        return base

    article = parts.get("article")
    if article:
        para = parts.get("para")
        point = parts.get("point")
        subpoint = parts.get("subpoint")

        if para:
            # Paragraph-based IDs use zero-padded 3-digit format: 001.002
            base = f"{celex}_{int(article):03d}.{int(para):03d}"
            if point:
                if subpoint:
                    return f"{base}_pt_{point}_rm_{subpoint}"
                return f"{base}_pt_{point}"
            return base

        # No paragraph — article-level or direct article-point
        if point:
            if subpoint:
                return f"{celex}_art_{article}_pt_{point}_rm_{subpoint}"
            return f"{celex}_art_{article}_pt_{point}"
        return f"{celex}_art_{article}"

    return None


def build_alternative_ids(celex: str, parts: dict[str, str]) -> list[str]:
    """Generate alternative provision IDs for ambiguous references.

    Definitions articles (e.g. Article 2) use point-based IDs
    (``art_2_pt_N``) rather than paragraph-based IDs (``002.00N``).
    When ``build_target_id`` produces a paragraph-format ID,
    this function returns the point-format alternative so the
    crosslinker can try both before falling back to document-level.
    """
    alternatives: list[str] = []
    article = parts.get("article")
    para = parts.get("para")
    point = parts.get("point")
    subpoint = parts.get("subpoint")

    if article and para and not point:
        # "Article N(P)" could be point P of Article N
        alternatives.append(f"{celex}_art_{article}_pt_{para}")
    elif article and para and point:
        # "Article N(P), point (X)" — point X under paragraph-as-point P
        alternatives.append(f"{celex}_art_{article}_pt_{para}")

    return alternatives


# ---------------------------------------------------------------------------
# 4. Verify targets exist in Neo4j
# ---------------------------------------------------------------------------

def verify_targets(tx, target_ids: set[str]) -> set[str]:
    """Return the subset of *target_ids* that exist as Provision or Guidance nodes."""
    if not target_ids:
        return set()
    result = tx.run(
        "UNWIND $ids AS id "
        "OPTIONAL MATCH (p:Provision {id: id}) "
        "OPTIONAL MATCH (g:Guidance  {id: id}) "
        "WITH id, coalesce(p, g) AS n "
        "WHERE n IS NOT NULL "
        "RETURN id",
        ids=list(target_ids),
    )
    return {r["id"] for r in result}


# ---------------------------------------------------------------------------
# 5. Write CITES edges (idempotent via MERGE, never creates nodes)
# ---------------------------------------------------------------------------

def write_edges(tx, edges: list[dict[str, Any]]) -> int:
    """MERGE CITES edges between existing nodes.

    Source may be either a ``:Provision`` or ``:Guidance`` node (MDCG
    guidance citing regulations).  Targets are always ``:Provision``.
    """
    if not edges:
        return 0
    result = tx.run(
        "UNWIND $batch AS e "
        "OPTIONAL MATCH (s1:Provision {id: e.source}) "
        "OPTIONAL MATCH (s2:Guidance  {id: e.source}) "
        "WITH e, coalesce(s1, s2) AS s "
        "WHERE s IS NOT NULL "
        "MATCH (t:Provision {id: e.target}) "
        "MERGE (s)-[r:CITES]->(t) "
        "  ON CREATE SET r.resolved_from = 'crosslinker', "
        "                r.ref_text = e.ref_text "
        "RETURN count(r) AS c",
        batch=edges,
    )
    return result.single()["c"]


# ---------------------------------------------------------------------------
# 6. Cleanup stale ExternalAct nodes for loaded regulations
# ---------------------------------------------------------------------------

def cleanup_external_acts(session) -> dict[str, int]:
    """Delete ExternalAct nodes whose IDs correspond to loaded regulations,
    along with their CITES_EXTERNAL edges. Also remove any orphaned
    CITES_EXTERNAL edges pointing to loaded Provision nodes (leftovers from
    previous runs). Returns counts."""

    # Build list of ExternalAct ID fragments that match loaded regulations
    # e.g. ext_regulation_eu_2017_745 for MDR
    id_fragments = []
    for number in CELEX_BY_NUMBER:
        # "2017/745" → "2017_745"
        frag = number.replace("/", "_")
        id_fragments.append(frag)

    # Delete ExternalAct nodes matching loaded regulations
    result = session.run(
        "MATCH (e:ExternalAct) "
        "WHERE any(frag IN $frags WHERE e.id CONTAINS frag) "
        "DETACH DELETE e "
        "RETURN count(e) AS deleted",
        frags=id_fragments,
    )
    deleted_nodes = result.single()["deleted"]

    # Also remove stale CITES_EXTERNAL edges between Provision nodes
    # (leftovers from any previous loader run)
    result = session.run(
        "MATCH (:Provision)-[r:CITES_EXTERNAL]->(:Provision) "
        "DELETE r RETURN count(r) AS deleted"
    )
    deleted_edges = result.single()["deleted"]

    # Remove CITES_REGULATION edges (legacy type, replaced by CITES)
    result = session.run(
        "MATCH ()-[r:CITES_REGULATION]->() "
        "DELETE r RETURN count(r) AS deleted"
    )
    deleted_reg_edges = result.single()["deleted"]

    return {
        "external_act_nodes_deleted": deleted_nodes,
        "stale_cites_external_edges": deleted_edges,
        "legacy_cites_regulation_edges": deleted_reg_edges,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def crosslink(dry_run: bool = False, cleanup: bool = False) -> dict[str, int]:
    """Main entry point. Returns summary counts."""
    load_dotenv()

    refs = discover_resolvable_refs()
    skipped = sum(1 for r in refs if r["_resolution"] == "skip")
    resolvable = [r for r in refs if r["_resolution"] == "resolvable"]

    # Build candidate edges — all are CITES (provision or document-level)
    candidate_edges: list[dict[str, Any]] = []
    provision_targets: set[str] = set()
    document_targets: set[str] = set()

    for rel in resolvable:
        source = rel["source"]
        target_celex = rel["_target_celex"]
        ref_text = rel.get("properties", {}).get("ref_text", "")

        parts = parse_ref_text(ref_text)
        target_id = build_target_id(target_celex, parts)

        if target_id:
            provision_targets.add(target_id)
            alts = build_alternative_ids(target_celex, parts)
            for alt in alts:
                provision_targets.add(alt)
            doc_fallback = f"{target_celex}_document"
            document_targets.add(doc_fallback)
            candidate_edges.append({
                "source": source,
                "target": target_id,
                "alternatives": alts,
                "fallback": doc_fallback,
                "ref_text": ref_text,
            })
        else:
            doc_target = f"{target_celex}_document"
            document_targets.add(doc_target)
            candidate_edges.append({
                "source": source,
                "target": doc_target,
                "fallback": None,
                "ref_text": ref_text,
            })

    # Connect to Neo4j, verify targets, write edges
    uri = _normalize_neo4j_uri(os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    user = os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j"))
    password = os.environ.get("NEO4J_PASSWORD", "password")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    provision_matched = 0
    document_fallback = 0

    try:
        with driver.session(database=database) as session:
            # Reset previous crosslinker output so re-runs are idempotent
            if not dry_run:
                reset = session.run(
                    "MATCH ()-[r:CITES {resolved_from: 'crosslinker'}]->() "
                    "DELETE r RETURN count(r) AS c"
                )
                reset_count = reset.single()["c"]
                if reset_count:
                    logger.info("Cleared %d stale crosslinker CITES edges.", reset_count)

            all_targets = provision_targets | document_targets
            existing = session.execute_read(verify_targets, all_targets)

            final_edges: list[dict[str, Any]] = []
            for edge in candidate_edges:
                if edge["target"] in existing:
                    provision_matched += 1
                    final_edges.append(edge)
                    continue
                # Try alternative IDs (e.g. point-format for definitions)
                alt_hit = None
                for alt in edge.get("alternatives", []):
                    if alt in existing:
                        alt_hit = alt
                        break
                if alt_hit:
                    provision_matched += 1
                    final_edges.append({**edge, "target": alt_hit})
                elif edge["fallback"] and edge["fallback"] in existing:
                    logger.debug(
                        "Target %s not found; falling back to %s",
                        edge["target"], edge["fallback"],
                    )
                    document_fallback += 1
                    final_edges.append({**edge, "target": edge["fallback"]})
                else:
                    logger.warning(
                        "Neither %s nor %s found in graph — skipping.",
                        edge["target"], edge.get("fallback", "—"),
                    )

            written = 0
            if not dry_run and final_edges:
                written = session.execute_write(write_edges, final_edges)

            # Cleanup stale ExternalAct nodes
            cleanup_stats = {"external_act_nodes_deleted": 0,
                             "stale_cites_external_edges": 0,
                             "legacy_cites_regulation_edges": 0}
            if cleanup and not dry_run:
                cleanup_stats = cleanup_external_acts(session)
    finally:
        driver.close()

    summary = {
        "provision_level": provision_matched,
        "document_level": document_fallback,
        "edges_written": written if not dry_run else 0,
        "skipped_not_in_catalog": skipped,
        "total_cites_external": len(refs),
        **cleanup_stats,
    }

    print("\n=== Crosslinker Summary ===")
    print(f"  {'CITES (provision-level):':<40} {provision_matched:>4}")
    print(f"  {'CITES (document-level fallback):':<40} {document_fallback:>4}")
    print(f"  {'Edges written:':<40} {summary['edges_written']:>4}")
    print(f"  {'Skipped (not in catalog):':<40} {skipped:>4}")
    print(f"  {'Total CITES_EXTERNAL processed:':<40} {len(refs):>4}")
    if cleanup and not dry_run:
        print(f"  {'ExternalAct nodes deleted:':<40} {cleanup_stats['external_act_nodes_deleted']:>4}")
        print(f"  {'Stale CITES_EXTERNAL edges deleted:':<40} {cleanup_stats['stale_cites_external_edges']:>4}")
        print(f"  {'Legacy CITES_REGULATION edges deleted:':<40} {cleanup_stats['legacy_cites_regulation_edges']:>4}")
    if dry_run:
        print("  (dry run — no changes written)")
    print()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Resolve CITES_EXTERNAL → CITES")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete stale ExternalAct nodes and legacy edge types")
    args = parser.parse_args()
    crosslink(dry_run=args.dry_run, cleanup=args.cleanup)
