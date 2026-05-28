#!/usr/bin/env python3
"""Audit authority metadata coverage across catalog, local artifacts, and Neo4j.

This is an operational audit for the binding-force rollout. It answers:

1. Which legislation and guidance documents are expected from the catalogs?
2. Which documents have local source artifacts and parsed.json files?
3. Which documents are currently loaded in Neo4j?
4. Which local or graph documents are missing ``binding_force`` or ``source_type``?
5. Which documents exist only in Neo4j or only on disk?

Usage::

    python scripts/audit_authority_coverage.py
    python scripts/audit_authority_coverage.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from domain.legislation_catalog import LEGISLATION
from domain.mdcg_catalog import MDCG_DOCUMENTS
from infrastructure.graphdb.neo4j.loader import RegulationGraphLoader

DATA_DIR = PROJECT_ROOT / "data"
LEGISLATION_DIR = DATA_DIR / "legislation"
GUIDANCE_DIR = DATA_DIR / "guidance"


def _local_doc_status(doc_id: str, doc_type: str) -> dict[str, Any]:
    """Return local artifact and metadata status for one document."""
    base_dir = LEGISLATION_DIR if doc_type == "legislation" else GUIDANCE_DIR
    doc_dir = base_dir / doc_id / "EN"
    parsed_path = doc_dir / "parsed.json"
    raw_dir = doc_dir / "raw"

    clean_md_files = sorted(doc_dir.glob("*_clean.md")) if doc_type == "guidance" else []
    raw_files = sorted(raw_dir.iterdir()) if raw_dir.exists() else []

    status: dict[str, Any] = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "doc_dir_exists": doc_dir.exists(),
        "parsed_exists": parsed_path.exists(),
        "raw_dir_exists": raw_dir.exists(),
        "raw_files": [path.name for path in raw_files],
        "clean_markdown_files": [path.name for path in clean_md_files],
        "parsed_total": 0,
        "parsed_missing_binding": None,
        "parsed_missing_source": None,
        "parsed_binding_values": [],
        "parsed_source_values": [],
    }

    if not parsed_path.exists():
        return status

    data = json.loads(parsed_path.read_text(encoding="utf-8"))
    provisions = data.get("provisions", [])
    status["parsed_total"] = len(provisions)
    status["parsed_missing_binding"] = sum(
        1 for provision in provisions if "binding_force" not in provision
    )
    status["parsed_missing_source"] = sum(
        1 for provision in provisions if "source_type" not in provision
    )
    status["parsed_binding_values"] = sorted(
        {provision.get("binding_force") for provision in provisions}
    )
    status["parsed_source_values"] = sorted(
        {provision.get("source_type") for provision in provisions}
    )
    return status


def _graph_status() -> dict[str, dict[str, Any]]:
    """Return Neo4j authority metadata coverage keyed by CELEX/doc_id."""
    query = """
    CALL {
      MATCH (n:Provision)
      RETURN n.celex AS celex,
             'legislation' AS doc_type,
             count(*) AS total,
             sum(CASE WHEN n.binding_force IS NULL THEN 1 ELSE 0 END) AS missing_binding,
             sum(CASE WHEN n.source_type IS NULL THEN 1 ELSE 0 END) AS missing_source,
             collect(DISTINCT n.binding_force) AS binding_values,
             collect(DISTINCT n.source_type) AS source_values
      UNION ALL
      MATCH (n:Guidance)
      RETURN n.celex AS celex,
             'guidance' AS doc_type,
             count(*) AS total,
             sum(CASE WHEN n.binding_force IS NULL THEN 1 ELSE 0 END) AS missing_binding,
             sum(CASE WHEN n.source_type IS NULL THEN 1 ELSE 0 END) AS missing_source,
             collect(DISTINCT n.binding_force) AS binding_values,
             collect(DISTINCT n.source_type) AS source_values
    }
    RETURN celex, doc_type, total, missing_binding, missing_source, binding_values, source_values
    ORDER BY doc_type, celex
    """

    with RegulationGraphLoader.from_env() as loader:
        with loader._driver.session(database=loader._database) as session:  # noqa: SLF001
            rows = session.run(query).data()

    return {
        row["celex"]: {
            "doc_id": row["celex"],
            "doc_type": row["doc_type"],
            "graph_total": row["total"],
            "graph_missing_binding": row["missing_binding"],
            "graph_missing_source": row["missing_source"],
            "graph_binding_values": sorted(v for v in row["binding_values"] if v is not None),
            "graph_source_values": sorted(v for v in row["source_values"] if v is not None),
        }
        for row in rows
    }


def _catalog_doc_ids() -> dict[str, set[str]]:
    return {
        "legislation": set(LEGISLATION),
        "guidance": set(MDCG_DOCUMENTS),
    }


def build_report() -> dict[str, Any]:
    """Build the full authority coverage audit report."""
    catalog_ids = _catalog_doc_ids()
    graph_rows = _graph_status()

    local_rows: dict[str, dict[str, Any]] = {}
    for doc_id in sorted(catalog_ids["legislation"]):
        local_rows[doc_id] = _local_doc_status(doc_id, "legislation")
    for doc_id in sorted(catalog_ids["guidance"]):
        local_rows[doc_id] = _local_doc_status(doc_id, "guidance")

    all_doc_ids = sorted(set(local_rows) | set(graph_rows))
    documents: list[dict[str, Any]] = []
    for doc_id in all_doc_ids:
        local = local_rows.get(doc_id, {
            "doc_id": doc_id,
            "doc_type": graph_rows.get(doc_id, {}).get("doc_type", "unknown"),
            "doc_dir_exists": False,
            "parsed_exists": False,
            "raw_dir_exists": False,
            "raw_files": [],
            "clean_markdown_files": [],
            "parsed_total": 0,
            "parsed_missing_binding": None,
            "parsed_missing_source": None,
            "parsed_binding_values": [],
            "parsed_source_values": [],
        })
        graph = graph_rows.get(doc_id, {
            "doc_id": doc_id,
            "doc_type": local.get("doc_type", "unknown"),
            "graph_total": 0,
            "graph_missing_binding": None,
            "graph_missing_source": None,
            "graph_binding_values": [],
            "graph_source_values": [],
        })
        documents.append({**local, **graph})

    graph_only = sorted(doc_id for doc_id in graph_rows if doc_id not in local_rows)
    expected_missing_from_graph = sorted(
        doc_id for doc_id in local_rows if doc_id not in graph_rows
    )
    local_missing_parsed = sorted(
        doc["doc_id"] for doc in documents if doc["doc_dir_exists"] and not doc["parsed_exists"]
    )
    catalog_missing_local = sorted(
        doc["doc_id"] for doc in documents if doc["doc_id"] in local_rows and not doc["doc_dir_exists"]
    )
    parsed_missing_authority = sorted(
        doc["doc_id"]
        for doc in documents
        if doc["parsed_exists"] and (doc["parsed_missing_binding"] or doc["parsed_missing_source"])
    )
    graph_missing_authority = sorted(
        doc["doc_id"]
        for doc in documents
        if doc["graph_total"] and (doc["graph_missing_binding"] or doc["graph_missing_source"])
    )

    return {
        "summary": {
            "catalog_legislation": len(catalog_ids["legislation"]),
            "catalog_guidance": len(catalog_ids["guidance"]),
            "documents_audited": len(documents),
            "graph_only_documents": graph_only,
            "expected_documents_missing_from_graph": expected_missing_from_graph,
            "catalog_missing_local_directories": catalog_missing_local,
            "local_directories_missing_parsed_json": local_missing_parsed,
            "parsed_documents_missing_authority": parsed_missing_authority,
            "graph_documents_missing_authority": graph_missing_authority,
        },
        "documents": documents,
    }


def _print_human(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("=== Authority Coverage Audit ===")
    print(f"Catalog legislation docs: {summary['catalog_legislation']}")
    print(f"Catalog guidance docs:    {summary['catalog_guidance']}")
    print(f"Documents audited:        {summary['documents_audited']}")
    print()

    print("Graph-only documents:")
    print("  " + (", ".join(summary["graph_only_documents"]) if summary["graph_only_documents"] else "none"))
    print("Expected catalog documents missing from graph:")
    print("  " + (
        ", ".join(summary["expected_documents_missing_from_graph"])
        if summary["expected_documents_missing_from_graph"] else "none"
    ))
    print("Catalog docs missing local directories:")
    print("  " + (", ".join(summary["catalog_missing_local_directories"]) if summary["catalog_missing_local_directories"] else "none"))
    print("Local directories missing parsed.json:")
    print("  " + (", ".join(summary["local_directories_missing_parsed_json"]) if summary["local_directories_missing_parsed_json"] else "none"))
    print("Parsed docs missing authority metadata:")
    print("  " + (", ".join(summary["parsed_documents_missing_authority"]) if summary["parsed_documents_missing_authority"] else "none"))
    print("Graph docs missing authority metadata:")
    print("  " + (", ".join(summary["graph_documents_missing_authority"]) if summary["graph_documents_missing_authority"] else "none"))
    print()

    print("Per-document status:")
    for doc in report["documents"]:
        print(f"- {doc['doc_id']} ({doc['doc_type']})")
        print(
            "  local="
            f"dir:{'yes' if doc['doc_dir_exists'] else 'no'} "
            f"parsed:{'yes' if doc['parsed_exists'] else 'no'} "
            f"raw:{len(doc['raw_files'])} clean_md:{len(doc['clean_markdown_files'])}"
        )
        if doc["parsed_exists"]:
            print(
                "  parsed="
                f"total:{doc['parsed_total']} "
                f"missing_binding:{doc['parsed_missing_binding']} "
                f"missing_source:{doc['parsed_missing_source']} "
                f"binding:{doc['parsed_binding_values']} "
                f"source:{doc['parsed_source_values']}"
            )
        print(
            "  graph="
            f"total:{doc['graph_total']} "
            f"missing_binding:{doc['graph_missing_binding']} "
            f"missing_source:{doc['graph_missing_source']} "
            f"binding:{doc['graph_binding_values']} "
            f"source:{doc['graph_source_values']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit authority metadata coverage across CRSS layers")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    _print_human(report)


if __name__ == "__main__":
    main()
