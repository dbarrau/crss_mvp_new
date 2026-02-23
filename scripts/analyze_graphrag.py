#!/usr/bin/env python3
"""
Improved GraphRAG JSON Analyzer for EUR-Lex / EU regulations
Handles both dict-based and list-based relation formats
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Union

def analyze_graphrag_json(file_path: str):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    print(f"🔄 Loading {file_path.name} ... (this may take 10-60s for huge files)")

    try:
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # Basic top-level checks
    provisions: List[Dict] = data.get("provisions", [])
    relations: List[Union[Dict, List]] = data.get("relations", [])
    celex = data.get("celex_id", "UNKNOWN")
    graph_version = data.get("graph_version", "UNKNOWN")

    print(f"✅ Loaded: {len(provisions):,} provisions | {len(relations):,} relations")
    print(f"   CELEX: {celex} | Graph version: {graph_version}\n")

    # === 1. PROVISION ANALYSIS ===
    kind_counter = Counter()
    depth_counter = Counter()
    orphan_count = 0
    root_count = 0
    path_mismatch_count = 0
    null_text_count = 0
    has_obligations_count = 0
    has_applies_to_count = 0
    has_semantic_role_count = 0

    parent_to_children = defaultdict(list)
    provision_dict = {}  # id → provision

    for p in provisions:
        pid = p.get("id")
        if not pid:
            continue
        provision_dict[pid] = p

        kind = p.get("kind", "UNKNOWN")
        kind_counter[kind] += 1

        depth = p.get("hierarchy_depth")
        if isinstance(depth, (int, float)):
            depth_counter[int(depth)] += 1

        parent_id = p.get("parent_id")
        if parent_id:
            parent_to_children[parent_id].append(pid)
        else:
            root_count += 1

        if not parent_id and kind != "document":
            orphan_count += 1

        # Path consistency check
        path = p.get("path", [])
        if isinstance(path, list) and path and parent_id and path[-1] != parent_id:
            path_mismatch_count += 1

        if not p.get("text"):
            null_text_count += 1

        if p.get("obligations"):
            has_obligations_count += 1
        if p.get("applies_to"):
            has_applies_to_count += 1
        if p.get("semantic_role"):
            has_semantic_role_count += 1

    # === 2. RELATION ANALYSIS (robust against list vs dict) ===
    relation_type_counter = Counter()
    relation_formats = Counter()

    for r in relations:
        fmt = type(r).__name__
        relation_formats[fmt] += 1

        rel_type = None

        if isinstance(r, dict):
            rel_type = r.get("type")
        elif isinstance(r, list) and len(r) >= 3:
            # possible formats: [source, target, type] or [source, target, {"type": ...}]
            if isinstance(r[2], str):
                rel_type = r[2]
            elif isinstance(r[2], dict) and "type" in r[2]:
                rel_type = r[2]["type"]
        # add more formats here if needed

        if rel_type:
            relation_type_counter[rel_type] += 1

    # === 3. HIERARCHY & QUALITY SUMMARY ===
    max_depth = max(depth_counter.keys(), default=0)
    deepest_kinds_sample = [p["kind"] for p in provisions
                           if p.get("hierarchy_depth") == max_depth][:5]

    report = {
        "metadata": {
            "celex_id": celex,
            "graph_version": graph_version,
            "total_provisions": len(provisions),
            "total_relations": len(relations),
            "file_name": str(file_path)
        },
        "hierarchy_summary": {
            "root_nodes": root_count,
            "orphans": orphan_count,
            "max_hierarchy_depth": max_depth,
            "depth_distribution": dict(sorted(depth_counter.items())),
            "top_kinds": dict(kind_counter.most_common(15)),
            "annex_count": kind_counter.get("annex", 0),
            "recital_count": kind_counter.get("recital", 0),
            "article_count": kind_counter.get("article", 0),
            "point_count": kind_counter.get("point", 0),
            "roman_item_count": kind_counter.get("roman_item", 0),
            "deepest_kinds_sample": deepest_kinds_sample
        },
        "data_quality": {
            "provisions_with_null_text": null_text_count,
            "path_mismatches": path_mismatch_count,
            "provisions_with_obligations": has_obligations_count,
            "pct_with_obligations": round(100 * has_obligations_count / len(provisions), 1) if provisions else 0,
            "provisions_with_applies_to": has_applies_to_count,
            "provisions_with_semantic_role": has_semantic_role_count,
        },
        "relations": {
            "relation_formats_found": dict(relation_formats),
            "relation_types": dict(relation_type_counter.most_common(12))
        },
        "issues": []
    }

    # Auto-detect common problems
    if orphan_count > 5:
        report["issues"].append(f"⚠️ {orphan_count} orphan provisions (no parent)")
    if path_mismatch_count > 0:
        report["issues"].append(f"⚠️ {path_mismatch_count} path[-1] ≠ parent_id mismatches")
    if max_depth > 14:
        report["issues"].append(f"⚠️ Very deep hierarchy (max depth {max_depth}) — check for cycles/over-nesting")
    if report["hierarchy_summary"]["roman_item_count"] == 0 and report["hierarchy_summary"]["point_count"] > 30:
        report["issues"].append("⚠️ No roman_item nodes but many points — parser may miss (i), (ii), …")
    if "preamble" not in kind_counter or "enacting_terms" not in kind_counter:
        report["issues"].append("⚠️ Missing 'preamble' and/or 'enacting_terms' top-level containers")

    # === OUTPUT ===
    print("=" * 80)
    print("📊 EUR-LEX / GRAPHRAG ANALYSIS REPORT")
    print("=" * 80)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if report["issues"]:
        print("\n🚩 DETECTED ISSUES:")
        for issue in report["issues"]:
            print(f"  • {issue}")

    # Save compact report
    report_path = file_path.with_name(f"{file_path.stem}_analysis_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Report saved → {report_path}")
    print("Done.")

    # Optional: uncomment to see first few relations when debugging
    # print("\nDebug — first 5 relations:")
    # for i, r in enumerate(relations[:5]):
    #     print(f"  {i}: {type(r).__name__} → {r}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_graphrag.py /path/to/your/parsed.json")
        sys.exit(1)
    analyze_graphrag_json(sys.argv[1])
