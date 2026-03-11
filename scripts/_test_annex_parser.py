#!/usr/bin/env python3
"""Quick test of the new annex parser on MDR and AI Act."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.run_pipeline import run


def test_regulation(celex: str, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"Testing {label} ({celex})")
    print(f"{'='*60}")

    result = run(celex, "EN")
    if result is None:
        print("  FAILED: pipeline returned None")
        return

    with open(result, "r") as f:
        data = json.load(f)

    provisions = data.get("provisions", [])
    print(f"  Total provisions: {len(provisions)}")

    # Count by kind
    kinds: dict[str, int] = {}
    for p in provisions:
        k = p.get("kind", "unknown")
        kinds[k] = kinds.get(k, 0) + 1
    for k, v in sorted(kinds.items()):
        print(f"    {k}: {v}")

    # Show annex structure
    annexes = [p for p in provisions if p["kind"] == "annex"]
    print(f"\n  Annexes found: {len(annexes)}")
    for a in annexes:
        children = a.get("children", [])
        child_kinds = {}
        for cid in children:
            for p in provisions:
                if p["id"] == cid:
                    ck = p.get("kind", "?")
                    child_kinds[ck] = child_kinds.get(ck, 0) + 1
        print(f"    {a['id']}: {a.get('title', '?')[:50]}  children={len(children)} {dict(child_kinds)}")

    # Show sample nodes from Annex I (first 10 children)
    anx_i = next((p for p in provisions if p["id"].endswith("_anx_I")), None)
    if anx_i:
        print(f"\n  Sample: Annex I children (first 10):")
        for cid in anx_i.get("children", [])[:10]:
            child = next((p for p in provisions if p["id"] == cid), None)
            if child:
                t = child.get("text", "")[:60]
                print(f"    {child['id']}  [{child['kind']}] num={child.get('number')} \"{t}...\"")

    # Show Annex VII structure (or Annex III for AI Act)
    test_annex_id = f"{celex}_anx_VII" if celex == "32017R0745" else f"{celex}_anx_III"
    test_annex = next((p for p in provisions if p["id"] == test_annex_id), None)
    if test_annex:
        print(f"\n  Deep dive: {test_annex['id']}")
        _show_tree(provisions, test_annex["id"], depth=0, max_depth=3)


def _show_tree(provisions, node_id, depth, max_depth):
    if depth > max_depth:
        return
    node = next((p for p in provisions if p["id"] == node_id), None)
    if not node:
        return
    indent = "    " + "  " * depth
    t = node.get("text", "")[:50]
    print(f"{indent}{node['id'].split('_', 1)[-1] if '_' in node['id'] else node['id']}  [{node['kind']}] num={node.get('number')} \"{t}\"")
    for cid in node.get("children", [])[:8]:
        _show_tree(provisions, cid, depth + 1, max_depth)
    if len(node.get("children", [])) > 8:
        print(f"{indent}  ... +{len(node['children']) - 8} more")


if __name__ == "__main__":
    test_regulation("32017R0745", "MDR")
    test_regulation("32024R1689", "EU AI Act")
    print("\nDone.")
