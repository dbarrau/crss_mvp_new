#!/usr/bin/env python3
"""Validate hierarchical integrity of the re-parsed annex structure."""
import json, sys
from pathlib import Path

results = []

for celex in ["32017R0745", "32024R1689"]:
    parsed = Path(__file__).resolve().parents[1] / "data" / "legislation" / celex / "EN" / "parsed.json"
    with open(parsed) as f:
        data = json.load(f)

    by_id = {p["id"]: p for p in data["provisions"]}
    errors = []

    for p in data["provisions"]:
        pid = p.get("parent_id")
        # Check parent exists
        if pid and pid not in by_id:
            errors.append(f"  ORPHAN: {p['id']} references missing parent {pid}")
        # Check child references
        for cid in p.get("children", []):
            if cid not in by_id:
                errors.append(f"  MISSING_CHILD: {p['id']} references missing child {cid}")
            else:
                child = by_id[cid]
                if child.get("parent_id") != p["id"]:
                    errors.append(f"  PARENT_MISMATCH: child {cid} parent_id={child.get('parent_id')} != {p['id']}")

    # Check for duplicate IDs
    ids = [p["id"] for p in data["provisions"]]
    dups = [x for x in ids if ids.count(x) > 1]
    if dups:
        errors.append(f"  DUPLICATE_IDS: {set(dups)}")

    # Validate annex hierarchy: chapter under annex, section under chapter/annex/section
    for p in data["provisions"]:
        if p["kind"] == "annex_chapter":
            parent = by_id.get(p.get("parent_id", ""), {})
            if parent.get("kind") != "annex":
                errors.append(f"  BAD_PARENT: chapter {p['id']} under {parent.get('kind')} {parent.get('id')}")

    # Print specific annex I hierarchy for MDR
    if celex == "32017R0745":
        results.append(f"\n=== {celex} Annex I Hierarchy ===")
        anx_i = by_id.get("32017R0745_anx_I")
        if anx_i:
            for cid in anx_i.get("children", [])[:15]:
                child = by_id.get(cid, {})
                kind = child.get("kind", "?")
                num = child.get("number", "")
                title = (child.get("title") or child.get("text", ""))[:60]
                results.append(f"  {kind} {num}: {title}")
                # Show grandchildren for chapter nodes
                if kind == "annex_chapter":
                    for gcid in child.get("children", [])[:6]:
                        gc = by_id.get(gcid, {})
                        gk = gc.get("kind", "?")
                        gn = gc.get("number", "")
                        gt = (gc.get("title") or gc.get("text", ""))[:60]
                        results.append(f"    {gk} {gn}: {gt}")
                        # Show great-grandchildren for section 10
                        if gn == "10":
                            for ggcid in gc.get("children", [])[:8]:
                                ggc = by_id.get(ggcid, {})
                                results.append(f"      {ggc.get('kind','?')} {ggc.get('number','')}: {(ggc.get('title') or ggc.get('text',''))[:50]}")

    results.append(f"\n{celex}: {len(errors)} errors")
    for e in errors[:20]:
        results.append(e)

with open("/tmp/crss_validate.txt", "w") as f:
    f.write("\n".join(results) + "\n")
print("Validation complete")
