#!/usr/bin/env python3
"""Verify annex hierarchy improvements."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

with open("data/regulations/32017R0745/EN/parsed.json") as f:
    data = json.load(f)

provisions = data["provisions"]
by_id = {p["id"]: p for p in provisions}

# Check Annex VII point 1.1.6 - previously collapsed into sec_3 text blob
p = by_id.get("32017R0745_anx_VII_1.1.6")
if p:
    print("=== Annex VII 1.1.6 ===")
    print(f"  kind: {p['kind']}")
    print(f"  number: {p.get('number')}")
    print(f"  parent: {p['parent_id']}")
    print(f"  text: {p['text'][:80]}...")
    print(f"  children ({len(p['children'])}):")
    for cid in p["children"]:
        c = by_id.get(cid, {})
        print(f"    {c['id']}: [{c['kind']}] {c['text'][:60]}")
else:
    print("ERROR: anx_VII_1.1.6 not found!")

# Check Annex I Chapter structure
print("\n=== Annex I structure ===")
anx_i = by_id.get("32017R0745_anx_I")
if anx_i:
    for cid in anx_i["children"]:
        c = by_id.get(cid, {})
        print(f"  {c['id']}: [{c['kind']}] num={c.get('number')} \"{c['text'][:50]}\"")
        for ccid in c.get("children", [])[:3]:
            cc = by_id.get(ccid, {})
            print(f"    {cc['id']}: [{cc['kind']}] num={cc.get('number')}")
        if len(c.get("children", [])) > 3:
            print(f"    ... +{len(c['children'])-3} more")

# Check Annex VIII classification rules
print("\n=== Annex VIII structure ===")
anx_viii = by_id.get("32017R0745_anx_VIII")
if anx_viii:
    for cid in anx_viii["children"]:
        c = by_id.get(cid, {})
        print(f"  {c['id']}: [{c['kind']}] num={c.get('number')} \"{c.get('title','')[:50]}\"")
        for ccid in c.get("children", [])[:5]:
            cc = by_id.get(ccid, {})
            print(f"    {cc['id']}: [{cc['kind']}] num={cc.get('number')} \"{cc['text'][:40]}\"")
        if len(c.get("children", [])) > 5:
            print(f"    ... +{len(c['children'])-5} more")

# Count unique kinds in annex provisions
annex_kinds = {}
for p in provisions:
    if p.get("kind", "").startswith("annex"):
        k = p["kind"]
        annex_kinds[k] = annex_kinds.get(k, 0) + 1
print("\n=== Annex node kinds ===")
for k, v in sorted(annex_kinds.items()):
    print(f"  {k}: {v}")
