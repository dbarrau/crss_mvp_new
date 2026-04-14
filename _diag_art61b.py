"""Diagnostic: check annex coverage, Article 61 children, and 'equivalent' hits."""
import json, re
from collections import Counter

with open("data/legislation/32017R0745/EN/parsed.json") as f:
    data = json.load(f)

# 1. Annex coverage
annexes = [p for p in data["provisions"] if (p.get("kind") or "").startswith("annex")]
print(f"Total annex provisions: {len(annexes)}")
top_annex = Counter()
for p in annexes:
    ref = p.get("display_ref", "") or p["id"]
    m = re.match(r"(Annex\s+[IVX]+)", ref)
    top_annex[m.group(1) if m else f"other:{ref[:40]}"] += 1
for k, v in sorted(top_annex.items()):
    print(f"  {k}: {v} provisions")

# 2. Article 61 paragraphs
art61_kids = [p for p in data["provisions"] if p["id"].startswith("32017R0745_061.")]
print(f"\nArticle 61 paragraphs: {len(art61_kids)}")
for p in art61_kids[:5]:
    text = (p.get("text") or "")[:150]
    refs = p.get("internal_refs", [])
    print(f"  {p['id']} ({p.get('display_ref','')}): {text}")
    if refs:
        print(f"    internal_refs: {refs[:5]}")

# 3. CITES edges from 061.* nodes
cites_61 = [r for r in data.get("relations", [])
            if r.get("type") == "CITES" and r.get("source", "").startswith("32017R0745_061.")]
print(f"\nCITES from Article 61 paragraphs: {len(cites_61)}")
for c in cites_61[:10]:
    print(f"  {c['source']} -> {c['target']}")

# 4. Provisions mentioning 'equivalent'
print("\n=== Provisions containing 'equivalent'/'equivalence' ===")
count = 0
for p in data["provisions"]:
    text = (p.get("text") or "").lower()
    if "equivalen" in text:
        count += 1
        idx = text.find("equivalen")
        snippet = text[max(0, idx-40):idx+60]
        if count <= 10:
            print(f"  {p['id']} ({p.get('display_ref','')}): ...{snippet}...")
print(f"Total provisions with 'equivalent': {count}")
