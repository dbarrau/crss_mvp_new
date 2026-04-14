"""Diagnostic: check Article 61 cross-refs and Annex XIV in parsed data."""
import json

with open("data/legislation/32017R0745/EN/parsed.json") as f:
    data = json.load(f)

# Article 61
for p in data["provisions"]:
    if p["id"] == "32017R0745_art_61":
        print("=== Article 61 ===")
        print("display_ref:", p.get("display_ref"))
        print("text[:300]:", (p.get("text") or "")[:300])
        print("internal_refs:", p.get("internal_refs", [])[:8])
        print("external_refs:", p.get("external_refs", [])[:8])
        break

# Annex XIV provisions
annex_xiv = [p for p in data["provisions"]
             if "xiv" in (p.get("display_ref") or "").lower()]
print(f"\n=== Annex XIV provisions: {len(annex_xiv)} ===")
for p in annex_xiv[:8]:
    print(f"  {p['id']}: {p.get('display_ref','')} | kind={p.get('kind','')} | text[:100]={(p.get('text') or '')[:100]}")

# CITES from Article 61 subtree
cites_from_61 = [r for r in data.get("relations", [])
                 if r.get("type") == "CITES" and "art_61" in r.get("source", "")]
print(f"\n=== CITES from art_61 subtree: {len(cites_from_61)} ===")
for c in cites_from_61:
    print(f"  {c['source']} -> {c['target']}")

# CITES targeting Annex XIV
annex_xiv_ids = {p["id"] for p in annex_xiv}
cites_to_xiv = [r for r in data.get("relations", [])
                if r.get("type") == "CITES" and r.get("target", "") in annex_xiv_ids]
print(f"\n=== CITES targeting Annex XIV: {len(cites_to_xiv)} ===")
for c in cites_to_xiv[:10]:
    print(f"  {c['source']} -> {c['target']}")

# Check if "equivalent" appears anywhere in Annex XIV
print("\n=== 'equivalent' in Annex XIV text ===")
for p in annex_xiv:
    text = (p.get("text") or "").lower()
    if "equivalent" in text or "equivalence" in text:
        print(f"  {p['id']}: ...{text[max(0,text.find('equiv')-30):text.find('equiv')+50]}...")
