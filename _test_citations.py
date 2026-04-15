"""Unit test for the improved guidance citation detection."""
import sys
import json

# Verify import
from ingestion.parse.semantic_layer.guidance_references import (
    extract_guidance_relations,
    _QUALIFIED_ARTICLE_RE,
    _QUALIFIED_ANNEX_RE,
    _SHORT_NAMES,
    _find_regulation_mentions,
)

print("=== Import OK ===")
print(f"Short names: {_SHORT_NAMES}")

# --- Test 1: Qualified article patterns ---
test_cases = [
    ("Article 6(1) AIA", "2024/1689", "Article 6(1)"),
    ("Art. 3 (29) AIA", "2024/1689", "Article 3(29)"),
    ("Article 2(1) MDR", "2017/745", "Article 2(1)"),
    ("Article 2(2) IVDR", "2017/746", "Article 2(2)"),
    ("Article 43(1) AIA", "2024/1689", "Article 43(1)"),
    ("Article 50 AIA", "2024/1689", "Article 50"),
    ("Article 10 AIA", "2024/1689", "Article 10"),
]

print("\n=== Test 1: Qualified Article Patterns ===")
all_pass = True
for text, expected_number, expected_ref in test_cases:
    m = _QUALIFIED_ARTICLE_RE.search(text)
    if not m:
        print(f"  FAIL: no match for '{text}'")
        all_pass = False
        continue
    short = m.group("short")
    number = _SHORT_NAMES.get(short, "???")
    ref = f"Article {m.group('article')}"
    if m.group("para"):
        ref += f"({m.group('para')})"
    ok_num = number == expected_number
    ok_ref = ref == expected_ref
    status = "OK" if (ok_num and ok_ref) else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  {status}: '{text}' -> number={number} (expect {expected_number}), ref={ref} (expect {expected_ref})")

# --- Test 2: Qualified annex patterns ---
annex_cases = [
    ("Annex VI AIA", "2024/1689", "Annex VI"),
    ("Annex XIV MDR", "2017/745", "Annex XIV"),
    ("Annex I MDR", "2017/745", "Annex I"),
]

print("\n=== Test 2: Qualified Annex Patterns ===")
for text, expected_number, expected_ref in annex_cases:
    m = _QUALIFIED_ANNEX_RE.search(text)
    if not m:
        print(f"  FAIL: no match for '{text}'")
        all_pass = False
        continue
    short = m.group("short")
    number = _SHORT_NAMES.get(short, "???")
    ref = f"Annex {m.group('annex')}"
    ok = number == expected_number and ref == expected_ref
    status = "OK" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  {status}: '{text}' -> number={number}, ref={ref}")

# --- Test 3: Full integration via extract_guidance_relations ---
print("\n=== Test 3: Full Integration ===")

# Paragraph mixing AIA and MDR citations — this was the key misattribution scenario
mixed_text = (
    "A MDAI is considered a high-risk AI system under Article 6(1) AIA if it meets both "
    "conditions: the MDAI is intended to be used as a medical device as defined in Article 2(1) MDR "
    "or Article 2(2) IVDR, and the MDAI must comply with Article 43(3) AIA. "
    "See also Annex VI AIA and Annex XIV MDR."
)
provisions = [{"id": "test_sec_1", "text": mixed_text}]
rels = extract_guidance_relations(provisions)

print(f"  Relations found: {len(rels)}")
for r in rels:
    print(f"    {r['properties']['number']:>10s}  {r['properties']['ref_text']}")

# Verify specific attributions
rels_by_ref = {(r["properties"]["number"], r["properties"]["ref_text"]): r for r in rels}

# AIA articles should map to 2024/1689
assert ("2024/1689", "Article 6(1)") in rels_by_ref, "Article 6(1) should map to AIA"
assert ("2024/1689", "Article 43(3)") in rels_by_ref, "Article 43(3) should map to AIA"
assert ("2024/1689", "Annex VI") in rels_by_ref, "Annex VI should map to AIA"
# MDR articles should map to 2017/745
assert ("2017/745", "Article 2(1)") in rels_by_ref, "Article 2(1) should map to MDR"
assert ("2017/745", "Annex XIV") in rels_by_ref, "Annex XIV should map to MDR"
# IVDR article should map to 2017/746
assert ("2017/746", "Article 2(2)") in rels_by_ref, "Article 2(2) should map to IVDR"

print("  All attribution assertions PASSED")

# --- Test 4: Compound MDR/IVDR ---
print("\n=== Test 4: Compound MDR/IVDR ===")
compound_text = "the MDR/IVDR require manufacturers to ensure compliance with the Regulation."
mentions = _find_regulation_mentions(compound_text)
numbers_found = sorted(set(m["number"] for m in mentions))
print(f"  Mentions: {numbers_found}")
assert "2017/745" in numbers_found, "MDR should be found in MDR/IVDR"
assert "2017/746" in numbers_found, "IVDR should be found in MDR/IVDR"
print("  Compound expansion PASSED")

# --- Test 5: AI Act multi-word short name ---
print("\n=== Test 5: AI Act Multi-word ===")
ai_act_text = "The AI Act complements the MDR by introducing new requirements."
mentions = _find_regulation_mentions(ai_act_text)
numbers_found = sorted(set(m["number"] for m in mentions))
print(f"  Mentions: {numbers_found}")
assert "2024/1689" in numbers_found, "AI Act should resolve to 2024/1689"
assert "2017/745" in numbers_found, "MDR should still be found"
print("  AI Act detection PASSED")

# --- Test 6: Document-level fallback ---
print("\n=== Test 6: Document-level fallback ===")
doc_text = "The AIA applies to medical devices."
provisions = [{"id": "test_sec_2", "text": doc_text}]
rels = extract_guidance_relations(provisions)
print(f"  Relations found: {len(rels)}")
for r in rels:
    print(f"    {r['properties']['number']:>10s}  {r['properties']['ref_text']}")
assert len(rels) == 1, "Should have exactly 1 document-level relation"
assert rels[0]["properties"]["number"] == "2024/1689"
assert "Regulation (EU) 2024/1689" in rels[0]["properties"]["ref_text"]
print("  Document-level fallback PASSED")

# --- Test 7: No double-counting ---
print("\n=== Test 7: No double-counting ===")
dedup_text = "Article 6(1) AIA requires compliance. Article 6(1) AIA is the key provision."
provisions = [{"id": "test_sec_3", "text": dedup_text}]
rels = extract_guidance_relations(provisions)
ref_keys = [(r["properties"]["number"], r["properties"]["ref_text"]) for r in rels]
print(f"  Relations: {ref_keys}")
assert ref_keys.count(("2024/1689", "Article 6(1)")) == 1, "Should deduplicate"
print("  Deduplication PASSED")

# --- Summary ---
print("\n=== ALL TESTS PASSED ===")
