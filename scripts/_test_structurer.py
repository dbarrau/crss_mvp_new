#!/usr/bin/env python3
"""Quick test of the MDCG structurer on both documents."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.parse.guidance.mdcg_structurer import structure_mdcg

docs = [
    ("data/guidance/MDCG_2020_3/EN/mdcg_2020_3_rev1_clean.md", "MDCG_2020_3", "MDCG 2020-3 Rev.1"),
    ("data/guidance/MDCG_2019_11/EN/mdcg_2019_11_en_clean.md", "MDCG_2019_11", "MDCG 2019-11 Rev.1"),
]

for md_path, doc_id, doc_name in docs:
    print(f"\n{'='*80}")
    print(f"  {doc_name} ({doc_id})")
    print(f"{'='*80}")

    result = structure_mdcg(md_path, doc_id, doc_name, "EN")
    provs = result["provisions"]
    print(f"  Total provisions: {len(provs)}")
    print()

    for p in provs:
        indent = "  " * p["hierarchy_depth"]
        num = p.get("number") or "-"
        kind = p["kind"]
        pid = p["id"]
        dref = (p.get("display_ref") or "")[:60]
        nc = len(p["children"])
        tlen = len(p.get("text") or "")
        print(f"  {indent}{kind:25s}  ch={nc:2d}  num={num:8s}  text={tlen:5d}ch  {dref}")

    # Schema check
    required = {"id", "kind", "text", "hierarchy_depth", "path", "parent_id", "children", "lang"}
    ok = True
    for p in provs:
        missing = required - set(p.keys())
        if missing:
            print(f"  SCHEMA ERROR: {p['id']} missing {missing}")
            ok = False
            break
    if ok:
        print(f"\n  Schema check: OK")

    # text_for_analysis stats
    enriched = sum(1 for p in provs if p.get("text_for_analysis"))
    print(f"  text_for_analysis: {enriched} / {len(provs)}")
