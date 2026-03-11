#!/usr/bin/env python3
"""One-shot re-parse of both regulations with the updated annex parser."""
import sys, json, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.run_pipeline import run

results = {}
for celex in ["32017R0745", "32024R1689"]:
    try:
        out = run(celex, "EN")
        results[celex] = str(out) if out else "FAILED"
    except Exception as e:
        results[celex] = f"ERROR: {e}\n{traceback.format_exc()}"

# Write results to a known file
with open("/tmp/crss_reparse_results.txt", "w") as f:
    for k, v in results.items():
        f.write(f"{k}: {v}\n")

# Also do a quick analysis of the output
for celex in ["32017R0745", "32024R1689"]:
    parsed = Path(__file__).resolve().parents[1] / "data" / "regulations" / celex / "EN" / "parsed.json"
    if parsed.exists():
        with open(parsed) as fp:
            data = json.load(fp)
        kinds = {}
        for p in data["provisions"]:
            k = p["kind"]
            kinds[k] = kinds.get(k, 0) + 1
        with open("/tmp/crss_reparse_results.txt", "a") as f:
            f.write(f"\n{celex} provision counts:\n")
            for k in sorted(kinds):
                f.write(f"  {k}: {kinds[k]}\n")

            # Check annex hierarchy depth
            annex_provs = [p for p in data["provisions"] if p["kind"].startswith("annex")]
            max_depth = max((p.get("hierarchy_depth", 0) for p in annex_provs), default=0)
            f.write(f"  max annex hierarchy_depth: {max_depth}\n")

            # Show some sample annex parent relationships
            by_id = {p["id"]: p for p in data["provisions"]}
            sample = [p for p in data["provisions"] if p["kind"] == "annex_chapter"][:3]
            for s in sample:
                parent = by_id.get(s.get("parent_id", ""), {})
                f.write(f"  chapter: {s['id']} (parent: {parent.get('kind', '?')} {parent.get('id', '?')})\n")

            sample_sec = [p for p in data["provisions"] if p["kind"] == "annex_section"][:5]
            for s in sample_sec:
                parent = by_id.get(s.get("parent_id", ""), {})
                f.write(f"  section: {s['id']} num={s.get('number')} (parent: {parent.get('kind', '?')} {parent.get('id', '?')})\n")

with open("/tmp/crss_reparse_results.txt", "a") as f:
    f.write("\nDONE\n")
print("Script complete - results in /tmp/crss_reparse_results.txt")
