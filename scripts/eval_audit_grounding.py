#!/usr/bin/env python3
"""Audit gap-fill grounding eval — does a definition *want* resolve to the RIGHT
provision, even when the auditor cites the wrong paragraph number?

The audit/revision loop asks the Auditor LLM for ``missing_provision_refs`` and
feeds them into retrieval. The auditor names each target by a paragraph number it
recalls from memory; for a *definition* that number is unreliable (observed:
"Article 3(40) AI Act (definition of 'safety component')" — safety component is
Article 3(14); 3(40) is 'biometric categorisation system'). If the gap-fill
trusts the number, it injects the WRONG definition while the needed one never
arrives. ``application/_audit.py`` therefore routes definition-wants through the
term channel (``find_by_term``) and drops the number.

This harness measures that guard across the WHOLE defined-term corpus, with no
LLM calls:

  For every ``:DefinedTerm`` (ground truth = its ``DEFINED_BY`` provision):
    1. synthesise a realistic auditor want that names the term but cites a WRONG
       number:  "Article 3(<wrong>) <reg> (definition of '<term>')";
    2. run the real gap-fill resolution (_extract_defined_term +
       _resolve_definition_want) on it;
    3. check it lands on the term's TRUE definition provision.

It reports term-resolution accuracy + the terms that fail to resolve (extraction
or find_by_term coverage gaps — actionable), and the number-based counterfactual
(how often trusting the cited number would misretrieve). A drop in accuracy is a
regression in the audit-grounding guard.

Usage::

    python scripts/eval_audit_grounding.py                 # all defined terms
    python scripts/eval_audit_grounding.py --limit 50      # first N (quick)
    python scripts/eval_audit_grounding.py --celex 32024R1689
    python scripts/eval_audit_grounding.py --out audit_grounding.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
from neo4j import GraphDatabase

from application._audit import _extract_defined_term, _resolve_definition_want
from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri
from retrieval.graph_retriever import GraphRetriever

# A deliberately-wrong point number: high enough that no real Article 3 point
# uses it, so any correct resolution must have come from the TERM, not the cite.
_WRONG_POINT = 997


def _load_ground_truth(limit: int | None, celex: str | None) -> list[dict]:
    """Every DefinedTerm and the provision that defines it (the correct target)."""
    load_dotenv(_PROJECT_ROOT / ".env")
    uri = _normalize_neo4j_uri(os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    user = os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j"))
    pw = os.environ.get("NEO4J_PASSWORD", "password")
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    cypher = (
        "MATCH (d:DefinedTerm)-[:DEFINED_BY]->(p:Provision) "
        + ("WHERE d.celex = $celex " if celex else "")
        + "RETURN d.term AS term, d.celex AS celex, d.regulation AS regulation, "
        "       p.id AS def_id, p.display_ref AS def_ref "
        "ORDER BY d.celex, p.id"
    )
    try:
        with driver.session(database=db) as s:
            rows = s.run(cypher, celex=celex).data()
    finally:
        driver.close()
    return rows[:limit] if limit else rows


def _reg_alias(celex: str, regulation: str | None) -> str:
    return {"32024R1689": "AI Act", "32017R0745": "MDR", "32017R0746": "IVDR",
            "32016R0679": "GDPR"}.get(celex, regulation or celex)


def run(limit: int | None, celex: str | None, out: str | None) -> dict:
    truth = _load_ground_truth(limit, celex)
    if not truth:
        print("No DefinedTerm→provision pairs found (is Neo4j loaded?).")
        return {}

    retriever = GraphRetriever()
    try:
        results = []
        for row in truth:
            term, def_id = row["term"], row["def_id"]
            reg = _reg_alias(row["celex"], row.get("regulation"))
            # Two realistic auditor phrasings, both citing a WRONG number — one
            # quoted (exercises _QUOTED_TERM_RE), one unquoted (exercises
            # _DEFINITION_OF_RE). Both must resolve to the term by TERM.
            wants = [
                f"Article 3({_WRONG_POINT}) {reg} (definition of '{term}')",
                f"Article 3({_WRONG_POINT}) {reg} (definition of {term})",
            ]
            variants = []
            for want in wants:
                extracted = _extract_defined_term(want)
                resolved = (
                    _resolve_definition_want(retriever, extracted, {row["celex"]})
                    if extracted else []
                )
                resolved_id = resolved[0].get("article_id") if resolved else None
                variants.append({
                    "extracted_term": extracted,
                    "resolved_id": resolved_id,
                    "extraction_ok": bool(extracted),
                    "resolution_ok": resolved_id == def_id,
                })

            results.append({
                "term": term,
                "celex": row["celex"],
                "true_def_id": def_id,
                # a case passes only if EVERY phrasing resolves correctly
                "extraction_ok": all(v["extraction_ok"] for v in variants),
                "resolution_ok": all(v["resolution_ok"] for v in variants),
                "resolved_id": variants[0]["resolved_id"],
                "variants": variants,
            })
    finally:
        retriever.close()

    n = len(results)
    extr_ok = sum(r["extraction_ok"] for r in results)
    res_ok = sum(r["resolution_ok"] for r in results)
    failures = [r for r in results if not r["resolution_ok"]]

    print("\n=== Audit gap-fill grounding ===")
    print(f"  DefinedTerms tested            : {n}")
    print(f"  Term extracted from the want   : {extr_ok}/{n}  ({extr_ok/n:.1%})")
    print(f"  Resolved to the TRUE provision : {res_ok}/{n}  ({res_ok/n:.1%})")
    print(f"  (the cited number was 3({_WRONG_POINT}) — a wrong number the guard must ignore)")
    if failures:
        print(f"\n  {len(failures)} unresolved/mismatched (find_by_term or extraction gaps):")
        for r in failures[:20]:
            why = "no term extracted" if not r["extraction_ok"] else (
                "unresolved" if not r["resolved_id"] else f"→ {r['resolved_id']}")
            print(f"    {r['celex']}  {r['term']!r:40} {why}")
    else:
        print("\n  ✅ every definition-want resolved to the correct provision by term.")

    summary = {
        "n": n, "extraction_ok": extr_ok, "resolution_ok": res_ok,
        "resolution_rate": round(res_ok / n, 4),
        "failures": [{"celex": r["celex"], "term": r["term"],
                      "true_def_id": r["true_def_id"], "resolved_id": r["resolved_id"]}
                     for r in failures],
    }
    if out:
        out_path = Path(out)
        if not out_path.is_absolute() and out_path.parent == Path("."):
            out_path = _PROJECT_ROOT / "eval" / "runs" / out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": summary, "cases": results}, indent=2))
        print(f"\n  wrote {out_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit gap-fill grounding eval")
    ap.add_argument("--limit", type=int, default=None, help="test only the first N terms")
    ap.add_argument("--celex", default=None, help="restrict to one regulation")
    ap.add_argument("--out", default=None, help="write JSON (bare name → eval/runs/)")
    args = ap.parse_args()
    run(args.limit, args.celex, args.out)


if __name__ == "__main__":
    main()
