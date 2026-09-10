#!/usr/bin/env python3
"""Audit how well :DefinedTerm nodes are connected to the knowledge graph.

A defined term is only useful if it is reachable — structurally (the provisions
that use it link to it; if it names an actor, an :ActorRole materialises it) and
at query time (it is in the retrieval index and its definition can be surfaced).
When a term is silently disconnected, the agent falls back to the model's memory
for that concept — the exact fabrication risk this system exists to remove.

Finding these one "what about?" at a time does not scale. This is the standing
sweep: it scores every DefinedTerm on a handful of connectivity signals and
reports only the UNEXPECTED gaps.

The hard part is not detection — it is not crying wolf. A naive orphan sweep
flags every generic term (``risk`` appears in 811 AI Act provisions with zero
USES_TERM edges) — but those are the *deliberate* ``term_linker._SKIP_TERMS``
stoplist (generic single words are excluded on purpose so specific multi-word
terms keep the signal). So this audit reconciles against intent BEFORE flagging:

  * intentional USES_TERM exclusions (the shared ``_SKIP_TERMS`` stoplist), and
  * a usage-magnitude floor (a term used in < USAGE_FLOOR provisions is a benign
    rare definition, not a linker failure).

Only a term that is disconnected AND not intentionally excluded AND not rare is a
FAIL. Actor detection likewise does not trust the parser's ``category`` (it labels
authorities ``body``/``other``, and a substring rescue false-positives on ``Body
orifice``): actor candidates are surfaced as REVIEW from the definition-body cue.

Tiers:
  FAIL   (exit non-zero) — deterministic, no judgement:
           · a DefinedTerm whose ``source_provision_id`` resolves to no node
           · a non-stoplist term used in >= USAGE_FLOOR provisions with 0 USES_TERM
           · a category='actor' term with no :ActorRole (parser said actor)
  REVIEW (exit 0)        — heuristic, human triages:
           · an institutional-actor term (definition cue) with no :ActorRole
  OK                     — reconciled: stoplist exclusions + benign rare terms

    python scripts/audit_defined_term_connectivity.py
    python scripts/audit_defined_term_connectivity.py --json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from canonicalization.term_linker import _SKIP_TERMS          # intentional USES_TERM exclusions
from domain.ontology.defined_terms import ACTOR_SIGNALS       # definition-body actor phrases

# A term used in fewer than this many provisions is a benign rare definition, so
# its absence of USES_TERM edges is not a linker failure. (Measured: the real
# generic terms sit at 100s of provisions and are all stoplisted; genuinely rare
# terms sit at <= 8.)
USAGE_FLOOR = 10

# Institutional-actor cues in the DEFINITION body — catch authorities and the
# conformity-assessment body ("a body that performs…", "the authority responsible
# for…") that the parser filed under category 'body'/'other', while excluding
# non-actors ("Body orifice" = "any natural or artificial opening in the body…";
# "personal data" = "relating to an identified or identifiable natural person").
# Uses the full "natural or legal person" phrase, never bare "person", on purpose.
_ACTOR_DEF_CUES = tuple(ACTOR_SIGNALS) + (
    "authority responsible", "authority competent", "body responsible",
    "body that performs", "body which performs", "person referred to in",
)

_LOAD = """
MATCH (t:DefinedTerm)
OPTIONAL MATCH (t)<-[u:USES_TERM]-()
WITH t, count(u) AS uses
OPTIONAL MATCH (a:ActorRole {source_defined_term_id: t.id})
WITH t, uses, count(a) AS roles
OPTIONAL MATCH (p:Provision {id: t.source_provision_id})
RETURN t.term AS term, t.category AS category, t.celex AS celex,
       t.source_provision_id AS src, uses, roles,
       (p IS NOT NULL) AS def_ok,
       toLower(coalesce(p.text_for_analysis, p.text, '')) AS def_text
ORDER BY celex, term
"""

_USAGE = (
    "MATCH (p:Provision {celex:$c}) "
    "WHERE toLower(coalesce(p.text_for_analysis, p.text, '')) CONTAINS $t "
    "RETURN count(p) AS n"
)


def _is_actor_candidate(category: str | None, def_text: str) -> bool:
    if category == "actor":
        return True
    return any(cue in def_text for cue in _ACTOR_DEF_CUES)


def main(as_json: bool = False) -> int:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USERNAME", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "testpassword"))
    fails: list[dict] = []
    reviews: list[dict] = []
    stoplist_orphans: list[dict] = []
    rare_orphans: list[dict] = []

    with GraphDatabase.driver(uri, auth=auth) as drv, drv.session() as s:
        rows = s.run(_LOAD).data()
        total = len(rows)
        for r in rows:
            term, celex = r["term"], r["celex"]
            low = term.lower()

            # (1) broken definition link — deterministic FAIL
            if r["src"] and not r["def_ok"]:
                fails.append({"kind": "broken_definition", "term": term,
                              "celex": celex, "detail": f"source_provision_id {r['src']} resolves to no node"})

            # (2) USES_TERM connectivity — reconcile before flagging
            if r["uses"] == 0:
                if low in _SKIP_TERMS:
                    stoplist_orphans.append({"term": term, "celex": celex})
                else:
                    usage = s.run(_USAGE, c=celex, t=low).single()["n"]
                    if usage >= USAGE_FLOOR:
                        fails.append({"kind": "uses_term_disconnect", "term": term, "celex": celex,
                                      "detail": f"used in {usage} provisions but has 0 USES_TERM edges "
                                                "(not on the _SKIP_TERMS stoplist)"})
                    else:
                        rare_orphans.append({"term": term, "celex": celex, "usage": usage})

            # (3) actor without role
            if r["roles"] == 0 and _is_actor_candidate(r["category"], r["def_text"]):
                rec = {"term": term, "celex": celex, "category": r["category"]}
                if r["category"] == "actor":
                    rec["kind"] = "actor_category_without_role"
                    fails.append(rec)                       # parser said actor → deterministic FAIL
                else:
                    reviews.append(rec)                     # institutional-actor cue → REVIEW

    if as_json:
        print(json.dumps({"total": total, "fail": fails, "review": reviews,
                          "reconciled": {"stoplist": stoplist_orphans, "rare": rare_orphans}}, indent=2))
        return len(fails)

    print(f"DefinedTerm connectivity audit — {total} terms\n")
    if fails:
        print(f"✗ FAIL — {len(fails)} unexpected disconnect(s):")
        for f in fails:
            print(f"    [{f['kind']}] {f['celex']}  {f['term']!r} — {f.get('detail','')}")
    else:
        print("✓ FAIL tier clean — no unexpected disconnects.")
    if reviews:
        print(f"\n⚠ REVIEW — {len(reviews)} institutional-actor term(s) with no :ActorRole "
              "(triage: give a role, or confirm not an actor):")
        for r in reviews:
            print(f"    {r['celex']}  {r['term']!r} (category={r['category']})")
    print(f"\n· Reconciled (not findings): {len(stoplist_orphans)} deliberate _SKIP_TERMS exclusion(s), "
          f"{len(rare_orphans)} benign rare term(s) below the {USAGE_FLOOR}-provision floor.")
    print("\nFAIL is a build gate; REVIEW is advisory. Re-run after every canonicalization.")
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(1 if main("--json" in sys.argv) else 0)
