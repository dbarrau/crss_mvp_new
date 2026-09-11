#!/usr/bin/env python
"""Audit that every hardcoded retrieval "anchor" still resolves in the graph.

CRSS compensates for retrieval gaps with a layer of curated guards: regex→ref
tables that force-load a specific provision when a topic appears in the question
(``_CONTEXT_ANCHOR_REFS``), route-level qualification backbones, definition
anchors, and curated OBLIGATION_OF / legal-reasoning edges. Every one of these
names a provision by a *frozen* display_ref (e.g. "Article 43(4)", "Annex XVI").

The danger is silent rot. If a re-ingest, a renumbering, or an amendment moves
or deletes the target, the anchor does not error — it simply resolves to nothing
(MISSING), or worse, ``retrieve_by_refs`` falls back to the parent article
(DEGRADED): the anchor for "Article 43(4)" quietly becomes "load Article 43",
the decisive paragraph never reaches context, and the answer degrades with no
signal. This sweep converts that silent failure into a loud one.

Resolution reuses the exact production path (``_DIRECT_REF_CYPHER`` +
``_parent_article_ref`` + ``ref_norm_key``), so a ref that passes here is a ref
that resolves at query time. Three states per anchor:

    OK        exact display_ref match exists under the anchor's CELEX
    DEGRADED  no exact match, but the parent article exists → silent fallback
    MISSING   neither the ref nor its parent article resolves → hard-broken

Exit code is non-zero if any anchor is DEGRADED or MISSING, so this can gate a
build (run after every re-ingest / consolidation).

    python scripts/audit_anchor_resolution.py
    python scripts/audit_anchor_resolution.py --quiet   # only problems
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# --- production resolution primitives (single source of truth) --------------
from retrieval._cypher import ref_norm_key  # noqa: E402
from retrieval._traversal import (  # noqa: E402
    _DIRECT_REF_CYPHER,
    _parent_article_ref,
)

# --- the anchor tables under audit ------------------------------------------
from application._config import (  # noqa: E402
    _CONTEXT_ANCHOR_REFS,
    _IMPLICIT_PROVISION_REFS,
)
from application._definitions import _ANCHOR_DEFINITION_TERMS  # noqa: E402
from application._retrieval import _DEFINITIONS_REF_BY_CELEX  # noqa: E402
from application._routing import _build_legal_qualification_targets  # noqa: E402
from domain.legislation_catalog import (  # noqa: E402
    AI_ACT_CELEX,
    MDR_CELEX,
    IVDR_CELEX,
    GDPR_CELEX,
)
from domain.ontology.legal_reasoning_chains import (  # noqa: E402
    _ALL_EDGES,
    _ALL_OBLIGATION_PATCHES,
)

# The community-decomposition gate articles are a function-local dict in
# application/_retrieval.py (_retrieve_via_reasoning_chain, `_GATE_ARTICLES`).
# Not importable, so mirrored here — KEEP IN SYNC with that literal.
_DECOMPOSITION_GATE_ARTICLES: dict[str, list[str]] = {
    AI_ACT_CELEX: ["Article 6", "Article 51", "Article 5"],
    MDR_CELEX: ["Article 52", "Article 10"],
    IVDR_CELEX: ["Article 48", "Article 10"],
}


@dataclass
class Check:
    group: str          # which anchor table
    celex: str
    ref: str            # the frozen ref (or term / role_term) the anchor names
    detail: str = ""    # human context (e.g. cue, rationale head)
    kind: str = "ref"   # "ref" | "term" | "role"


# ---------------------------------------------------------------------------
# Resolvers — mirror production semantics exactly
# ---------------------------------------------------------------------------
def resolve_ref(session, ref: str, celex: str) -> str:
    """Return OK / DEGRADED / MISSING for a display_ref under *celex*."""
    rows = session.run(_DIRECT_REF_CYPHER, refs=[ref], celexes=[celex]).data()
    if any(ref_norm_key(r["display_ref"]) == ref_norm_key(ref)
           for r in rows if r.get("display_ref")):
        return "OK"
    parent = _parent_article_ref(ref)
    if parent:
        prows = session.run(_DIRECT_REF_CYPHER, refs=[parent], celexes=[celex]).data()
        if any(ref_norm_key(r["display_ref"]) == ref_norm_key(parent)
               for r in prows if r.get("display_ref")):
            return "DEGRADED"
    return "MISSING"


def resolve_term(session, term: str, celex: str) -> str:
    """Return OK / MISSING for a DefinedTerm under *celex*."""
    tn = re.sub(r"\s+", "_", term.strip().lower())
    row = session.run(
        "MATCH (d:DefinedTerm) WHERE d.term_normalized = $tn AND d.celex = $celex "
        "RETURN d LIMIT 1",
        tn=tn, celex=celex,
    ).single()
    return "OK" if row else "MISSING"


def resolve_role(session, role_term: str, celex: str) -> str:
    """Return OK / MISSING for an ActorRole under *celex*."""
    row = session.run(
        "MATCH (r:ActorRole) WHERE r.term_normalized = $rt AND r.celex = $celex "
        "RETURN r LIMIT 1",
        rt=role_term, celex=celex,
    ).single()
    return "OK" if row else "MISSING"


# ---------------------------------------------------------------------------
# Collect every anchor into a flat list of Checks
# ---------------------------------------------------------------------------
def _short(text: str, n: int = 60) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def collect_checks() -> list[Check]:
    checks: list[Check] = []

    for _pattern, celex, ref in _CONTEXT_ANCHOR_REFS:
        checks.append(Check("context_anchor_refs", celex, ref,
                            detail=_short(_pattern.pattern)))

    for _pattern, celex, ref in _IMPLICIT_PROVISION_REFS:
        checks.append(Check("implicit_provision_refs", celex, ref,
                            detail=_short(_pattern.pattern)))

    for celex, term in _ANCHOR_DEFINITION_TERMS.items():
        checks.append(Check("anchor_definition_terms", celex, term, kind="term"))

    for celex, ref in _DEFINITIONS_REF_BY_CELEX.items():
        checks.append(Check("definitions_ref_by_celex", celex, ref))

    for celex, refs in _DECOMPOSITION_GATE_ARTICLES.items():
        for ref in refs:
            checks.append(Check("decomposition_gate_articles", celex, ref))

    # Qualification backbone: exercise the builder across a matrix engineered to
    # trip every branch, then union the emitted (ref, celex) targets. Auto-syncs
    # with the source — if a branch stops emitting a ref, it drops from coverage
    # (visible in the printed count) rather than silently asserting a stale ref.
    all_regs = {
        "EU AI Act", "MDR 2017/745", "IVDR 2017/746",
        "General Data Protection Regulation (GDPR) 2016/679",
    }
    backbone_questions = [
        # actor-status + modification/role-transition (→ Art 25) + Annex III +
        # DPIA + controller/processor cluster + exemption/hospital
        ("When does a deployer become a provider after a substantial modification "
         "of a high-risk AI system listed in Annex III? Consider the DPIA, prior "
         "consultation, controller and processor roles, personal data breach "
         "notification, and the exemption for a hospital placing it on the market.",
         [("manufacturer", MDR_CELEX)]),
        # in-house developer branch (suppresses Art 25) + exemption/in-house
        ("A hospital develops an in-house AI system; what are the manufacturer "
         "obligations and the in-house exemption when placing it on the market?",
         [("manufacturer", MDR_CELEX)]),
    ]
    seen_backbone: set[tuple[str, str]] = set()
    for question, role_specs in backbone_questions:
        for target in _build_legal_qualification_targets(
            question, mentioned_regs=all_regs, role_specs=role_specs,
        ):
            celexes = target.celexes or frozenset()
            for celex in (celexes or {""}):
                key = (target.ref, celex)
                if celex and key not in seen_backbone:
                    seen_backbone.add(key)
                    checks.append(Check("qualification_backbone", celex, target.ref))

    for edge in _ALL_EDGES:
        checks.append(Check("reasoning_edges", edge.celex, edge.source_ref,
                            detail=f"{edge.type} source"))
        target_celex = edge.cross_celex or edge.celex
        for tref in edge.target_refs:
            checks.append(Check("reasoning_edges", target_celex, tref,
                                detail=f"{edge.type} target"))

    for patch in _ALL_OBLIGATION_PATCHES:
        checks.append(Check("obligation_patches", patch.celex, patch.provision_ref,
                            detail=_short(patch.rationale)))
        checks.append(Check("obligation_patches", patch.celex, patch.role_term,
                            detail="role_term", kind="role"))

    return checks


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true",
                    help="print only DEGRADED / MISSING anchors")
    args = ap.parse_args()

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USERNAME", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "testpassword"))

    checks = collect_checks()
    results: list[tuple[Check, str]] = []
    with GraphDatabase.driver(uri, auth=auth) as drv, drv.session() as s:
        for c in checks:
            if c.kind == "term":
                status = resolve_term(s, c.ref, c.celex)
            elif c.kind == "role":
                status = resolve_role(s, c.ref, c.celex)
            else:
                status = resolve_ref(s, c.ref, c.celex)
            results.append((c, status))

    # Dedupe identical (group, celex, ref, kind) rows so a ref reused across
    # several regexes is reported once.
    seen: set[tuple] = set()
    deduped: list[tuple[Check, str]] = []
    for c, status in results:
        key = (c.group, c.celex, c.ref, c.kind)
        if key not in seen:
            seen.add(key)
            deduped.append((c, status))
    results = deduped

    ok = sum(1 for _c, s in results if s == "OK")
    degraded = [(c, s) for c, s in results if s == "DEGRADED"]
    missing = [(c, s) for c, s in results if s == "MISSING"]

    # Report, grouped by anchor table
    by_group: dict[str, list[tuple[Check, str]]] = {}
    for c, status in results:
        by_group.setdefault(c.group, []).append((c, status))

    print("=" * 78)
    print("ANCHOR RESOLUTION AUDIT")
    print("=" * 78)
    for group in sorted(by_group):
        rows = by_group[group]
        g_ok = sum(1 for _c, s in rows if s == "OK")
        print(f"\n{group}  ({g_ok}/{len(rows)} OK)")
        for c, status in sorted(rows, key=lambda r: (r[1] != "MISSING",
                                                     r[1] != "DEGRADED", r[0].ref)):
            if args.quiet and status == "OK":
                continue
            mark = {"OK": "  ok ", "DEGRADED": " DEGR", "MISSING": "MISS!"}[status]
            label = {"term": "term", "role": "role"}.get(c.kind, "")
            tag = f" ({label})" if label else ""
            detail = f"   — {c.detail}" if c.detail and status != "OK" else ""
            print(f"  [{mark}] {c.celex}  {c.ref}{tag}{detail}")

    print("\n" + "=" * 78)
    print(f"TOTAL {len(results)} anchors:  {ok} OK  |  "
          f"{len(degraded)} DEGRADED  |  {len(missing)} MISSING")
    print("=" * 78)

    if degraded or missing:
        print("\nBROKEN ANCHORS (these silently degrade answers):")
        for c, status in missing + degraded:
            print(f"  {status:8} {c.group}: {c.celex} {c.ref} "
                  f"({'exact ref gone; ' if status == 'DEGRADED' else ''}"
                  f"{c.detail})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
