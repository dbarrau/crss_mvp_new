#!/usr/bin/env python
"""Measure whether each curated GRAPH EDGE earns its keep.

The context anchors and the qualification backbone are query-time force-loads
(see scripts/eval_anchor_ablation.py). The curated edges in
domain/ontology/legal_reasoning_chains.py are a different kind of guard: they are
baked into the graph at canonicalization and change *traversal* —
- reasoning edges (TRIGGERS_OBLIGATION_CLUSTER / IS_PREREQUISITE_FOR /
  REQUIRES_PRIOR_CHECK / DEROGATES_FROM) are followed by retrieve_by_chain, which
  runs ONLY on the classification_chain route, seeded by the classification gate
  articles (_GATE_ARTICLES: AI 6/51/5, MDR 52/10, IVDR 48/10);
- curated OBLIGATION_OF patches feed retrieve_by_roles (the role channel).

An edge's job is to make its payload reachable via its own traversal. So this
measures that directly and structurally (no LLM, no eval questions needed): delete
the one edge, re-run the traversal, diff, restore. Fast (~one Neo4j traversal per
edge) and question-independent.

Verdicts:
  LOAD-BEARING  deleting the edge removes payload from its traversal (unique path)
  REDUNDANT     payload still reachable without it (another edge provides it)
  UNREACHABLE   the edge is never traversed at all — its payload is not reachable
                from the seeds/role even WITH it (e.g. DEROGATES_FROM, which
                retrieve_by_chain does not follow; or a reg with no chain gate)
  INEFFECTIVE   (OBLIGATION_OF) the patched provision is not surfaced to the role
                even with the edge (role-node / resolution mismatch)

SAFETY: edges are deleted one at a time and recreated immediately via the loader's
own MERGE cyphers; a final (and on-crash) full reload from source
(load_edges + load_obligation_patches, both idempotent) guarantees the graph is
restored to exactly the committed source of truth.

    python scripts/eval_edge_ablation.py
    python scripts/eval_edge_ablation.py --out edge_ablation_v1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from retrieval._cypher import ref_norm_key  # noqa: E402
from domain.ontology.legal_reasoning_chains import (  # noqa: E402
    _ALL_EDGES,
    _ALL_OBLIGATION_PATCHES,
)
from scripts.load_legal_reasoning_chains import (  # noqa: E402
    _CREATE_EDGE_CYPHER,
    _CREATE_OBLIGATION_OF_CYPHER,
    _resolve_ref,
    load_edges,
    load_obligation_patches,
)

# Mirror of the function-local _GATE_ARTICLES in application/_retrieval.py
# (_expand_classification_chain). retrieve_by_chain is seeded ONLY from these, so
# an edge unreachable from them is never used by the chain traversal. KEEP IN SYNC.
_GATE_ARTICLES: dict[str, list[str]] = {
    "32024R1689": ["Article 6", "Article 51", "Article 5"],
    "32017R0745": ["Article 52", "Article 10"],
    "32017R0746": ["Article 48", "Article 10"],
}

_DELETE_EDGE_CYPHER = """\
MATCH (src)-[r:`{rel_type}` {source_file: 'legal_reasoning_chains.py'}]->(tgt)
WHERE src.id = $src_id AND tgt.id = $tgt_id
DELETE r RETURN count(r) AS n
"""
_DELETE_OBLIGATION_CYPHER = """\
MATCH (p)-[e:OBLIGATION_OF {source_file: 'legal_reasoning_chains.py'}]->(role:ActorRole)
WHERE p.id = $provision_id
  AND role.celex = $celex AND role.term_normalized = $role_term
DELETE e RETURN count(e) AS n
"""


def _reachable_by_chain(retriever, celex: str) -> set[tuple[str, str]]:
    """(celex, norm_ref) set reachable by retrieve_by_chain from celex's gates."""
    gates = _GATE_ARTICLES.get(celex)
    if not gates:
        return set()
    rows = retriever.retrieve_by_chain(gates, celex)
    return {((r.get("celex") or ""), ref_norm_key(r.get("article_ref") or ""))
            for r in rows if r.get("article_ref")}


def _role_refs(retriever, role_term: str, celex: str) -> set[str]:
    """Norm-ref set of a role's obligation provisions (uncapped k)."""
    rows = retriever.retrieve_by_roles([(role_term, celex)], k=200)
    return {ref_norm_key(r.get("article_ref") or "") for r in rows if r.get("article_ref")}


def _ref_hit(target_norm: str, present: set[str]) -> bool:
    """A target is present if an exact norm-ref matches or extends it at '('."""
    return any(r == target_norm or r.startswith(target_norm + "(") for r in present)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="write JSON results (bare name → eval/runs/)")
    args = ap.parse_args()

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USERNAME", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "testpassword"))
    db = os.environ.get("NEO4J_DATABASE", "neo4j")

    from retrieval.graph_retriever import GraphRetriever
    retriever = GraphRetriever()
    driver = GraphDatabase.driver(uri, auth=auth)

    results: list[dict] = []
    t0 = time.perf_counter()
    try:
        # ── Reasoning (chain) edges ──────────────────────────────────────────
        # Baseline reachability per gate-seeded celex, computed once up front.
        base_chain = {celex: _reachable_by_chain(retriever, celex) for celex in _GATE_ARTICLES}
        print(f"Reasoning edges: {len(_ALL_EDGES)}   OBLIGATION_OF patches: "
              f"{len(_ALL_OBLIGATION_PATCHES)}\n")
        print("── reasoning (chain) edges ──")
        for edge in _ALL_EDGES:
            tgt_celex = edge.cross_celex or edge.celex
            targets = [( tgt_celex, ref_norm_key(t)) for t in edge.target_refs]
            if edge.celex not in _GATE_ARTICLES:
                status = "UNREACHABLE"     # reg has no classification gate seeds
                results.append({"guard": "reasoning_edge", "type": edge.type,
                                "celex": edge.celex, "ref": edge.source_ref,
                                "targets": list(edge.target_refs), "status": status,
                                "lost": []})
                print(f"[{status:12}] {edge.type:26} {edge.celex} {edge.source_ref}"
                      f" (reg has no chain gate)")
                continue

            base = base_chain[edge.celex]
            in_base = [t for t in targets if _ref_hit(t[1],
                       {r for (cx, r) in base if cx == t[0]})]
            with driver.session(database=db) as s:
                src = _resolve_ref(s, edge.source_ref, edge.celex)
                lost: list[str] = []
                if src.node_id and in_base:
                    for t in edge.target_refs:
                        tr = _resolve_ref(s, t, tgt_celex)
                        if tr.node_id:
                            s.run(_DELETE_EDGE_CYPHER.replace("{rel_type}", edge.type),
                                  src_id=src.node_id, tgt_id=tr.node_id)
                    ab = _reachable_by_chain(retriever, edge.celex)
                    for (cx, nr) in (base - ab):
                        lost.append(nr)
                    # restore this edge's targets immediately
                    for t in edge.target_refs:
                        tr = _resolve_ref(s, t, tgt_celex)
                        if tr.node_id:
                            s.run(_CREATE_EDGE_CYPHER.replace("{rel_type}", edge.type),
                                  src_id=src.node_id, tgt_id=tr.node_id,
                                  rationale=edge.rationale)
            if not in_base:
                status = "UNREACHABLE"     # targets never reached from gates via this edge
            elif lost:
                status = "LOAD-BEARING"
            else:
                status = "REDUNDANT"
            results.append({"guard": "reasoning_edge", "type": edge.type,
                            "celex": edge.celex, "ref": edge.source_ref,
                            "targets": list(edge.target_refs), "status": status,
                            "lost": lost})
            print(f"[{status:12}] {edge.type:26} {edge.celex} {edge.source_ref}"
                  f" -> {', '.join(edge.target_refs)[:60]}")

        # ── Curated OBLIGATION_OF patches ────────────────────────────────────
        print("\n── curated OBLIGATION_OF patches ──")
        for patch in _ALL_OBLIGATION_PATCHES:
            tnorm = ref_norm_key(patch.provision_ref)
            base = _role_refs(retriever, patch.role_term, patch.celex)
            present_base = _ref_hit(tnorm, base)
            present_ab = present_base
            if present_base:
                with driver.session(database=db) as s:
                    res = _resolve_ref(s, patch.provision_ref, patch.celex)
                    if res.node_id:
                        s.run(_DELETE_OBLIGATION_CYPHER, provision_id=res.node_id,
                              celex=patch.celex, role_term=patch.role_term)
                        present_ab = _ref_hit(tnorm, _role_refs(
                            retriever, patch.role_term, patch.celex))
                        s.run(_CREATE_OBLIGATION_OF_CYPHER, provision_id=res.node_id,
                              celex=patch.celex, role_term=patch.role_term,
                              rationale=patch.rationale)
            if not present_base:
                status = "INEFFECTIVE"
            elif present_ab:
                status = "REDUNDANT"
            else:
                status = "LOAD-BEARING"
            results.append({"guard": "obligation_patch", "celex": patch.celex,
                            "ref": patch.provision_ref, "role": patch.role_term,
                            "status": status})
            print(f"[{status:12}] {patch.celex} {patch.provision_ref} -> {patch.role_term}")
    finally:
        # Bulletproof restore from the committed source of truth (idempotent).
        print("\nRestoring all curated edges from source (idempotent)…")
        load_edges(driver, db)
        load_obligation_patches(driver, db)
        driver.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    order = {"LOAD-BEARING": 0, "REDUNDANT": 1, "UNREACHABLE": 2, "INEFFECTIVE": 3}
    for guard in ("reasoning_edge", "obligation_patch"):
        rows = [r for r in results if r["guard"] == guard]
        buckets: dict[str, list[str]] = {}
        for r in rows:
            label = f"{r.get('type','') } {r['ref']}".strip()
            buckets.setdefault(r["status"], []).append(label)
        print("\n" + "=" * 78)
        print(f"{guard.upper()} SUMMARY  ({len(rows)} edges)")
        print("=" * 78)
        for status in sorted(buckets, key=lambda s: order.get(s, 9)):
            print(f"\n{status}  ({len(buckets[status])})")
            for label in buckets[status]:
                print(f"    {label}")
    print(f"\n({round(time.perf_counter()-t0)}s)")

    if args.out:
        p = Path(args.out)
        if not p.is_absolute() and p.parent == Path("."):
            p = ROOT / "eval" / "runs" / p.name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=2))
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
