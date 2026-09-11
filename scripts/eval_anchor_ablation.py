#!/usr/bin/env python
"""Measure whether each curated context-anchor is LOAD-BEARING or DEAD WEIGHT.

Step 1 (scripts/audit_anchor_resolution.py) proved every anchor still resolves.
This asks the sharper question: does the anchor *earn its keep*? An anchor
force-loads a decisive provision on a topic cue — but if the dense/lexical/graph
channels already surface that provision on the same questions, the anchor is
redundant scar tissue that can be deleted. If they do NOT, the anchor is the
sole reason the provision reaches context, and deleting it silently degrades the
answer.

Method (retrieval-only, no generation — confound-free and cheap):
For every anchor in ``_CONTEXT_ANCHOR_REFS`` and each eval question where the
anchor FIRES (its cue matches and its CELEX is in scope), run the production
retrieval pipeline twice — once as-is, once with *only that anchor* removed from
``context_anchor_refs`` — and check whether the anchor's target provision is
present in the retrieved bag either way.

Per firing case:
    LOAD-BEARING  target present with anchor on, absent with it off
    REDUNDANT     target present even with the anchor off (other channels found it)
    (INEFFECTIVE  target absent even with the anchor on — anchor fires but the
                  provision never survives to context; a separate defect)

Per anchor, aggregated over its firing cases:
    LOAD-BEARING  load-bearing on >=1 case            → keep
    REDUNDANT     fires but redundant on every case   → delete candidate
    UNTESTED      fires on 0 eval questions           → no eval exercises it;
                  cannot measure — write an eval case or reconsider it
    INEFFECTIVE   fires but target never surfaces      → investigate

Requires a live Neo4j + MISTRAL_API_KEY (HyDE/decompose use a small model).
Forces CRSS_GRAPH_EXPANSION=1 (production path) and CRSS_CLARIFY=0 (so role-less
questions retrieve instead of stubbing at the scope gate).

    python scripts/eval_anchor_ablation.py
    python scripts/eval_anchor_ablation.py --out anchor_ablation_v1.json
    python scripts/eval_anchor_ablation.py --anchor "Article 113"   # one anchor
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

os.environ["CRSS_GRAPH_EXPANSION"] = "1"
os.environ["CRSS_CLARIFY"] = "0"

from application._config import _CONTEXT_ANCHOR_REFS  # noqa: E402
from retrieval._cypher import ref_norm_key  # noqa: E402

QUALITY_SET = ROOT / "eval" / "quality_set.json"
GOLDEN_SET = ROOT / "eval" / "golden_set.json"


def _load_questions() -> list[dict]:
    """Pool every eval question (quality + golden) — only the text is needed."""
    cases: list[dict] = []
    for path in (QUALITY_SET, GOLDEN_SET):
        for c in json.loads(path.read_text()):
            if c.get("question"):
                cases.append({"id": c["id"], "question": c["question"]})
    return cases


def _target_present(target: str, provisions: list[dict], definitions: list[dict]) -> bool:
    """Is *target* present in the retrieved bag?

    Matching is on the format-insensitive ``ref_norm_key`` (folds ", point " /
    ", indent " and spaces), so "Annex III, point 4" and "Article 2(6)" compare
    correctly regardless of how a node stores its display_ref. A target is
    present if any retrieved ref equals its key OR extends it at a "(" boundary
    — so the article-level anchor "Article 51" is satisfied by "Article 51(1)",
    while "Article 51" never falsely matches "Article 510", and a point anchor
    ("Annex III, point 4") is NOT satisfied by a parent-only "Annex III".

    (A hand-rolled key, not scripts.check_answer_keys._cite_pattern: that builds
    a trailing ``\\b`` which never matches a ref ending in ")" such as
    "Article 2(6)", silently scoring paren-terminated paragraphs as absent.)
    """
    refs: list[str] = []
    for p in provisions:
        refs.append(p.get("article_ref") or "")
        refs.append(p.get("article_path") or "")
        for c in p.get("children") or []:
            refs.append(c.get("ref") or "")
        for c in p.get("cited_provisions") or []:
            refs.append(c.get("ref") or "")
    for d in definitions:
        refs.append(d.get("article_ref") or "")

    tk = ref_norm_key(target)
    for r in refs:
        if not r:
            continue
        rk = ref_norm_key(r)
        if rk == tk or rk.startswith(tk + "("):
            return True
    return False


def _retrieve(question: str, retriever, client, *, anchor_refs: list[str]):
    """Run the production retrieval phase with a specific context_anchor_refs set.

    Mirrors scripts/eval_graph_ablation._run_arm_retrieval: detection → route
    plan → sufficiency → corrective pass, stopping before generation. Returns
    (provisions, definitions, det).
    """
    from application.agent import (
        _detect_scenario,
        _evaluate_route_sufficiency,
        _expand_definitions_from_provisions,
        _retrieve_route_provisions,
        _run_corrective_retrieval_pass,
    )
    det = _detect_scenario(question, retriever, 12)
    rr = _retrieve_route_provisions(
        question, retriever, client=client, k=det.k, route=det.route,
        target_celexes=det.target_celexes, explicit_refs=det.explicit_refs,
        role_specs=det.role_specs, context_anchor_refs=anchor_refs,
    )
    provisions = rr["provisions"]
    definitions = _expand_definitions_from_provisions(
        provisions, retriever, det.definitions, target_celexes=det.target_celexes,
    )
    sufficiency = _evaluate_route_sufficiency(
        route=det.route, question=question, explicit_refs=det.explicit_refs,
        target_celexes=det.target_celexes, role_specs=det.role_specs,
        provisions=provisions, definitions=definitions,
        direct_provisions=rr["direct_provisions"], role_provisions=rr["role_provisions"],
        legal_qualification_targets=rr["legal_qualification_targets"],
    )
    if not sufficiency["ok"]:
        _run_corrective_retrieval_pass(
            question, retriever, client=client, k=det.k, route=det.route,
            target_celexes=det.target_celexes, explicit_refs=det.explicit_refs,
            role_specs=det.role_specs, provisions=provisions,
            direct_provisions=rr["direct_provisions"], role_provisions=rr["role_provisions"],
            definitions=definitions, sufficiency=sufficiency, hyde_text=rr["hyde_text"],
            legal_qualification_targets=rr["legal_qualification_targets"],
        )
    return provisions, definitions, det


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="write JSON results (bare name → eval/runs/)")
    ap.add_argument("--anchor", help="restrict to anchors whose target ref contains this")
    args = ap.parse_args()

    # Unique target refs under audit (a ref may back several cue regexes).
    targets: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _pat, celex, ref in _CONTEXT_ANCHOR_REFS:
        if args.anchor and args.anchor.lower() not in ref.lower():
            continue
        key = (celex, ref)
        if key not in seen:
            seen.add(key)
            targets.append(key)

    cases = _load_questions()
    print(f"Anchors under audit: {len(targets)}   Eval questions pooled: {len(cases)}\n")

    from mistralai.client import Mistral
    from retrieval.graph_retriever import GraphRetriever
    retriever = GraphRetriever()
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    # Detect each case once; cache det.context_anchor_refs so we know which
    # anchors fire where without re-running detection per anchor.
    from application.agent import _detect_scenario
    fired: dict[str, list[str]] = {}   # case_id -> anchors that fired (ref strings)
    det_cache: dict[str, object] = {}
    for c in cases:
        det = _detect_scenario(c["question"], retriever, 12)
        det_cache[c["id"]] = det
        fired[c["id"]] = list(det.context_anchor_refs or [])

    results: list[dict] = []
    t0 = time.perf_counter()
    for celex, ref in targets:
        firing = [c for c in cases if ref in fired[c["id"]]]
        per_case: list[dict] = []
        for c in firing:
            on_prov, on_def, _ = _retrieve(
                c["question"], retriever, client,
                anchor_refs=fired[c["id"]],
            )
            ablated = [a for a in fired[c["id"]] if a != ref]
            off_prov, off_def, _ = _retrieve(
                c["question"], retriever, client, anchor_refs=ablated,
            )
            on_p = _target_present(ref, on_prov, on_def)
            off_p = _target_present(ref, off_prov, off_def)
            if not on_p:
                verdict = "INEFFECTIVE"
            elif off_p:
                verdict = "REDUNDANT"
            else:
                verdict = "LOAD-BEARING"
            per_case.append({"case": c["id"], "verdict": verdict,
                             "present_on": on_p, "present_off": off_p})

        if not firing:
            status = "UNTESTED"
        elif any(pc["verdict"] == "LOAD-BEARING" for pc in per_case):
            status = "LOAD-BEARING"
        elif all(pc["verdict"] == "INEFFECTIVE" for pc in per_case):
            status = "INEFFECTIVE"
        else:
            status = "REDUNDANT"

        results.append({"celex": celex, "ref": ref, "status": status,
                        "n_firing": len(firing), "cases": per_case})
        detail = ""
        if firing:
            lb = sum(1 for pc in per_case if pc["verdict"] == "LOAD-BEARING")
            detail = f"  load-bearing on {lb}/{len(firing)} firing case(s): " \
                     + ", ".join(pc["case"] for pc in per_case)
        print(f"[{status:12}] {celex}  {ref}   (fires on {len(firing)}){detail}")

    # Summary
    order = {"LOAD-BEARING": 0, "REDUNDANT": 1, "INEFFECTIVE": 2, "UNTESTED": 3}
    buckets: dict[str, list[str]] = {}
    for r in results:
        buckets.setdefault(r["status"], []).append(r["ref"])
    print("\n" + "=" * 78)
    print(f"ANCHOR LOAD-BEARING SUMMARY   ({round(time.perf_counter()-t0)}s)")
    print("=" * 78)
    for status in sorted(buckets, key=lambda s: order.get(s, 9)):
        print(f"\n{status}  ({len(buckets[status])})")
        for ref in buckets[status]:
            print(f"    {ref}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute() and out.parent == Path("."):
            out = ROOT / "eval" / "runs" / out.name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
