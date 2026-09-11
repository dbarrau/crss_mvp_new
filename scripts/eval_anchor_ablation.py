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
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

os.environ["CRSS_GRAPH_EXPANSION"] = "1"
os.environ["CRSS_CLARIFY"] = "0"

from application._config import _CONTEXT_ANCHOR_REFS, _REG_NAME_TO_CELEX  # noqa: E402
from domain.legislation_catalog import GDPR_CELEX  # noqa: E402
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


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).strip().lower()


def _target_body(retriever, celex: str, ref: str) -> str:
    """Ground-truth body text of the anchor target, path prefix stripped.

    Searches the direct-ref lookup's top provisions and their children for the
    node whose ref matches *ref* (folded via ref_norm_key). The body after the
    " | " path separator is the actual provision text — what must reach the LLM.
    """
    tk = ref_norm_key(ref)
    for p in retriever.retrieve_by_refs([ref], celex_filter={celex}):
        if ref_norm_key(p.get("article_ref") or "") == tk:
            return (p.get("article_text") or "").split(" | ", 1)[-1]
        for c in p.get("children") or []:
            if ref_norm_key(c.get("ref") or "") == tk:
                return (c.get("text") or "").split(" | ", 1)[-1]
    return ""


def _render(provisions: list[dict], det) -> str:
    """Render provisions exactly as the agent does for this route."""
    from application._context import _SUBJECT_RENDER_ROUTES, _format_context
    return _format_context(
        provisions, allow_subject_render=det.route.id in _SUBJECT_RENDER_ROUTES)


def _text_present(body: str, rendered: str) -> bool:
    """Does the target's *body text* appear in the *rendered* context string?"""
    b = _norm(body)
    if len(b) < 40:                     # body too short to fingerprint reliably
        return bool(b) and b in _norm(rendered)
    probe = b[15:95]                    # skip the leading "N." number, take 80 chars
    return probe in _norm(rendered)


def _retrieved_as(target: str, provisions: list[dict]) -> str | None:
    """Is the target retrieved as its own top block, a child, or not at all?"""
    tk = ref_norm_key(target)
    for p in provisions:
        if ref_norm_key(p.get("article_ref") or "") == tk:
            return "top"
    for p in provisions:
        for c in p.get("children") or []:
            if ref_norm_key(c.get("ref") or "") == tk:
                return "child"
    return None


def _present(target: str, body: str, provisions: list[dict], det) -> bool:
    """Does the target's CONTENT actually reach the LLM?

    The decisive test for a legal tool combines two signals that each fail alone:
    - a ref-label match over-counts: a paragraph rolled up under a retrieved
      parent shows its label but the subtree render caps its text out
      (false "present": Article 2(6)/43(4) on HQ_001);
    - a raw body-text match under-counts the OTHER way: the same distinctive
      phrase can appear elsewhere in context while the provision itself was never
      retrieved (false "present": Article 4/51 on HQ_039/HQ_027, text coincides
      but the provision is trimmed out entirely).

    So a target counts as present only if it is actually RETRIEVED (its own block,
    or a child) AND — for a child, which the subtree cap can truncate — its body
    text survives into the render. An own block always renders its text.
    """
    kind = _retrieved_as(target, provisions)
    if kind is None:
        return False
    if kind == "top":
        return True
    return _text_present(body, _render(provisions, det))


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


def _budgeted_after_trim(provisions: list[dict], definitions: list[dict], det) -> list[dict]:
    """Replicate the agent's context-budget trim (application/agent.py ~800).

    Retrieval presence is necessary but not sufficient: the agent reserves budget
    for the definitions/applicability block, then trims the provision tail to fit
    _CONTEXT_CHAR_BUDGET, keeping the pinned backbone tier (_direct_ref_match)
    first. A force-loaded anchor pins its target into that surviving tier; the
    SAME provision arriving as a lower-tier vector hit (anchor off) can be trimmed
    out. This returns the provisions that actually survive into the rendered
    context — the set the LLM really sees.
    """
    from datetime import date

    from application.agent import (
        _CONTEXT_CHAR_BUDGET,
        _collect_context_celexes,
        _format_definitions,
    )
    from application._context import _SUBJECT_RENDER_ROUTES, _trim_provisions_to_budget
    from domain.ontology.applicability import applicability_note

    parts: list[str] = []
    celexes = _collect_context_celexes(provisions, definitions) or set(
        det.target_celexes or set())
    note = applicability_note(celexes, date.today())
    if note:
        parts.append(note)
    if definitions:
        parts.append("LEGAL DEFINITIONS (from the definitions article):\n"
                     + _format_definitions(definitions))
    sep = "\n\n---\n\n"
    reserved = sum(len(p) for p in parts) + len(sep) * len(parts)
    prov_budget = max(0, _CONTEXT_CHAR_BUDGET - reserved)
    allow_subject = det.route.id in _SUBJECT_RENDER_ROUTES
    return _trim_provisions_to_budget(provisions, prov_budget,
                                      allow_subject_render=allow_subject)


def _verdict(on_p: bool, off_p: bool) -> str:
    if not on_p:
        return "INEFFECTIVE"
    return "REDUNDANT" if off_p else "LOAD-BEARING"


def _aggregate(per_case: list[dict], key: str, firing: list) -> str:
    if not firing:
        return "UNTESTED"
    verdicts = [pc[key] for pc in per_case]
    if any(v == "LOAD-BEARING" for v in verdicts):
        return "LOAD-BEARING"
    if all(v == "INEFFECTIVE" for v in verdicts):
        return "INEFFECTIVE"
    return "REDUNDANT"


def _ablate_target(ref, celex, firing, retriever, client, det_cache, off_fn) -> dict:
    """Run the on/off ablation for one target over its firing cases.

    ``off_fn(question, det) -> (provisions, definitions, det)`` is the
    mechanism-specific ablated retrieval (drop the anchor from context_anchor_refs
    for context anchors, or monkeypatch the backbone builder for backbone refs).
    The ON arm is always the unmodified production retrieval.
    """
    body = _target_body(retriever, celex, ref) if firing else ""
    per_case: list[dict] = []
    for c in firing:
        det = det_cache[c["id"]]
        on_prov, on_def, on_det = _retrieve(
            c["question"], retriever, client, anchor_refs=list(det.context_anchor_refs or []))
        off_prov, off_def, off_det = off_fn(c["question"], det)
        # Presence = the provision's CONTENT reaches the LLM (retrieved AND text
        # survives render), never a bare ref-label. retrieval-level = untrimmed
        # bag; render-level = also survives the context-budget trim.
        ret_on = _present(ref, body, on_prov, on_det)
        ret_off = _present(ref, body, off_prov, off_det)
        on_budg = _budgeted_after_trim(on_prov, on_def, on_det)
        off_budg = _budgeted_after_trim(off_prov, off_def, off_det)
        rnd_on = _present(ref, body, on_budg, on_det)
        rnd_off = _present(ref, body, off_budg, off_det)
        per_case.append({
            "case": c["id"],
            "retrieval_verdict": _verdict(ret_on, ret_off),
            "render_verdict": _verdict(rnd_on, rnd_off),
            "ret_on": ret_on, "ret_off": ret_off, "rnd_on": rnd_on, "rnd_off": rnd_off,
            "trimmed": len(off_prov) - len(off_budg),
        })
    retrieval_status = _aggregate(per_case, "retrieval_verdict", firing)
    render_status = _aggregate(per_case, "render_verdict", firing)
    rec = {"celex": celex, "ref": ref, "status": render_status,
           "retrieval_status": retrieval_status, "render_status": render_status,
           "n_firing": len(firing), "cases": per_case}
    flag = ("  <-- trim-promoted" if (render_status == "LOAD-BEARING"
                                      and retrieval_status == "REDUNDANT") else "")
    detail = f"  cases: {', '.join(pc['case'] for pc in per_case)}" if firing else ""
    print(f"[render {render_status:12} | retrieval {retrieval_status:12}] "
          f"{celex}  {ref}  (fires {len(firing)}){detail}{flag}")
    return rec


def _report(results: list[dict], t0: float, label: str, out: str | None) -> None:
    order = {"LOAD-BEARING": 0, "REDUNDANT": 1, "INEFFECTIVE": 2, "UNTESTED": 3}
    buckets: dict[str, list[str]] = {}
    for r in results:
        buckets.setdefault(r["render_status"], []).append(r["ref"])
    print("\n" + "=" * 78)
    print(f"{label} LOAD-BEARING SUMMARY (render-level)   ({round(time.perf_counter()-t0)}s)")
    print("=" * 78)
    for status in sorted(buckets, key=lambda s: order.get(s, 9)):
        print(f"\n{status}  ({len(buckets[status])})")
        for ref in buckets[status]:
            print(f"    {ref}")
    promoted = [r for r in results if r["render_status"] == "LOAD-BEARING"
                and r["retrieval_status"] == "REDUNDANT"]
    if promoted:
        print("\nTRIM-PROMOTED (redundant in the retrieved bag, but the anchor pins it")
        print("into the budget-surviving tier — deleting it would drop it at render):")
        for r in promoted:
            print(f"    {r['ref']}")
    if out:
        p = Path(out)
        if not p.is_absolute() and p.parent == Path("."):
            p = ROOT / "eval" / "runs" / p.name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {p}")


def _backbone_targets_for(det, question, orig_builder) -> list[tuple[str, str]]:
    """The (ref, celex) targets the qualification backbone force-loads for a case.

    Fires on ``legal_qualification`` (always) and ``cross_regulation`` when GDPR
    is in scope (see application/_retrieval.py seed phase). Mirrors that gate and
    the mentioned_regs derivation, then reads what the builder actually emits.
    """
    rid = det.route.id
    if rid == "legal_qualification":
        pass
    elif rid == "cross_regulation" and det.target_celexes and GDPR_CELEX in det.target_celexes:
        pass
    else:
        return []
    mentioned = {name for name, celex in _REG_NAME_TO_CELEX.items()
                 if det.target_celexes and celex in det.target_celexes}
    out: list[tuple[str, str]] = []
    for t in orig_builder(question, mentioned_regs=mentioned, role_specs=det.role_specs):
        for cx in (t.celexes or frozenset()):
            out.append((t.ref, cx))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="write JSON results (bare name → eval/runs/)")
    ap.add_argument("--anchor", help="restrict to targets whose ref contains this")
    ap.add_argument("--mode", choices=["context", "backbone", "all"], default="context",
                    help="context anchors (_CONTEXT_ANCHOR_REFS), the qualification "
                         "backbone (_build_legal_qualification_targets), or both")
    args = ap.parse_args()

    cases = _load_questions()
    from mistralai.client import Mistral
    from retrieval.graph_retriever import GraphRetriever
    retriever = GraphRetriever()
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    from application.agent import _detect_scenario
    det_cache: dict[str, object] = {c["id"]: _detect_scenario(c["question"], retriever, 12)
                                    for c in cases}
    t0 = time.perf_counter()
    all_results: list[dict] = []

    # ── Context anchors: ablate by dropping the ref from context_anchor_refs ──
    if args.mode in ("context", "all"):
        targets, seen = [], set()
        for _pat, celex, ref in _CONTEXT_ANCHOR_REFS:
            if args.anchor and args.anchor.lower() not in ref.lower():
                continue
            if (celex, ref) not in seen:
                seen.add((celex, ref))
                targets.append((celex, ref))
        print(f"CONTEXT anchors: {len(targets)}   Eval questions pooled: {len(cases)}\n")
        results = []
        for celex, ref in targets:
            firing = [c for c in cases
                      if ref in (det_cache[c["id"]].context_anchor_refs or [])]

            def off_fn(question, det, _ref=ref):
                anchors = [a for a in (det.context_anchor_refs or []) if a != _ref]
                return _retrieve(question, retriever, client, anchor_refs=anchors)

            rec = _ablate_target(ref, celex, firing, retriever, client, det_cache, off_fn)
            rec["guard"] = "context_anchor"
            results.append(rec)
        all_results += results
        _report(results, t0, "CONTEXT ANCHOR",
                (args.out if args.mode == "context" else None))

    # ── Qualification backbone: ablate by monkeypatching the builder ─────────
    if args.mode in ("backbone", "all"):
        import application._retrieval as R
        orig_builder = R._build_legal_qualification_targets

        fired_bb: dict[str, list[tuple[str, str]]] = {}
        universe, seen = [], set()
        for c in cases:
            emitted = _backbone_targets_for(det_cache[c["id"]], c["question"], orig_builder)
            fired_bb[c["id"]] = emitted
            for key in emitted:
                if key not in seen:
                    seen.add(key)
                    universe.append(key)
        if args.anchor:
            universe = [(rf, cx) for (rf, cx) in universe
                        if args.anchor.lower() in rf.lower()]
        print(f"\nBACKBONE targets: {len(universe)}   Eval questions pooled: {len(cases)}\n")

        results_bb = []
        for ref, celex in universe:
            firing = [c for c in cases if (ref, celex) in fired_bb[c["id"]]]

            def off_fn(question, det, _ref=ref, _celex=celex):
                def patched(*a, **k):
                    return [t for t in orig_builder(*a, **k)
                            if not (t.ref == _ref and _celex in (t.celexes or frozenset()))]
                R._build_legal_qualification_targets = patched
                try:
                    return _retrieve(question, retriever, client,
                                     anchor_refs=list(det.context_anchor_refs or []))
                finally:
                    R._build_legal_qualification_targets = orig_builder

            rec = _ablate_target(ref, celex, firing, retriever, client, det_cache, off_fn)
            rec["guard"] = "qualification_backbone"
            results_bb.append(rec)
        all_results += results_bb
        _report(results_bb, t0, "BACKBONE",
                (args.out if args.mode == "backbone" else None))

    if args.mode == "all" and args.out:
        p = Path(args.out)
        if not p.is_absolute() and p.parent == Path("."):
            p = ROOT / "eval" / "runs" / p.name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(all_results, indent=2))
        print(f"\nWrote combined {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
