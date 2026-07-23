#!/usr/bin/env python3
"""Graph-ablation eval — isolate what the *graph* in GraphRAG actually buys.

``eval_answer_quality.py`` treats the retriever as a black box: it measures the
generated answer but cannot tell you whether the knowledge graph earned its
keep versus a plain dense+lexical vector RAG over the same corpus. That is the
core scientific claim of a GraphRAG system, and the current GraphRAG-eval
literature (GraphRAG-Bench, "RAG vs. GraphRAG: A Systematic Evaluation") is
built precisely around isolating it.

This harness runs each keyed case twice against the *same* retriever, embedder,
chunking, HyDE and generation model — flipping only ``CRSS_GRAPH_EXPANSION``:

  * **full** (``CRSS_GRAPH_EXPANSION=1``) — the production GraphRAG path: dense
    + lexical seed, then CITES cross-references, INTERPRETS guidance, reverse
    cross-reg links, role/chain/community traversal and curated backbones.
  * **flat** (``CRSS_GRAPH_EXPANSION=0``) — a vanilla RAG baseline: the same
    dense+lexical top-k with the matched provision's own body (HAS_PART), but
    every graph-reasoning edge and channel removed.

It then scores both answers with the deterministic, law-grounded answer keys
(``must_cite`` in ``eval/quality_set.json`` via ``check_answer_keys.check_answer``)
— **no LLM judge**, so the only cost is 2× generation and the signal is a clean,
reproducible cite-recall delta. The headline is the set of *decisive citations
the graph recovers that flat retrieval drops*: exactly the multi-hop /
cross-reference provisions a graph is supposed to reach.

Usage
-----
    python scripts/eval_graph_ablation.py                    # all keyed cases
    python scripts/eval_graph_ablation.py --case HQ_001 HQ_003
    python scripts/eval_graph_ablation.py --limit 6 --out /tmp/ablation.json

Exit code is always 0 (measurement, not a gate).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")

ROOT = Path(__file__).parent.parent
QUALITY_SET_PATH = ROOT / "eval" / "quality_set.json"

# The clarify gate stubs role-less obligation questions before retrieval, which
# would make both arms retrieve nothing and mask the graph delta. Force it off
# for the ablation (mirrors eval-clarify-gate-confound guidance).
os.environ.setdefault("CRSS_CLARIFY", "0")

_CASE_TIMEOUT_S = int(os.environ.get("CRSS_EVAL_CASE_TIMEOUT", "300"))


class _ArmTimeout(Exception):
    pass


def _on_alarm(signum, frame):  # noqa: ANN001
    raise _ArmTimeout()


def _cite_labels(must_cite: list) -> list[str]:
    """Render must_cite entries into the same label form check_answer uses."""
    return [
        " or ".join(req) if isinstance(req, list) else req
        for req in must_cite
    ]


def _run_arm(case: dict, retriever, *, k: int, graph_on: bool) -> dict:
    """Generate + key-check one arm (graph on/off) under a wall-clock timeout."""
    from application.agent import ask
    from scripts.check_answer_keys import check_answer

    os.environ["CRSS_GRAPH_EXPANSION"] = "1" if graph_on else "0"
    t0 = time.perf_counter()
    signal.alarm(_CASE_TIMEOUT_S)
    try:
        answer = ask(case["question"], retriever, k=k, history=case.get("history"))
    except Exception as exc:  # noqa: BLE001 — one bad arm must not kill the run
        return {"answer": "", "error": str(exc), "gen_s": round(time.perf_counter() - t0, 1)}
    finally:
        signal.alarm(0)

    check = check_answer(answer, case["answer_key"])
    return {
        "answer": answer,
        "gen_s": round(time.perf_counter() - t0, 1),
        "cite_recall": check["cite_recall"],
        "state_recall": check["state_recall"],
        "missed_cites": check["missed_cites"],
        "answer_chars": len(answer),
    }


def _retrieved_ref_blob(provisions: list[dict], definitions: list[dict]) -> str:
    """Concatenate every provision reference reachable in the retrieved bag.

    Mirrors what the context renderer can show the LLM: the anchor provision,
    its HAS_PART body (which includes parents/siblings), the CITES/INTERPRETS
    neighbours (already stripped in the flat arm), and the definitions channel.
    """
    parts: list[str] = []
    for p in provisions:
        parts.append(p.get("article_ref") or "")
        parts.append(p.get("article_path") or "")
        for c in p.get("children") or []:
            parts.append(c.get("ref") or "")
        for c in p.get("cited_provisions") or []:
            parts.append(c.get("ref") or "")
        for c in p.get("interpreting_guidance") or []:
            parts.append(c.get("ref") or "")
        for c in p.get("interpreted_provisions") or []:
            parts.append(c.get("ref") or "")
    for d in definitions:
        parts.append(d.get("article_ref") or "")
    return "\n".join(parts)


def _run_arm_retrieval(case: dict, retriever, *, k: int, graph_on: bool) -> dict:
    """Retrieval-only arm: no answer generation, no parametric-knowledge confound.

    The generation-mode metric asks "did the *answer* name the provision" — which
    a strong LLM can satisfy from parametric memory even when retrieval failed
    (observed: flat arm citing AI Act Article 25 that was never retrieved). This
    mode asks the confound-free question: is each ``must_cite`` provision present
    in the *retrieved context* at all? It mirrors the agent's retrieval phase
    (detection → route plan → sufficiency → corrective pass) and stops before
    the LLM. Only HyDE/decompose calls run (small model, ~1s), so the full set
    costs seconds per case.
    """
    from mistralai.client import Mistral
    from application.agent import (  # re-exported private symbols (see CLAUDE.md)
        _detect_scenario,
        _evaluate_route_sufficiency,
        _expand_definitions_from_provisions,
        _retrieve_route_provisions,
        _run_corrective_retrieval_pass,
    )
    from scripts.check_answer_keys import _cite_pattern

    os.environ["CRSS_GRAPH_EXPANSION"] = "1" if graph_on else "0"
    t0 = time.perf_counter()
    signal.alarm(_CASE_TIMEOUT_S)
    try:
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        det = _detect_scenario(case["question"], retriever, k)
        rr = _retrieve_route_provisions(
            case["question"], retriever,
            client=client, k=det.k, route=det.route,
            target_celexes=det.target_celexes,
            explicit_refs=det.explicit_refs,
            role_specs=det.role_specs,
            context_anchor_refs=det.context_anchor_refs,
        )
        provisions = rr["provisions"]
        definitions = _expand_definitions_from_provisions(
            provisions, retriever, det.definitions, target_celexes=det.target_celexes,
        )
        sufficiency = _evaluate_route_sufficiency(
            route=det.route, question=case["question"],
            explicit_refs=det.explicit_refs, target_celexes=det.target_celexes,
            role_specs=det.role_specs, provisions=provisions,
            definitions=definitions,
            direct_provisions=rr["direct_provisions"],
            role_provisions=rr["role_provisions"],
            legal_qualification_targets=rr["legal_qualification_targets"],
        )
        if not sufficiency["ok"]:
            _run_corrective_retrieval_pass(
                case["question"], retriever,
                client=client, k=det.k, route=det.route,
                target_celexes=det.target_celexes,
                explicit_refs=det.explicit_refs, role_specs=det.role_specs,
                provisions=provisions,
                direct_provisions=rr["direct_provisions"],
                role_provisions=rr["role_provisions"],
                definitions=definitions, sufficiency=sufficiency,
                hyde_text=rr["hyde_text"],
                legal_qualification_targets=rr["legal_qualification_targets"],
            )
    except Exception as exc:  # noqa: BLE001
        return {"answer": "", "error": str(exc), "gen_s": round(time.perf_counter() - t0, 1)}
    finally:
        signal.alarm(0)

    blob = _retrieved_ref_blob(provisions, definitions)
    found: list[str] = []
    missed: list[str] = []
    for req in case["answer_key"].get("must_cite", []):
        alts = req if isinstance(req, list) else [req]
        label = " or ".join(alts) if isinstance(req, list) else req
        (found if any(_cite_pattern(a).search(blob) for a in alts) else missed).append(label)
    n = len(found) + len(missed)
    return {
        "answer": "",
        "gen_s": round(time.perf_counter() - t0, 1),
        "cite_recall": round(len(found) / n, 3) if n else 1.0,
        "state_recall": None,
        "missed_cites": missed,
        "n_provisions": len(provisions),
        "answer_chars": 0,
    }


def _run_case(case: dict, retriever, *, k: int, retrieval_only: bool = False) -> dict:
    labels = _cite_labels(case["answer_key"].get("must_cite", []))
    arm = _run_arm_retrieval if retrieval_only else _run_arm
    full = arm(case, retriever, k=k, graph_on=True)
    flat = arm(case, retriever, k=k, graph_on=False)

    full_missed = set(full.get("missed_cites") or [])
    flat_missed = set(flat.get("missed_cites") or [])
    # Cites the graph surfaced that flat retrieval dropped (missed in flat,
    # found in full) — the decisive multi-hop/cross-reference provisions.
    recovered = sorted(flat_missed - full_missed)
    # Cites flat had but the graph lost — should be ~empty; >0 warrants a look.
    lost = sorted(full_missed - flat_missed)

    return {
        "id": case["id"],
        "label": case.get("label", ""),
        "n_must_cite": len(labels),
        "full": full,
        "flat": flat,
        "cite_recall_full": full.get("cite_recall"),
        "cite_recall_flat": flat.get("cite_recall"),
        "cite_recall_delta": (
            round(full["cite_recall"] - flat["cite_recall"], 3)
            if full.get("cite_recall") is not None and flat.get("cite_recall") is not None
            else None
        ),
        "recovered_by_graph": recovered,
        "lost_by_graph": lost,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="CRSS graph-ablation eval (flat vs full, deterministic keys)")
    ap.add_argument("--case", nargs="+", metavar="ID", help="Run only these case IDs")
    ap.add_argument("--k", type=int, default=20, help="Retrieval budget k (default: 20)")
    ap.add_argument("--limit", type=int, help="Cap the number of keyed cases run")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="Check must_cite presence in the RETRIEVED CONTEXT instead of "
                         "the generated answer — no generation, no parametric-knowledge "
                         "confound; the whole set runs in minutes")
    ap.add_argument("--out", help="Write full JSON results to this path")
    args = ap.parse_args()

    # Default: archive runs under eval/runs/. A bare --out filename (no directory)
    # resolves there; an explicit path (with a directory) is used as-is.
    if args.out and os.path.dirname(args.out) == "":
        (ROOT / "eval" / "runs").mkdir(parents=True, exist_ok=True)
        args.out = str(ROOT / "eval" / "runs" / args.out)

    cases: list[dict] = json.loads(QUALITY_SET_PATH.read_text())
    # Only cases with a non-empty must_cite key can yield a cite-recall delta.
    keyed = [c for c in cases if (c.get("answer_key") or {}).get("must_cite")]
    if args.case:
        ids = set(args.case)
        keyed = [c for c in keyed if c["id"] in ids]
    if args.limit:
        keyed = keyed[: args.limit]
    if not keyed:
        print("No keyed cases matched (need answer_key.must_cite).", file=sys.stderr)
        return 2

    mode = "retrieved-context cite coverage (no generation)" if args.retrieval_only \
        else "answer-level cite recall (2× generation/case)"
    print(f"\nCRSS Graph-Ablation Eval — {len(keyed)} keyed case(s), k={args.k}, mode: {mode}")
    print("full = GraphRAG (all channels)   flat = dense+lexical vector only")
    print("Connecting to Neo4j and loading embeddings …")
    from retrieval.graph_retriever import GraphRetriever
    retriever = GraphRetriever()
    signal.signal(signal.SIGALRM, _on_alarm)

    results = []
    for case in keyed:
        r = _run_case(case, retriever, k=args.k, retrieval_only=args.retrieval_only)
        results.append(r)
        if args.out:  # kill-safe partial flush
            Path(args.out).write_text(json.dumps({"partial": True, "results": results}, indent=2))
        rf, rl = r["cite_recall_full"], r["cite_recall_flat"]
        rf_s = f"{rf:.0%}" if rf is not None else "  ERR"
        rl_s = f"{rl:.0%}" if rl is not None else "  ERR"
        d_s = f"{r['cite_recall_delta']:+.0%}" if r["cite_recall_delta"] is not None else "  n/a"
        extra = ""
        if r["recovered_by_graph"]:
            extra += f"   graph-only cites: {r['recovered_by_graph']}"
        if r["lost_by_graph"]:
            extra += f"   ⚠ lost by graph: {r['lost_by_graph']}"
        print(f"  {r['id']:<8} cite-recall  full={rf_s:>5}  flat={rl_s:>5}  Δ={d_s:>5}{extra}")

    retriever.close()

    scored = [r for r in results if r["cite_recall_delta"] is not None]
    mean_full = (
        round(sum(r["cite_recall_full"] for r in scored) / len(scored), 3) if scored else None
    )
    mean_flat = (
        round(sum(r["cite_recall_flat"] for r in scored) / len(scored), 3) if scored else None
    )
    total_recovered = sum(len(r["recovered_by_graph"]) for r in results)
    total_lost = sum(len(r["lost_by_graph"]) for r in results)
    cases_with_recovery = sum(1 for r in results if r["recovered_by_graph"])
    improved = sum(1 for r in scored if r["cite_recall_delta"] > 0)
    unchanged = sum(1 for r in scored if r["cite_recall_delta"] == 0)
    regressed = sum(1 for r in scored if r["cite_recall_delta"] < 0)

    print(f"\n{'─' * 64}")
    if mean_full is not None:
        print(f"  Mean cite-recall:   full {mean_full:.0%}   flat {mean_flat:.0%}   "
              f"Δ {mean_full - mean_flat:+.0%}   ← the graph's contribution")
    print(f"  Decisive cites recovered only by the graph: {total_recovered} "
          f"across {cases_with_recovery}/{len(results)} case(s)  ← headline")
    print(f"  Cases improved by graph: {improved}   unchanged: {unchanged}   regressed: {regressed}")
    if total_lost:
        print(f"  ⚠ Cites the graph DROPPED vs flat: {total_lost}  (investigate — graph should not lose cites)")
    print(f"{'─' * 64}\n")

    summary = {
        "n_cases": len(results),
        "k": args.k,
        "mode": "retrieval_only" if args.retrieval_only else "generation",
        "mean_cite_recall_full": mean_full,
        "mean_cite_recall_flat": mean_flat,
        "mean_cite_recall_delta": (
            round(mean_full - mean_flat, 3) if mean_full is not None else None
        ),
        "cites_recovered_only_by_graph": total_recovered,
        "cites_lost_by_graph": total_lost,
        "cases_improved": improved,
        "cases_unchanged": unchanged,
        "cases_regressed": regressed,
    }
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "results": results}, indent=2))
        print(f"Wrote {args.out}")

    # Leave the env flag as production-default for anything that imports after us.
    os.environ["CRSS_GRAPH_EXPANSION"] = "1"
    return 0


if __name__ == "__main__":
    sys.exit(main())
