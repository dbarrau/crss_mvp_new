#!/usr/bin/env python3
"""Retrieval eval harness — measures recall against the golden test set.

Runs the full routing + retrieval pipeline (no LLM calls) and checks whether
each test case's expected regulations and provisions appear in the retrieved bag.

Usage
-----
    python scripts/eval_retrieval.py                    # all cases
    python scripts/eval_retrieval.py --case TC_001      # single case
    python scripts/eval_retrieval.py --case TC_001 TC_009 TC_012
    python scripts/eval_retrieval.py --k 10            # wider retrieval budget
    python scripts/eval_retrieval.py --verbose          # show retrieved refs

Exit code: 0 if all pass, 1 if any fail.

Notes
-----
HyDE (Hypothetical Document Embeddings) is stubbed — the question itself is
used as the hypothetical passage — so this script makes zero LLM calls and
runs against live Neo4j only.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

logging.basicConfig(
    level=logging.WARNING,  # suppress retriever/agent INFO noise during eval
    format="%(levelname)-8s %(message)s",
)

from application.agent import (
    _detect_mentioned_regulations,
    _extract_provision_refs,
    _extract_implicit_provision_refs,
    _is_definition_question,
    _select_question_route,
    _detect_question_roles,
    _detect_defined_terms,
    _REG_NAME_TO_CELEX,
    _retrieve_route_provisions,
)
from retrieval.graph_retriever import GraphRetriever


GOLDEN_SET_PATH = Path(__file__).parent.parent / "eval" / "golden_set.json"

# ANSI colours (stripped when not a tty)
_USE_COLOUR = sys.stdout.isatty()
_GREEN  = "\033[92m" if _USE_COLOUR else ""
_RED    = "\033[91m" if _USE_COLOUR else ""
_YELLOW = "\033[93m" if _USE_COLOUR else ""
_BOLD   = "\033[1m"  if _USE_COLOUR else ""
_RESET  = "\033[0m"  if _USE_COLOUR else ""


def _stub_hyde(question: str, _client: Any) -> str:
    """HyDE stub: returns the question as the hypothetical passage.

    Avoids any LLM call while still providing a text vector for classification_chain
    routes that require a hyde_vec. Quality is lower than real HyDE but sufficient
    to check that the routing and retrieval machinery works correctly.
    """
    return question


def _run_case(
    case: dict,
    retriever: GraphRetriever,
    *,
    k: int,
    verbose: bool,
) -> dict:
    """Execute one golden test case and return a result dict."""
    question = case["question"]
    expected_celexes: list[str] = case.get("expected_celexes", [])
    must_contain_refs: list[str] = case.get("must_contain_refs", [])
    expected_route: str = case.get("expected_route", "")

    t0 = time.perf_counter()

    # Mirror the pipeline in ask_stream (no LLM involvement)
    keyword_mentioned_regs = _detect_mentioned_regulations(question)
    mentioned_regs = set(keyword_mentioned_regs)

    # Run defined-term detection (no LLM) so that implicit regulation links
    # are captured — e.g. "medical device" in a question implicitly references
    # MDR even when "MDR" is not spelled out.  Mirrors the step in ask_stream
    # that adds regs discovered via definitions to mentioned_regs.
    definitions = _detect_defined_terms(question, retriever)
    for d in definitions:
        reg = d.get("regulation", "")
        if reg and reg not in mentioned_regs and reg in _REG_NAME_TO_CELEX:
            mentioned_regs.add(reg)

    target_celexes: set[str] | None = None
    if mentioned_regs:
        target_celexes = {
            _REG_NAME_TO_CELEX[r]
            for r in mentioned_regs
            if r in _REG_NAME_TO_CELEX
        }
        if len(mentioned_regs) > 1:
            has_guidance = any(
                _REG_NAME_TO_CELEX.get(r, "").startswith("MDCG_")
                for r in mentioned_regs
            )
            per_reg = 4 if has_guidance else 3
            k = max(k, len(mentioned_regs) * per_reg)

    role_specs = _detect_question_roles(question, target_celexes=target_celexes)
    explicit_refs = _extract_provision_refs(question)
    for _ref in _extract_implicit_provision_refs(question, target_celexes=target_celexes):
        if _ref not in explicit_refs:
            explicit_refs.append(_ref)
    is_def_q, _ = _is_definition_question(question)
    route = _select_question_route(
        question,
        explicit_refs=explicit_refs,
        mentioned_regs=mentioned_regs,
        keyword_mentioned_regs=keyword_mentioned_regs,
        role_specs=role_specs,
        is_definition_question=is_def_q,
    )

    retrieval_result = _retrieve_route_provisions(
        question,
        retriever,
        client=None,
        k=k,
        route=route,
        target_celexes=target_celexes,
        explicit_refs=explicit_refs,
        role_specs=role_specs,
        hyde_builder=_stub_hyde,
    )

    elapsed = time.perf_counter() - t0

    # Collect all retrieved provisions from every bucket
    all_provisions: list[dict] = []
    for bucket in ("provisions", "direct_provisions", "role_provisions",
                   "backbone_provisions"):
        all_provisions.extend(retrieval_result.get(bucket) or [])

    retrieved_celexes: set[str] = {
        p.get("celex", "") for p in all_provisions if p.get("celex")
    }
    retrieved_refs: list[str] = [
        p.get("article_ref", "") for p in all_provisions if p.get("article_ref")
    ]

    # --- Checks ---
    missing_celexes = [c for c in expected_celexes if c not in retrieved_celexes]
    missing_refs: list[str] = []
    for ref in must_contain_refs:
        ref_lower = ref.lower()
        if not any(ref_lower in r.lower() for r in retrieved_refs):
            missing_refs.append(ref)

    route_match = (not expected_route) or (route.id == expected_route)

    passed = not missing_celexes and not missing_refs

    return {
        "id": case["id"],
        "label": case.get("label", ""),
        "passed": passed,
        "route_match": route_match,
        "actual_route": route.id,
        "expected_route": expected_route,
        "missing_celexes": missing_celexes,
        "missing_refs": missing_refs,
        "retrieved_celexes": sorted(retrieved_celexes),
        "retrieved_refs": list(dict.fromkeys(retrieved_refs)),  # deduped, ordered
        "n_provisions": len(all_provisions),
        "elapsed_s": elapsed,
    }


def _print_result(result: dict, verbose: bool) -> None:
    status = f"{_GREEN}PASS{_RESET}" if result["passed"] else f"{_RED}FAIL{_RESET}"
    route_flag = (
        "" if result["route_match"]
        else f" {_YELLOW}[ROUTE: got {result['actual_route']!r}, want {result['expected_route']!r}]{_RESET}"
    )
    print(
        f"  {status}  {_BOLD}{result['id']}{_RESET}  {result['label']}"
        f"  ({result['n_provisions']} provisions, {result['elapsed_s']:.2f}s){route_flag}"
    )

    if not result["passed"]:
        if result["missing_celexes"]:
            print(f"        {_RED}Missing CELEX:{_RESET} {result['missing_celexes']}")
        if result["missing_refs"]:
            print(f"        {_RED}Missing refs:{_RESET}  {result['missing_refs']}")

    if verbose:
        regs = ", ".join(result["retrieved_celexes"]) or "(none)"
        refs = ", ".join(result["retrieved_refs"][:15]) or "(none)"
        print(f"        Retrieved CELEX: {regs}")
        print(f"        Retrieved refs:  {refs}")


def main() -> int:
    parser = argparse.ArgumentParser(description="CRSS retrieval eval harness")
    parser.add_argument(
        "--case", nargs="+", metavar="ID",
        help="Run only these test case IDs (e.g. TC_001 TC_009)",
    )
    parser.add_argument("--k", type=int, default=8, help="Retrieval budget k (default: 8)")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved refs per case")
    args = parser.parse_args()

    golden: list[dict] = json.loads(GOLDEN_SET_PATH.read_text())

    if args.case:
        ids = set(args.case)
        golden = [c for c in golden if c["id"] in ids]
        if not golden:
            print(f"No cases matched: {args.case}", file=sys.stderr)
            return 2

    print(f"\n{_BOLD}CRSS Retrieval Eval — {len(golden)} case(s), k={args.k}{_RESET}\n")
    print("Connecting to Neo4j and loading embeddings …")
    retriever = GraphRetriever()
    print(f"Index loaded ({retriever._matrix.shape if retriever._matrix is not None else 'empty'})\n")

    results: list[dict] = []
    for case in golden:
        r = _run_case(case, retriever, k=args.k, verbose=args.verbose)
        _print_result(r, verbose=args.verbose)
        results.append(r)

    retriever.close()

    # Summary
    passed = sum(1 for r in results if r["passed"])
    route_match = sum(1 for r in results if r["route_match"])
    total = len(results)
    avg_t = sum(r["elapsed_s"] for r in results) / total if total else 0

    print(f"\n{'─' * 60}")
    print(f"  Recall (provisions):  {_GREEN if passed == total else _RED}{passed}/{total}{_RESET} passed")
    print(f"  Route accuracy:       {route_match}/{total} correct")
    print(f"  Avg retrieval time:   {avg_t:.2f}s")
    print(f"{'─' * 60}\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
