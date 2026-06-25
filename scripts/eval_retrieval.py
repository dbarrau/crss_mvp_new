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
    python scripts/eval_retrieval.py --snapshot base.json   # freeze a baseline
    python scripts/eval_retrieval.py --diff base.json       # drift vs baseline

Exit code: 0 if all pass, 1 if any fail.

For the read-path rewrite, capture a baseline on ``main`` with ``--snapshot``,
then run ``--diff`` on the rewrite branch to see exactly which provisions each
migration step adds or drops per case (see docs/rewrite/REGRESSION_NET.md).

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
    _retrieve_route_provisions,
    _expand_definitions_from_provisions,
    _evaluate_route_sufficiency,
    _run_corrective_retrieval_pass,
)
from application.scenario import detect_scenario
from retrieval.graph_retriever import GraphRetriever


GOLDEN_SET_PATH = Path(__file__).parent.parent / "eval" / "golden_set.json"

# ANSI colours (stripped when not a tty)
_USE_COLOUR = sys.stdout.isatty()
_GREEN  = "\033[92m" if _USE_COLOUR else ""
_RED    = "\033[91m" if _USE_COLOUR else ""
_YELLOW = "\033[93m" if _USE_COLOUR else ""
_BOLD   = "\033[1m"  if _USE_COLOUR else ""
_RESET  = "\033[0m"  if _USE_COLOUR else ""


def _provision_identity(p: dict) -> str:
    """Stable identity for a retrieved provision, for baseline/diff comparison.

    Prefers the canonical node id (``<celex>_<kind>_<ref>``); falls back to
    ``celex|article_ref`` only when no id is present.  ``article_ref`` /
    ``display_ref`` alone is *not* used as identity — annex sub-node
    display_refs are non-unique, which would corrupt a diff.
    """
    pid = p.get("article_id") or p.get("id")
    if pid:
        return str(pid)
    celex = p.get("celex", "?")
    ref = p.get("article_ref") or p.get("ref") or "?"
    return f"{celex}|{ref}"


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

    # Detection — the exact deterministic stage the agent runs, driven through
    # the shared scenario.py entry point so this net gates it (no LLM). The
    # returned k carries the per-regulation budget bump.
    det = detect_scenario(question, retriever, k)
    definitions = det.definitions
    target_celexes = det.target_celexes
    role_specs = det.role_specs
    explicit_refs = det.explicit_refs
    route = det.route
    k = det.k

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

    # Mirror ask_stream's post-retrieval pipeline: expand definitions, evaluate
    # route sufficiency, and run the bounded corrective pass when coverage is
    # insufficient. Every call here is deterministic under the LLM stubs
    # (sufficiency/definitions use no LLM; the corrective pass's only LLM hook is
    # HyDE, stubbed via _stub_hyde), so the net stays a deterministic gate while
    # now covering the sufficiency + corrective stages the agent actually runs.
    provisions = retrieval_result["provisions"]
    direct_provisions = retrieval_result["direct_provisions"]
    role_provisions = retrieval_result["role_provisions"]
    legal_qualification_targets = retrieval_result["legal_qualification_targets"]
    definitions = _expand_definitions_from_provisions(
        provisions, retriever, definitions, target_celexes=target_celexes,
    )
    sufficiency = _evaluate_route_sufficiency(
        route=route,
        question=question,
        explicit_refs=explicit_refs,
        target_celexes=target_celexes,
        role_specs=role_specs,
        provisions=provisions,
        definitions=definitions,
        direct_provisions=direct_provisions,
        role_provisions=role_provisions,
        legal_qualification_targets=legal_qualification_targets,
    )
    corrective_actions: list[str] = []
    if not sufficiency["ok"]:
        recovery = _run_corrective_retrieval_pass(
            question,
            retriever,
            client=None,
            k=k,
            route=route,
            target_celexes=target_celexes,
            explicit_refs=explicit_refs,
            role_specs=role_specs,
            provisions=provisions,
            direct_provisions=direct_provisions,
            role_provisions=role_provisions,
            definitions=definitions,
            sufficiency=sufficiency,
            hyde_text=retrieval_result["hyde_text"],
            legal_qualification_targets=legal_qualification_targets,
            hyde_builder=_stub_hyde,
        )
        corrective_actions = recovery["actions"]

    elapsed = time.perf_counter() - t0

    # Collect all retrieved provisions from every bucket (post-corrective: the
    # corrective pass mutates these lists in place)
    all_provisions: list[dict] = []
    for bucket in (provisions, direct_provisions, role_provisions):
        all_provisions.extend(bucket or [])

    retrieved_celexes: set[str] = {
        p.get("celex", "") for p in all_provisions if p.get("celex")
    }
    retrieved_refs: list[str] = [
        p.get("article_ref", "") for p in all_provisions if p.get("article_ref")
    ]
    retrieved_ids: list[str] = sorted({_provision_identity(p) for p in all_provisions})

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
        "retrieved_ids": retrieved_ids,
        "n_provisions": len(all_provisions),
        "corrective_actions": corrective_actions,
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

    if result.get("corrective_actions"):
        print(f"        {_YELLOW}Corrective:{_RESET}    {'; '.join(result['corrective_actions'])}")

    if verbose:
        regs = ", ".join(result["retrieved_celexes"]) or "(none)"
        refs = ", ".join(result["retrieved_refs"][:15]) or "(none)"
        print(f"        Retrieved CELEX: {regs}")
        print(f"        Retrieved refs:  {refs}")


def _write_snapshot(results: list[dict], path: str) -> None:
    """Persist per-case retrieval as a baseline for later --diff comparisons."""
    snapshot = {
        r["id"]: {
            "route": r["actual_route"],
            "celexes": r["retrieved_celexes"],
            "retrieved_ids": r["retrieved_ids"],
        }
        for r in results
    }
    Path(path).write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    print(f"\n{_BOLD}Snapshot written:{_RESET} {path} ({len(snapshot)} case(s))")


def _print_diff(results: list[dict], baseline_path: str) -> int:
    """Report per-case retrieval drift vs a baseline snapshot.

    Returns the number of cases that *dropped* a previously-retrieved provision
    or changed route — the signals that must be reviewed before flipping a
    migration step from old to new path.
    """
    baseline: dict[str, Any] = json.loads(Path(baseline_path).read_text())
    print(f"\n{_BOLD}Retrieval diff vs {baseline_path}{_RESET}\n")

    total_added = total_dropped = 0
    regressed: list[str] = []
    for r in results:
        base = baseline.get(r["id"])
        if base is None:
            print(f"  {_YELLOW}NEW{_RESET}   {_BOLD}{r['id']}{_RESET}  (not in baseline)")
            continue
        cur_ids = set(r["retrieved_ids"])
        base_ids = set(base.get("retrieved_ids", []))
        added = sorted(cur_ids - base_ids)
        dropped = sorted(base_ids - cur_ids)
        route_changed = base.get("route") != r["actual_route"]
        total_added += len(added)
        total_dropped += len(dropped)
        if dropped or route_changed:
            regressed.append(r["id"])
        if not (added or dropped or route_changed):
            continue
        marker = f"{_RED}±{_RESET}" if (dropped or route_changed) else f"{_GREEN}+{_RESET}"
        print(f"  {marker}  {_BOLD}{r['id']}{_RESET}  {r['label']}")
        if route_changed:
            print(f"        {_YELLOW}route:{_RESET} {base.get('route')} → {r['actual_route']}")
        if dropped:
            print(f"        {_RED}dropped ({len(dropped)}):{_RESET} {dropped[:10]}")
        if added:
            print(f"        {_GREEN}added   ({len(added)}):{_RESET} {added[:10]}")

    print(f"\n{'─' * 60}")
    print(f"  Added provisions:   {_GREEN}+{total_added}{_RESET}")
    print(f"  Dropped provisions: {_RED}-{total_dropped}{_RESET}")
    print(f"  Cases regressed:    {_RED if regressed else _GREEN}{len(regressed)}{_RESET}"
          + (f"  {regressed}" if regressed else ""))
    print(f"{'─' * 60}\n")
    return len(regressed)


def main() -> int:
    parser = argparse.ArgumentParser(description="CRSS retrieval eval harness")
    parser.add_argument(
        "--case", nargs="+", metavar="ID",
        help="Run only these test case IDs (e.g. TC_001 TC_009)",
    )
    parser.add_argument("--k", type=int, default=8, help="Retrieval budget k (default: 8)")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved refs per case")
    parser.add_argument(
        "--snapshot", metavar="PATH",
        help="Write per-case retrieved provision IDs to PATH as a baseline",
    )
    parser.add_argument(
        "--diff", metavar="PATH",
        help="Compare this run's retrieval against a baseline snapshot at PATH",
    )
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

    if args.snapshot:
        _write_snapshot(results, args.snapshot)
    if args.diff:
        _print_diff(results, args.diff)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
