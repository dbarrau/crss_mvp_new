#!/usr/bin/env python3
"""Evaluate role-aware retrieval against a small gold set.

This checks the graph-backed role retrieval path directly, without LLM answer
generation. It is intended as a graph-hardening benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.ontology.actor_roles import detect_role_specs
from retrieval.graph_retriever import GraphRetriever


GOLD_CASES = [
    {
        "name": "Hospital deployment question",
        "question": "What must a hospital verify before putting a high-risk AI medical device into service?",
        "expected_any": {"Article 26", "Article 50", "Article 4"},
    },
    {
        "name": "Deployer obligations",
        "question": "What are the obligations of a deployer under the AI Act?",
        "expected_any": {"Article 26"},
    },
    {
        "name": "Importer obligations",
        "question": "What are the obligations of an importer under the AI Act?",
        "expected_any": {"Article 23"},
    },
    {
        "name": "Distributor obligations",
        "question": "What are the obligations of a distributor under the AI Act?",
        "expected_any": {"Article 24"},
    },
    {
        "name": "Operator umbrella role",
        "question": "What are the obligations of an operator under the AI Act?",
        "expected_any": {"Article 23", "Article 24", "Article 26"},
    },
]


def main() -> None:
    passed = 0
    print("=== Actor-Role Retrieval Gold Set ===")
    with GraphRetriever() as retriever:
        for case in GOLD_CASES:
            role_specs = detect_role_specs(case["question"], target_celexes={"32024R1689"})
            results = retriever.retrieve_by_roles(role_specs, k=10)
            refs = [row.get("article_ref") for row in results]
            matched = sorted(case["expected_any"] & set(refs))
            ok = bool(matched)
            if ok:
                passed += 1
            print(f"\n[{ 'PASS' if ok else 'FAIL' }] {case['name']}")
            print(f"  question: {case['question']}")
            print(f"  role_specs: {role_specs}")
            print(f"  expected_any: {sorted(case['expected_any'])}")
            print(f"  matched: {matched if matched else 'none'}")
            print(f"  top_refs: {refs[:8]}")

    print(f"\nSummary: {passed}/{len(GOLD_CASES)} cases passed")


if __name__ == "__main__":
    main()
