#!/usr/bin/env python3
"""Answer-quality eval — LLM-judge grading of CRSS answers against the senior
compliance-officer rubric.

Unlike ``eval_retrieval.py`` (which measures whether the right provisions are
*retrieved*), this harness measures whether the *generated answer* is reliable
enough for a senior compliance officer to use. It makes real LLM calls: one (or
more) to generate the answer via ``application.agent.ask`` and one to grade it.

The rubric lives in ``eval/rubric_prompt.txt`` so it can be iterated on without
touching code. Questions live in ``eval/quality_set.json``.

Usage
-----
    python scripts/eval_answer_quality.py                       # all cases
    python scripts/eval_answer_quality.py --case HQ_001         # single case
    python scripts/eval_answer_quality.py --case HQ_001 --out /tmp/q.json
    python scripts/eval_answer_quality.py --answer-file ans.md --case HQ_001
        # grade a pre-generated answer instead of regenerating

The judge model defaults to ``mistral-large-latest`` and can be overridden with
``CRSS_JUDGE_MODEL`` or ``--judge-model``. The judge runs at temperature 0; the
generation temperature is whatever the agent code uses.

Exit code: 0 always (this is a measurement, not a pass/fail gate). Inspect the
scores / JSON output.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")

ROOT = Path(__file__).parent.parent
RUBRIC_PATH = ROOT / "eval" / "rubric_prompt.txt"
QUALITY_SET_PATH = ROOT / "eval" / "quality_set.json"

_RELIANCE_PHRASES = [
    "Can be relied on with minimal revision",
    "Can be used as a strong first draft",
    "Cannot be relied on without major revision",
    "Unsafe for compliance reliance",
]


def _parse_score(text: str) -> float | None:
    """Extract the numeric final score from the judge's response."""
    m = re.search(
        r"final\s*score\s*[:\-]?\s*\**\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
        text,
        re.I,
    )
    if m:
        return float(m.group(1))
    # Fallback: first "X/10" anywhere
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10", text)
    return float(m.group(1)) if m else None


def _parse_reliance(text: str) -> str:
    for phrase in _RELIANCE_PHRASES:
        if phrase.lower() in text.lower():
            return phrase
    return "(unparsed)"


def _judge(prompt: str, model: str, runs: int) -> tuple[float | None, list[float], str]:
    """Run the judge `runs` times; return (mean_score, all_scores, last_text)."""
    from mistralai.client import Mistral

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    scores: list[float] = []
    last_text = ""
    for _ in range(runs):
        resp = client.chat.complete(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        last_text = resp.choices[0].message.content or ""
        s = _parse_score(last_text)
        if s is not None:
            scores.append(s)
    mean = round(sum(scores) / len(scores), 2) if scores else None
    return mean, scores, last_text


def _run_case(
    case: dict,
    retriever,
    rubric: str,
    *,
    k: int,
    judge_model: str,
    judge_runs: int,
    answer_override: str | None,
) -> dict:
    from application.agent import ask

    question = case["question"]
    t0 = time.perf_counter()

    if answer_override is not None:
        answer = answer_override
        gen_s = 0.0
    else:
        answer = ask(question, retriever, k=k)
        gen_s = time.perf_counter() - t0

    prompt = f"{rubric}\n\n## QUESTION\n{question}\n\n## CRSS ANSWER\n{answer}\n"

    tj = time.perf_counter()
    mean, scores, judge_text = _judge(prompt, judge_model, judge_runs)
    judge_s = time.perf_counter() - tj

    return {
        "id": case["id"],
        "label": case.get("label", ""),
        "score": mean,
        "scores": scores,
        "reliance": _parse_reliance(judge_text),
        "gen_s": round(gen_s, 1),
        "judge_s": round(judge_s, 1),
        "answer_chars": len(answer),
        "answer": answer,
        "judge_text": judge_text,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="CRSS answer-quality eval (LLM judge)")
    ap.add_argument("--case", nargs="+", metavar="ID", help="Run only these case IDs")
    ap.add_argument("--k", type=int, default=20, help="Retrieval budget k (default: 20)")
    ap.add_argument("--judge-model", default=os.environ.get("CRSS_JUDGE_MODEL", "mistral-large-latest"))
    ap.add_argument("--judge-runs", type=int, default=1, help="Judge calls per case, averaged")
    ap.add_argument("--answer-file", help="Grade this answer file instead of regenerating (single case only)")
    ap.add_argument("--out", help="Write full JSON results to this path")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-case judge text")
    ap.add_argument("--label", default="", help="Free-text tag stored in output (e.g. commit sha)")
    args = ap.parse_args()

    rubric = RUBRIC_PATH.read_text()
    cases: list[dict] = json.loads(QUALITY_SET_PATH.read_text())
    if args.case:
        ids = set(args.case)
        cases = [c for c in cases if c["id"] in ids]
        if not cases:
            print(f"No cases matched: {args.case}", file=sys.stderr)
            return 2

    answer_override = None
    if args.answer_file:
        if len(cases) != 1:
            print("--answer-file requires exactly one --case", file=sys.stderr)
            return 2
        answer_override = Path(args.answer_file).read_text()

    if not args.quiet:
        print(f"\nCRSS Answer-Quality Eval — {len(cases)} case(s), judge={args.judge_model}, label={args.label or '(none)'}")
        print("Connecting to Neo4j and loading embeddings …")
    from retrieval.graph_retriever import GraphRetriever
    retriever = GraphRetriever()

    results = []
    for case in cases:
        r = _run_case(
            case, retriever, rubric,
            k=args.k, judge_model=args.judge_model, judge_runs=args.judge_runs,
            answer_override=answer_override,
        )
        results.append(r)
        score_str = f"{r['score']}" if r["score"] is not None else "PARSE_FAIL"
        print(f"  {r['id']}  score={score_str}/10  [{r['reliance']}]  "
              f"(gen {r['gen_s']}s, judge {r['judge_s']}s)")
        if not args.quiet:
            print("\n" + "─" * 70)
            print(r["judge_text"])
            print("─" * 70 + "\n")

    retriever.close()

    out = {"label": args.label, "judge_model": args.judge_model, "results": results}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        if not args.quiet:
            print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
