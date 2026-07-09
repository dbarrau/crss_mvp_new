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


# Faithfulness / attribution flag blocks are prepended to the answer by the
# agent (see application/_faithfulness.build_warning_block). Parsing them counts
# how much unfaithful content this run produced and had to strip — a first-class
# regression signal: a migration that raises fabricated/misattributed counts
# must fail the net even if the rubric score holds. Requires the faithfulness
# check enabled (CRSS_FAITHFULNESS_CHECK != 0, the default).
_FABRICATED_RE = re.compile(r"FAITHFULNESS FLAG\*\*\s*[—\-]\s*(\d+)\s+of\s+(\d+)", re.I)
_MISATTRIBUTED_RE = re.compile(r"ATTRIBUTION FLAG\*\*\s*[—\-]\s*(\d+)\s+of\s+(\d+)", re.I)
_NEAR_RE = re.compile(r"Wording check\*\*\s*[—\-]\s*(\d+)\s+quote", re.I)


# Hard per-case wall-clock timeout. A Mistral stream that stalls mid-response
# blocks forever (no read timeout in the SDK path), which hangs the whole run —
# observed: one case stuck 69 min with leaked sockets. SIGALRM abandons a stuck
# case, records it, and lets the run finish in bounded time. Override with
# CRSS_EVAL_CASE_TIMEOUT (seconds).
_CASE_TIMEOUT_S = int(os.environ.get("CRSS_EVAL_CASE_TIMEOUT", "300"))
_timed_out_flag = {"v": False}

# A full 32-case run hammers the Mistral API back-to-back; the tail cases
# accumulate rate-limit backoff and a case that finishes in ~70-100s in isolation
# can stall past the timeout (or return empty on a transient 5xx). Those are
# harness-throughput artifacts, not answer-quality failures — so retry a
# failed/empty/timed-out case once after a cooldown that lets the backoff clear.
_MAX_ATTEMPTS = int(os.environ.get("CRSS_EVAL_ATTEMPTS", "2"))
_RETRY_PAUSE_S = int(os.environ.get("CRSS_EVAL_RETRY_PAUSE", "45"))


class _CaseTimeout(Exception):
    pass


def _on_alarm(signum, frame):  # noqa: ANN001
    _timed_out_flag["v"] = True
    raise _CaseTimeout()


def _timeout_result(case: dict) -> dict:
    return {
        "id": case["id"],
        "label": case.get("label", ""),
        "score": None,
        "scores": [],
        "reliance": "(timeout)",
        "faithfulness": {"fabricated": 0, "misattributed": 0, "near_verbatim": 0, "total_quotes": 0},
        "gen_s": _CASE_TIMEOUT_S,
        "judge_s": 0.0,
        "answer_chars": 0,
        "answer": "",
        "judge_text": f"TIMEOUT after {_CASE_TIMEOUT_S}s",
        "timed_out": True,
    }


def _parse_faithfulness(answer: str) -> dict[str, int]:
    """Extract fabricated / misattributed / near-verbatim quote counts."""
    fab = _FABRICATED_RE.search(answer)
    mis = _MISATTRIBUTED_RE.search(answer)
    near = _NEAR_RE.search(answer)
    total = 0
    if fab:
        total = max(total, int(fab.group(2)))
    if mis:
        total = max(total, int(mis.group(2)))
    return {
        "fabricated": int(fab.group(1)) if fab else 0,
        "misattributed": int(mis.group(1)) if mis else 0,
        "near_verbatim": int(near.group(1)) if near else 0,
        "total_quotes": total,
    }


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
        "faithfulness": _parse_faithfulness(answer),
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

    signal.signal(signal.SIGALRM, _on_alarm)

    def _flush_partial() -> None:
        # Kill-safe: persist after every case so a hang/kill never loses the run.
        if args.out:
            Path(args.out).write_text(json.dumps(
                {"label": args.label, "judge_model": args.judge_model,
                 "partial": True, "results": results}, indent=2))

    def _attempt_case(case: dict) -> dict:
        """One attempt at a case under the wall-clock timeout."""
        _timed_out_flag["v"] = False
        signal.alarm(_CASE_TIMEOUT_S)
        try:
            return _run_case(
                case, retriever, rubric,
                k=args.k, judge_model=args.judge_model, judge_runs=args.judge_runs,
                answer_override=answer_override,
            )
        except Exception as exc:  # noqa: BLE001 — one bad case must not kill the run
            r = _timeout_result(case)
            if not _timed_out_flag["v"]:
                r["reliance"] = "(error)"
                r["judge_text"] = f"ERROR: {exc}"
            return r
        finally:
            signal.alarm(0)

    def _case_ok(r: dict) -> bool:
        # A real result has a graded score AND a non-empty answer. A timeout /
        # transient error / empty answer is a throughput artifact worth retrying.
        return r.get("score") is not None and r.get("answer_chars", 0) > 0

    results = []
    for case in cases:
        r = _attempt_case(case)
        attempts_used = 1
        while not _case_ok(r) and attempts_used < _MAX_ATTEMPTS:
            print(f"  {case['id']}  {r['reliance'].upper()} on attempt {attempts_used} "
                  f"— retrying after {_RETRY_PAUSE_S}s cooldown "
                  f"(likely API backoff, not a quality failure)", flush=True)
            time.sleep(_RETRY_PAUSE_S)
            r = _attempt_case(case)
            attempts_used += 1
        r["attempts"] = attempts_used
        if not _case_ok(r):
            print(f"  {case['id']}  {r['reliance'].upper()} — {r['judge_text'][:80]} "
                  f"(after {attempts_used} attempt(s))", flush=True)
            results.append(r)
            _flush_partial()
            continue
        results.append(r)
        _flush_partial()
        score_str = f"{r['score']}" if r["score"] is not None else "PARSE_FAIL"
        fc = r["faithfulness"]
        faith_str = (
            f"fab={fc['fabricated']} mis={fc['misattributed']} "
            f"near={fc['near_verbatim']}/{fc['total_quotes']}"
        )
        print(f"  {r['id']}  score={score_str}/10  [{r['reliance']}]  {faith_str}  "
              f"(gen {r['gen_s']}s, judge {r['judge_s']}s)")
        if not args.quiet:
            print("\n" + "─" * 70)
            print(r["judge_text"])
            print("─" * 70 + "\n")

    retriever.close()

    graded = [r for r in results if r["score"] is not None]
    mean_score = round(sum(r["score"] for r in graded) / len(graded), 2) if graded else None
    tot_fab = sum(r["faithfulness"]["fabricated"] for r in results)
    tot_mis = sum(r["faithfulness"]["misattributed"] for r in results)
    tot_near = sum(r["faithfulness"]["near_verbatim"] for r in results)
    summary = {
        "n_cases": len(results),
        "mean_score": mean_score,
        "fabricated_total": tot_fab,
        "misattributed_total": tot_mis,
        "near_verbatim_total": tot_near,
    }
    print(f"\n{'─' * 60}")
    print(f"  Mean rubric score:   {mean_score}/10  ({len(graded)}/{len(results)} graded)")
    print(f"  Fabricated quotes:   {tot_fab}  (must be ≤ baseline)")
    print(f"  Misattributed quotes:{tot_mis}  (must be ≤ baseline)")
    print(f"  Near-verbatim quotes:{tot_near}")
    print(f"{'─' * 60}\n")

    out = {
        "label": args.label,
        "judge_model": args.judge_model,
        "summary": summary,
        "results": results,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        if not args.quiet:
            print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
