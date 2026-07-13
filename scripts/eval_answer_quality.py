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
    r = {
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
    _apply_reliance_gate(r)
    return r


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


def _majority_reliance(verdicts: list[str]) -> str:
    """Majority vote over reliance verdicts; ties break to the *worse* verdict.

    The score takes the median of the judge runs, so the verdict must not be
    read off a single (the last) run — one aberrant grading otherwise flips
    the reliance label while the score stays put. Conservative tie-break: for
    a compliance tool, when the judges split, the stricter verdict stands.
    """
    real = [v for v in verdicts if v in _RELIANCE_PHRASES]
    if not real:
        return verdicts[-1] if verdicts else "(unparsed)"
    counts = {v: real.count(v) for v in set(real)}
    best = max(counts.values())
    tied = [v for v, n in counts.items() if n == best]
    # _RELIANCE_PHRASES is ordered best → worst; pick the worst among the tied.
    return max(tied, key=_RELIANCE_PHRASES.index)


def _judge(prompt: str, model: str, runs: int) -> tuple[float | None, list[float], str, str]:
    """Run the judge `runs` times.

    Returns ``(median_score, all_scores, majority_reliance, last_text)``.
    Median (not mean) across runs: judge noise is heavy-tailed — one aberrant
    grading drags a mean but not a median, so multi-run numbers destined for a
    report should be run with ``--judge-runs 3``. The reliance verdict is
    majority-voted across the same runs (ties break to the worse verdict).
    """
    import statistics

    from mistralai.client import Mistral

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    scores: list[float] = []
    verdicts: list[str] = []
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
        verdicts.append(_parse_reliance(last_text))
    agg = round(statistics.median(scores), 2) if scores else None
    return agg, scores, _majority_reliance(verdicts), last_text


# ---------------------------------------------------------------------------
# Critical-defect gate
#
# The metric a compliance officer actually experiences is not the mean rubric
# score but "can I forward this without re-verifying every line" — so the
# harness derives a deterministic per-case verdict that the judge cannot
# override. A fabricated or misattributed quote is a critical defect no matter
# how the judge scored the prose (observed failure: 8.5/10 "strong first
# draft" with 4 fabricated quotes). The raw judge `score` is stored untouched
# for run-to-run comparability; the gate acts on the reliance verdict and the
# headline zero-critical-defect rate.
# ---------------------------------------------------------------------------

_RELIANCE_UNRELIABLE = {
    "Cannot be relied on without major revision",
    "Unsafe for compliance reliance",
}


def _critical_defects(result: dict) -> list[str]:
    """Return the list of critical defects for a graded case (empty = clean)."""
    defects: list[str] = []
    fc = result.get("faithfulness") or {}
    if fc.get("fabricated", 0):
        defects.append(f"fabricated_quotes={fc['fabricated']}")
    if fc.get("misattributed", 0):
        defects.append(f"misattributed_quotes={fc['misattributed']}")
    # Deterministic answer-key failure: a missed decisive citation or missed
    # key facts. This is the "omitted decisive provision" fatal rule enforced
    # by regex against a law-grounded key, not by the judge's impression.
    kc = result.get("answer_key_check")
    if kc and not kc.get("passed", True):
        detail = ",".join(kc.get("missed_cites") or []) or "state_recall_below_threshold"
        defects.append(f"answer_key_failed({detail})")
    if result.get("score") is None:
        defects.append("ungraded")  # timeout / error / parse failure — not usable
    elif result.get("reliance") in _RELIANCE_UNRELIABLE:
        defects.append("judge_verdict_unreliable")
    return defects


def _apply_reliance_gate(result: dict) -> None:
    """Attach ``critical_defects`` and the gated ``reliance_final`` verdict."""
    defects = _critical_defects(result)
    result["critical_defects"] = defects
    # Deterministic defects downgrade a positive judge verdict: unverifiable
    # quotes and missed decisive citations both make an answer something a
    # compliance officer must re-verify line by line before use.
    hard_defect = any(
        d.startswith(("fabricated", "misattributed", "answer_key_failed"))
        for d in defects
    )
    if hard_defect and result.get("reliance") not in _RELIANCE_UNRELIABLE:
        result["reliance_final"] = "Cannot be relied on without major revision"
        result["reliance_gated"] = True
    else:
        result["reliance_final"] = result.get("reliance")
        result["reliance_gated"] = False


def _verification_block(fc: dict[str, int]) -> str:
    """Render the deterministic quote-verification report for the judge."""
    return (
        "\n## CITATION VERIFICATION (deterministic machine check — outranks "
        "your impression of the citations)\n"
        "Every verbatim quote in the answer was verified against the retrieved "
        "legal corpus before this review:\n"
        f"- FABRICATED quotes (could not be grounded anywhere; redacted): {fc['fabricated']}\n"
        f"- MISATTRIBUTED quotes (real text, but absent from the provision cited): {fc['misattributed']}\n"
        f"- Near-verbatim quotes (grounded, minor wording drift): {fc['near_verbatim']}\n"
        "Apply the fatal-error rule for fabricated/misattributed quotes from "
        "the rubric above.\n"
    )


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
        # Multi-turn cases carry a `history` of prior chat turns (e.g. the
        # clarify-gate loop: role-less question → CRSS clarification → the
        # user's role reply as `question`). The agent's standalone-question
        # rewrite folds the history back in, which is exactly the production
        # path chat clients exercise.
        answer = ask(question, retriever, k=k, history=case.get("history"))
        gen_s = time.perf_counter() - t0

    # Verify quotes BEFORE judging so the deterministic report is part of the
    # judge's evidence — the judge underweights the inline warning blocks on
    # its own (observed: 8.5 with 4 fabricated quotes).
    faithfulness = _parse_faithfulness(answer)

    # Deterministic answer-key check (law-grounded key in the case file).
    key_check = None
    if case.get("answer_key"):
        from scripts.check_answer_keys import check_answer
        key_check = check_answer(answer, case["answer_key"])

    # For multi-turn cases the judge needs the full exchange to grade the
    # final answer in context of the original question.
    judged_question = question
    if case.get("history"):
        transcript = "\n".join(
            f"[{t.get('role', 'user')}] {t.get('content', '')}" for t in case["history"]
        )
        judged_question = f"{transcript}\n[user] {question}"

    # The judge_notes field tells the judge what correct behaviour looks like
    # for cases where it isn't obvious from the question alone (abstention
    # cases: the right answer is a scope refusal, not an analysis).
    notes_block = (
        f"\n## CASE-SPECIFIC GRADING NOTES\n{case['judge_notes']}\n"
        if case.get("judge_notes") else ""
    )
    prompt = (
        f"{rubric}\n\n## QUESTION\n{judged_question}\n{notes_block}"
        f"\n## CRSS ANSWER\n{answer}\n"
        + _verification_block(faithfulness)
    )

    tj = time.perf_counter()
    agg, scores, reliance, judge_text = _judge(prompt, judge_model, judge_runs)
    judge_s = time.perf_counter() - tj

    result = {
        "id": case["id"],
        "label": case.get("label", ""),
        "score": agg,
        "scores": scores,
        "reliance": reliance,
        "faithfulness": faithfulness,
        "answer_key_check": key_check,
        "gen_s": round(gen_s, 1),
        "judge_s": round(judge_s, 1),
        "answer_chars": len(answer),
        "answer": answer,
        "judge_text": judge_text,
    }
    _apply_reliance_gate(result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="CRSS answer-quality eval (LLM judge)")
    ap.add_argument("--case", nargs="+", metavar="ID", help="Run only these case IDs")
    ap.add_argument("--k", type=int, default=20, help="Retrieval budget k (default: 20)")
    ap.add_argument("--judge-model", default=os.environ.get("CRSS_JUDGE_MODEL", "mistral-large-latest"))
    ap.add_argument("--judge-runs", type=int, default=1,
                    help="Judge calls per case, median-aggregated (use 3 for report numbers)")
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
        verdict = r["reliance_final"] + (" ⛔GATED" if r.get("reliance_gated") else "")
        kc = r.get("answer_key_check")
        key_str = ""
        if kc:
            key_str = f"  key={'PASS' if kc['passed'] else 'FAIL'}"
            if kc.get("missed_cites"):
                key_str += f" missed={kc['missed_cites']}"
        print(f"  {r['id']}  score={score_str}/10  [{verdict}]  {faith_str}{key_str}  "
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

    # Headline: the fraction of answers a compliance officer could forward
    # without hitting a critical defect (fabricated/misattributed quote, a
    # judge verdict of unreliable, or a harness failure). This is the number
    # that tracks "reliably usable by a professional" — the mean score does not.
    clean = [r for r in results if not r.get("critical_defects")]
    defective = [r for r in results if r.get("critical_defects")]
    zcd_rate = round(100.0 * len(clean) / len(results), 1) if results else None
    keyed = [r for r in results if r.get("answer_key_check")]
    key_pass = sum(1 for r in keyed if r["answer_key_check"]["passed"])
    mean_cite_recall = (
        round(sum(r["answer_key_check"]["cite_recall"] for r in keyed) / len(keyed), 3)
        if keyed else None
    )
    summary = {
        "n_cases": len(results),
        "zero_critical_defect_rate_pct": zcd_rate,
        "zero_critical_defect_cases": len(clean),
        "critical_cases": {
            r["id"]: r["critical_defects"] for r in defective
        },
        "mean_score": mean_score,
        "answer_key_pass": f"{key_pass}/{len(keyed)}" if keyed else None,
        "mean_cite_recall": mean_cite_recall,
        "fabricated_total": tot_fab,
        "misattributed_total": tot_mis,
        "near_verbatim_total": tot_near,
    }
    print(f"\n{'─' * 60}")
    print(f"  Zero-critical-defect: {zcd_rate}%  ({len(clean)}/{len(results)} cases)  ← headline")
    print(f"  Mean rubric score:   {mean_score}/10  ({len(graded)}/{len(results)} graded)")
    if keyed:
        print(f"  Answer-key pass:     {key_pass}/{len(keyed)}  (mean cite-recall {mean_cite_recall:.0%})")
    print(f"  Fabricated quotes:   {tot_fab}  (must be ≤ baseline)")
    print(f"  Misattributed quotes:{tot_mis}  (must be ≤ baseline)")
    print(f"  Near-verbatim quotes:{tot_near}")
    if defective:
        print("  Critical cases:")
        for r in defective:
            print(f"    {r['id']}: {', '.join(r['critical_defects'])}")
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
