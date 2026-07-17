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
# Phantom-provision guard block (application/_phantom.py): citations to
# provisions that do not exist in the cited regulation (draft-numbering
# leakage). A distinct critical-defect class from fabricated quotes.
_PHANTOM_RE = re.compile(r"PHANTOM CITATION FLAG\*\*\s*[—\-]\s*(\d+)\s+statement", re.I)


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
        "judge_panel": {},
        "timed_out": True,
    }
    _apply_reliance_gate(r)
    return r


def _parse_faithfulness(answer: str) -> dict[str, int]:
    """Extract fabricated / misattributed / near-verbatim quote counts."""
    fab = _FABRICATED_RE.search(answer)
    mis = _MISATTRIBUTED_RE.search(answer)
    near = _NEAR_RE.search(answer)
    phantom = _PHANTOM_RE.search(answer)
    total = 0
    if fab:
        total = max(total, int(fab.group(2)))
    if mis:
        total = max(total, int(mis.group(2)))
    return {
        "fabricated": int(fab.group(1)) if fab else 0,
        "misattributed": int(mis.group(1)) if mis else 0,
        "near_verbatim": int(near.group(1)) if near else 0,
        "phantom_citations": int(phantom.group(1)) if phantom else 0,
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


# ---------------------------------------------------------------------------
# Judge panel — cross-family judging to defuse self-preference bias
#
# The answer generator is Mistral; grading with a Mistral judge invites
# self-preference bias — an LLM systematically scores its own family's outputs
# higher and prefers low-perplexity/familiar text ("Self-Preference Bias in
# LLM-as-a-Judge", arXiv:2410.21819). Median-of-N runs on one model reduces
# variance but NOT this bias. A panel of diverse-family judges is the standard
# mitigation (Ensemble/Panel-as-Judges). The panel is provider-agnostic and
# degrades gracefully: any provider whose SDK or API key is missing is skipped
# with a warning, so this runs Mistral-only today and activates cross-family the
# moment a key is added (openai is already installed; `pip install anthropic`
# enables the Claude judge). Keeping each judge's own median in ``per_model``
# makes the self-preference gap *visible* instead of averaging it away.
#
# Panel spec: comma-separated "provider:model" via --judge-panel or
# CRSS_JUDGE_PANEL, e.g. "mistral:mistral-large-latest,anthropic:claude-sonnet-5".
# ---------------------------------------------------------------------------


def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _provider_available(provider: str) -> tuple[bool, str]:
    """Return ``(available, reason_if_not)`` for a judge provider."""
    if provider == "mistral":
        ok = bool(os.environ.get("MISTRAL_API_KEY"))
        return ok, "" if ok else "MISTRAL_API_KEY not set"
    if provider == "openai":
        if not _has_module("openai"):
            return False, "openai SDK not installed"
        ok = bool(os.environ.get("OPENAI_API_KEY"))
        return ok, "" if ok else "OPENAI_API_KEY not set"
    if provider == "anthropic":
        if not _has_module("anthropic"):
            return False, "anthropic SDK not installed (pip install anthropic)"
        ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
        return ok, "" if ok else "ANTHROPIC_API_KEY not set"
    return False, f"unknown provider {provider!r}"


def _call_mistral(model: str, prompt: str) -> str:
    from mistralai.client import Mistral
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    resp = client.chat.complete(
        model=model, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def _call_openai(model: str, prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(model: str, prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model=model, max_tokens=2000, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        b.text for b in resp.content if getattr(b, "type", "") == "text"
    )


_PROVIDER_CALL = {
    "mistral": _call_mistral,
    "openai": _call_openai,
    "anthropic": _call_anthropic,
}


def _parse_panel_spec(spec: str) -> list[tuple[str, str]]:
    """Parse ``"provider:model,provider:model"`` into ``[(provider, model), …]``.

    A bare model with no ``provider:`` prefix defaults to ``mistral`` so an old
    ``--judge-model`` value passed through here still works.
    """
    panel: list[tuple[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            provider, model = item.split(":", 1)
            panel.append((provider.strip().lower(), model.strip()))
        else:
            panel.append(("mistral", item))
    return panel


def _resolve_panel(panel_spec: str | None, judge_model: str) -> list[tuple[str, str]]:
    """Resolve the effective panel, dropping providers with no SDK/key."""
    raw = _parse_panel_spec(panel_spec) if panel_spec else [("mistral", judge_model)]
    resolved: list[tuple[str, str]] = []
    for provider, model in raw:
        ok, reason = _provider_available(provider)
        if ok:
            resolved.append((provider, model))
        else:
            print(f"  ⚠ judge {provider}:{model} skipped — {reason}", file=sys.stderr)
    return resolved


def _judge(
    prompt: str, panel: list[tuple[str, str]], runs: int,
) -> tuple[float | None, list[float], str, str, dict]:
    """Run every model in *panel* *runs* times; aggregate across all calls.

    Returns ``(median_score, all_scores, majority_reliance, last_text, per_model)``.
    Median over the *pooled* judge calls (models × runs): a cross-family panel
    both lowers variance and cancels each model's self-preference, since no one
    family dominates the pool. ``per_model`` retains each judge's own median and
    verdict so the self-preference gap can be inspected rather than hidden.
    """
    import statistics

    all_scores: list[float] = []
    all_verdicts: list[str] = []
    per_model: dict[str, dict] = {}
    last_text = ""
    for provider, model in panel:
        m_scores: list[float] = []
        m_verdicts: list[str] = []
        for _ in range(runs):
            last_text = _PROVIDER_CALL[provider](model, prompt)
            s = _parse_score(last_text)
            if s is not None:
                m_scores.append(s)
                all_scores.append(s)
            v = _parse_reliance(last_text)
            m_verdicts.append(v)
            all_verdicts.append(v)
        per_model[f"{provider}:{model}"] = {
            "median": round(statistics.median(m_scores), 2) if m_scores else None,
            "scores": m_scores,
            "reliance": _majority_reliance(m_verdicts),
        }
    agg = round(statistics.median(all_scores), 2) if all_scores else None
    return agg, all_scores, _majority_reliance(all_verdicts), last_text, per_model


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
    if fc.get("phantom_citations", 0):
        defects.append(f"phantom_citations={fc['phantom_citations']}")
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
        d.startswith(("fabricated", "misattributed", "phantom", "answer_key_failed"))
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
    judge_panel: list[tuple[str, str]],
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
    agg, scores, reliance, judge_text, per_model = _judge(prompt, judge_panel, judge_runs)
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
        "judge_panel": per_model,
    }
    _apply_reliance_gate(result)
    return result


def _compute_graph_delta(case: dict, retriever, *, k: int) -> dict:
    """Retrieval-level graph-ablation delta for one case, fused into the quality
    run. Reuses the ablation's retrieval-only arm — NO second generation: the
    graph-on answer was already produced and judged above; here we only diff
    retrieved-context ``must_cite`` coverage graph-on vs graph-off (~2 cheap
    retrieval passes). This is the *unconfounded* delta (retrieval, not answer
    text), the one worth trusting. Restores CRSS_GRAPH_EXPANSION=1 afterwards so
    the next case still generates on the production (graph-on) path.
    """
    from scripts.eval_graph_ablation import _run_arm_retrieval
    try:
        full = _run_arm_retrieval(case, retriever, k=k, graph_on=True)
        flat = _run_arm_retrieval(case, retriever, k=k, graph_on=False)
    except Exception as exc:  # noqa: BLE001 — a delta failure must not sink the case
        return {"error": str(exc)}
    finally:
        os.environ["CRSS_GRAPH_EXPANSION"] = "1"
    full_missed = set(full.get("missed_cites") or [])
    flat_missed = set(flat.get("missed_cites") or [])
    rf, rl = full.get("cite_recall"), flat.get("cite_recall")
    return {
        "cite_recall_full": rf,
        "cite_recall_flat": rl,
        "cite_recall_delta": round(rf - rl, 3) if rf is not None and rl is not None else None,
        "recovered_by_graph": sorted(flat_missed - full_missed),
        "lost_by_graph": sorted(full_missed - flat_missed),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="CRSS answer-quality eval (LLM judge)")
    ap.add_argument("--case", nargs="+", metavar="ID", help="Run only these case IDs")
    ap.add_argument("--k", type=int, default=20, help="Retrieval budget k (default: 20)")
    ap.add_argument("--judge-model", default=os.environ.get("CRSS_JUDGE_MODEL", "mistral-large-latest"),
                    help="Single-judge model (used when --judge-panel is not given)")
    ap.add_argument("--judge-panel", default=os.environ.get("CRSS_JUDGE_PANEL"),
                    help='Cross-family judge panel, e.g. "mistral:mistral-large-latest,'
                         'anthropic:claude-sonnet-5,openai:gpt-4o". Providers with no '
                         'SDK/key are skipped. Defuses same-model self-preference bias.')
    ap.add_argument("--judge-runs", type=int, default=1,
                    help="Judge calls per case PER MODEL, median-aggregated (use 3 for report numbers)")
    ap.add_argument("--answer-file", help="Grade this answer file instead of regenerating (single case only)")
    ap.add_argument("--out", help="Write full JSON results to this path (bare filename -> eval/runs/)")
    ap.add_argument("--with-graph-delta", action="store_true",
                    help="Also compute the retrieval-level graph-ablation delta per keyed case "
                         "(reuses the retrieval phase — NO second generation; ~2 cheap retrieval "
                         "passes/case). Surfaces the graph's cite-recall contribution alongside "
                         "the quality score. Restores CRSS_GRAPH_EXPANSION=1 after each case.")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-case judge text")
    ap.add_argument("--label", default="", help="Free-text tag stored in output (e.g. commit sha)")
    args = ap.parse_args()

    # Default: archive runs under eval/runs/. A bare --out filename (no directory)
    # resolves there; an explicit path (with a directory) is used as-is.
    if args.out and os.path.dirname(args.out) == "":
        (ROOT / "eval" / "runs").mkdir(parents=True, exist_ok=True)
        args.out = str(ROOT / "eval" / "runs" / args.out)

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

    panel = _resolve_panel(args.judge_panel, args.judge_model)
    if not panel:
        print("No judge providers available — set MISTRAL_API_KEY (or another "
              "provider's SDK+key for the panel).", file=sys.stderr)
        return 2
    panel_str = ", ".join(f"{p}:{m}" for p, m in panel)

    if not args.quiet:
        print(f"\nCRSS Answer-Quality Eval — {len(cases)} case(s), judge panel=[{panel_str}], label={args.label or '(none)'}")
        print("Connecting to Neo4j and loading embeddings …")
    from retrieval.graph_retriever import GraphRetriever
    retriever = GraphRetriever()

    signal.signal(signal.SIGALRM, _on_alarm)

    def _flush_partial() -> None:
        # Kill-safe: persist after every case so a hang/kill never loses the run.
        if args.out:
            Path(args.out).write_text(json.dumps(
                {"label": args.label, "judge_panel": panel_str,
                 "partial": True, "results": results}, indent=2))

    def _attempt_case(case: dict) -> dict:
        """One attempt at a case under the wall-clock timeout."""
        _timed_out_flag["v"] = False
        signal.alarm(_CASE_TIMEOUT_S)
        try:
            return _run_case(
                case, retriever, rubric,
                k=args.k, judge_panel=panel, judge_runs=args.judge_runs,
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
        if args.with_graph_delta and (case.get("answer_key") or {}).get("must_cite"):
            r["graph_delta"] = _compute_graph_delta(case, retriever, k=args.k)
        results.append(r)
        _flush_partial()
        score_str = f"{r['score']}" if r["score"] is not None else "PARSE_FAIL"
        fc = r["faithfulness"]
        faith_str = (
            f"fab={fc['fabricated']} mis={fc['misattributed']} "
            f"near={fc['near_verbatim']}/{fc['total_quotes']}"
        )
        if fc.get("phantom_citations"):
            faith_str += f" phantom={fc['phantom_citations']}"
        verdict = r["reliance_final"] + (" ⛔GATED" if r.get("reliance_gated") else "")
        kc = r.get("answer_key_check")
        key_str = ""
        if kc:
            key_str = f"  key={'PASS' if kc['passed'] else 'FAIL'}"
            if kc.get("missed_cites"):
                key_str += f" missed={kc['missed_cites']}"
        gd_str = ""
        g = r.get("graph_delta")
        if g and g.get("cite_recall_delta") is not None:
            gd_str = f"  graphΔ={g['cite_recall_delta']:+.0%}"
            if g["recovered_by_graph"]:
                gd_str += f" +{g['recovered_by_graph']}"
            if g["lost_by_graph"]:
                gd_str += f" ⚠-{g['lost_by_graph']}"
        print(f"  {r['id']}  score={score_str}/10  [{verdict}]  {faith_str}{key_str}{gd_str}  "
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

    # Per-judge means across cases. With a cross-family panel this exposes the
    # self-preference gap directly: if the Mistral judge sits well above the
    # Claude/GPT judges on Mistral-generated answers, that spread IS the bias
    # the single-model score would have hidden.
    per_judge_scores: dict[str, list[float]] = {}
    for r in graded:
        for model_key, info in (r.get("judge_panel") or {}).items():
            if info.get("median") is not None:
                per_judge_scores.setdefault(model_key, []).append(info["median"])
    per_judge_mean = {
        m: round(sum(v) / len(v), 2) for m, v in per_judge_scores.items() if v
    }

    # Fused retrieval-level graph-ablation delta (only when --with-graph-delta).
    gd = [r["graph_delta"] for r in results
          if (r.get("graph_delta") or {}).get("cite_recall_delta") is not None]
    graph_delta_summary = None
    if gd:
        gd_full = round(sum(g["cite_recall_full"] for g in gd) / len(gd), 3)
        gd_flat = round(sum(g["cite_recall_flat"] for g in gd) / len(gd), 3)
        graph_delta_summary = {
            "n_cases": len(gd),
            "mean_cite_recall_full": gd_full,
            "mean_cite_recall_flat": gd_flat,
            "mean_cite_recall_delta": round(gd_full - gd_flat, 3),
            "cites_recovered_only_by_graph": sum(len(g["recovered_by_graph"]) for g in gd),
            "cites_lost_by_graph": sum(len(g["lost_by_graph"]) for g in gd),
        }

    summary = {
        "n_cases": len(results),
        "zero_critical_defect_rate_pct": zcd_rate,
        "zero_critical_defect_cases": len(clean),
        "critical_cases": {
            r["id"]: r["critical_defects"] for r in defective
        },
        "mean_score": mean_score,
        "per_judge_mean_score": per_judge_mean,
        "answer_key_pass": f"{key_pass}/{len(keyed)}" if keyed else None,
        "mean_cite_recall": mean_cite_recall,
        "fabricated_total": tot_fab,
        "misattributed_total": tot_mis,
        "near_verbatim_total": tot_near,
        "graph_delta": graph_delta_summary,
    }
    print(f"\n{'─' * 60}")
    print(f"  Zero-critical-defect: {zcd_rate}%  ({len(clean)}/{len(results)} cases)  ← headline")
    print(f"  Mean rubric score:   {mean_score}/10  ({len(graded)}/{len(results)} graded)")
    if len(per_judge_mean) > 1:
        spread = max(per_judge_mean.values()) - min(per_judge_mean.values())
        print("  Per-judge means (self-preference check):")
        for m, v in sorted(per_judge_mean.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {m:<34} {v}/10")
        print(f"    spread {spread:+.2f}  (large gap ⇒ the same-family judge was inflating)")
    if keyed:
        print(f"  Answer-key pass:     {key_pass}/{len(keyed)}  (mean cite-recall {mean_cite_recall:.0%})")
    print(f"  Fabricated quotes:   {tot_fab}  (must be ≤ baseline)")
    print(f"  Misattributed quotes:{tot_mis}  (must be ≤ baseline)")
    print(f"  Near-verbatim quotes:{tot_near}")
    if graph_delta_summary:
        gds = graph_delta_summary
        print(f"  Graph cite-recall:   full {gds['mean_cite_recall_full']:.0%}  "
              f"flat {gds['mean_cite_recall_flat']:.0%}  Δ {gds['mean_cite_recall_delta']:+.0%}  "
              f"(retrieval-level, {gds['n_cases']} keyed cases)  ← graph contribution")
        lost = gds['cites_lost_by_graph']
        print(f"  Cites recovered only by graph: {gds['cites_recovered_only_by_graph']}"
              + (f"   ⚠ lost by graph: {lost}  (investigate)" if lost else ""))
    if defective:
        print("  Critical cases:")
        for r in defective:
            print(f"    {r['id']}: {', '.join(r['critical_defects'])}")
    print(f"{'─' * 60}\n")

    out = {
        "label": args.label,
        "judge_panel": panel_str,
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
