#!/usr/bin/env python3
"""Generate-once eval artifact — the shared substrate for the cheap graders.

The eval strategy (see ``docs/eval_strategy.md``) is a cost pyramid: the LLM
judge is the expensive, noisy tip; deterministic checks are the free base. The
single biggest saving is to **generate each answer once** and let every cheap
grader read that one artifact, instead of each eval regenerating.

This runner produces that artifact. It drives the pipeline through ``ask_stream``
with its opt-in ``capture`` hook, so from ONE generation per case it records:

  * ``draft``  — the pre-audit first pass, finalised through the *identical*
                 deterministic tail (pointer-resolution → verify → bold →
                 post-process) as the final, so the two are graded on equal
                 footing — the only difference is whether the audit/revision
                 loop ran;
  * ``final`` / ``answer`` — the post-audit answer the user would see;
  * ``revised`` — did the audit loop actually change the answer;
  * ``draft_fab`` / ``final_fab`` — faithfulness fabrication counts
                 (``unverified`` + ``misattributed``) for each;
  * ``draft_confidence`` / ``final_confidence`` and the retrieved provision ids.

The envelope is ``{"meta": …, "results": [{"id", "answer", …}]}`` — the same
shape ``scripts/check_answer_keys.py --results`` consumes, and ``answer`` is the
final, so ONE artifact feeds three graders with **no** regeneration:

    python scripts/generate_eval_artifact.py --out artifact_vN.json           # generate once
    python scripts/eval_revision_delta.py    --artifact artifact_vN.json      # draft vs final (free)
    python scripts/check_answer_keys.py      --results  artifact_vN.json      # law-grounded correctness (free)
    # …and eval_answer_quality can judge `answer` via answer_override (the only paid step, run rarely).

Cost: one generation per case, ZERO judge calls. ``CRSS_CLARIFY`` defaults to 0
here (like the ablation harness) so role-less cases are not stubbed by the scope
gate; ``CRSS_AUDIT`` MUST stay on (its default) or draft == final everywhere and
the artifact says nothing about the revision loop — a warning fires if it is off.

Usage::

    python scripts/generate_eval_artifact.py --out artifact_vN.json
    python scripts/generate_eval_artifact.py --case HQ_001 HQ_006 --out /tmp/a.json
    python scripts/generate_eval_artifact.py --limit 5 --k 20
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import signal
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env", override=False)

# Off by default so the ask-first scope gate does not stub role-less cases into a
# clarify question (mirrors scripts/eval_graph_ablation.py). Override to profile
# the gate itself.
os.environ.setdefault("CRSS_CLARIFY", "0")

QUALITY_SET_PATH = _PROJECT_ROOT / "eval" / "quality_set.json"
_CASE_TIMEOUT_S = int(os.environ.get("CRSS_EVAL_CASE_TIMEOUT", "300"))


class _CaseTimeout(Exception):
    pass


def _on_alarm(signum, frame):  # noqa: ANN001
    raise _CaseTimeout()


def _resolve_out(out: str | None) -> Path | None:
    """Bare filename → eval/runs/ ; explicit path used as-is (shared convention)."""
    if not out:
        return None
    p = Path(out)
    if not p.is_absolute() and p.parent == Path("."):
        p = _PROJECT_ROOT / "eval" / "runs" / out
    return p


def _generate_case(case: dict, retriever, *, k: int) -> dict:
    """Run ONE case through ask_stream with the capture hook; no judge."""
    from application.agent import ask_stream

    question = case["question"]
    capture: dict = {}
    t0 = time.perf_counter()
    final_answer, error = "", None
    for event in ask_stream(
        question, retriever, k=k, history=case.get("history"), capture=capture
    ):
        et = event.get("type")
        if et == "done":
            final_answer = event.get("answer", "")
        elif et == "error":
            error = event.get("message")
    gen_s = round(time.perf_counter() - t0, 1)

    if error:
        return {"id": case["id"], "question": question, "answer": "",
                "error": error, "gen_s": gen_s}

    # `answer` mirrors `final` so check_answer_keys / the judge's answer_override
    # read this artifact unchanged. When no revision fired, draft == final and the
    # revision-delta grader scores the case as "loop did not change the answer".
    return {
        "id": case["id"],
        "question": question,
        "answer": capture.get("final", final_answer),
        "draft": capture.get("draft", ""),
        "final": capture.get("final", final_answer),
        "revised": bool(capture.get("revised", False)),
        "draft_fab": capture.get("draft_fab"),
        "final_fab": capture.get("final_fab"),
        "draft_confidence": capture.get("draft_confidence"),
        "final_confidence": capture.get("final_confidence"),
        "provision_ids": capture.get("provision_ids", []),
        "draft_provision_ids": capture.get("draft_provision_ids", []),
        "gen_s": gen_s,
    }


def run(cases: list[dict], *, k: int, out: str | None) -> dict:
    from retrieval.graph_retriever import GraphRetriever

    if os.environ.get("CRSS_AUDIT", "1") == "0":
        print("  ⚠ CRSS_AUDIT=0 — the audit/revision loop is OFF, so draft == final "
              "for every case and the revision-delta grader will have nothing to "
              "measure. Unset CRSS_AUDIT (or set it to 1) to capture the loop.",
              file=sys.stderr)

    retriever = GraphRetriever()
    signal.signal(signal.SIGALRM, _on_alarm)
    results: list[dict] = []
    try:
        for i, case in enumerate(cases, 1):
            cid = case.get("id", f"case_{i}")
            print(f"  [{i}/{len(cases)}] {cid} … ", end="", flush=True)
            signal.alarm(_CASE_TIMEOUT_S)
            try:
                r = _generate_case(case, retriever, k=k)
            except _CaseTimeout:
                r = {"id": cid, "question": case.get("question", ""), "answer": "",
                     "error": f"TIMEOUT after {_CASE_TIMEOUT_S}s", "gen_s": _CASE_TIMEOUT_S}
            except Exception as exc:  # noqa: BLE001 — one bad case must not sink the run
                r = {"id": cid, "question": case.get("question", ""), "answer": "",
                     "error": f"{type(exc).__name__}: {exc}", "gen_s": 0.0}
            finally:
                signal.alarm(0)
            results.append(r)
            if r.get("error"):
                print(f"ERROR ({r['error'][:60]})")
            else:
                tag = "revised" if r["revised"] else "no-revision"
                fab = (r.get("final_fab") or {})
                print(f"{tag}  fab(final)={fab.get('unverified', 0)}+{fab.get('misattributed', 0)}  "
                      f"{r['gen_s']}s")
    finally:
        retriever.close()

    n_ok = sum(1 for r in results if not r.get("error"))
    n_revised = sum(1 for r in results if r.get("revised"))
    artifact = {
        "meta": {
            "generated": _dt.datetime.now().isoformat(timespec="seconds"),
            "n": len(results),
            "n_ok": n_ok,
            "n_revised": n_revised,
            "k": k,
            "clarify": os.environ.get("CRSS_CLARIFY", "1"),
            "audit": os.environ.get("CRSS_AUDIT", "1"),
            "graph_expansion": os.environ.get("CRSS_GRAPH_EXPANSION", "1"),
            "model": os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
            "audit_model": os.environ.get("CRSS_AUDIT_MODEL", "mistral-medium-latest"),
        },
        "results": results,
    }

    print(f"\n  generated {n_ok}/{len(results)} case(s); revision fired on {n_revised}.")
    out_path = _resolve_out(out)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))
        print(f"  wrote {out_path}")
    return artifact


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the shared eval artifact (generate once, grade many).")
    ap.add_argument("--case", nargs="*", default=None, help="only these case ids (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cases")
    ap.add_argument("--k", type=int, default=20, help="retrieval top-k")
    ap.add_argument("--out", default=None, help="write JSON (bare name → eval/runs/)")
    args = ap.parse_args()

    cases: list[dict] = json.loads(QUALITY_SET_PATH.read_text())
    if args.case:
        want = set(args.case)
        cases = [c for c in cases if c["id"] in want]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("No cases selected.", file=sys.stderr)
        return 1

    print(f"Generating eval artifact for {len(cases)} case(s) "
          f"(CLARIFY={os.environ.get('CRSS_CLARIFY')}, AUDIT={os.environ.get('CRSS_AUDIT', '1')}, k={args.k})…")
    run(cases, k=args.k, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
