#!/usr/bin/env python3
"""Revision-delta grader — does the audit/revision loop help or hurt?

CRSS answers in two passes: a draft, then a bounded audit-and-revise loop
(``application/_audit.py``). The loop costs a second generation, and an earlier
A/B found it **net-negative** — it reintroduced quotes typed from memory
(fabricated quotes 2 → 11) for no rubric gain. This grader makes that trade-off
measurable per case, and — crucially for cost — does it **deterministically**,
with zero LLM-judge calls on the routine path.

It reads the shared artifact from ``scripts/generate_eval_artifact.py`` (one
generation per case captured the pre-audit *draft* and post-audit *final*, both
finalised through the identical tail) and, for every case where the loop
actually fired (draft != final), diffs:

  * ``Δcite_recall`` / ``Δstate_recall`` — law-grounded correctness via the
    answer-key checker (``scripts/check_answer_keys.check_answer``): did the
    revision get objectively *more right*?
  * ``Δfabrication`` — ``final_fab − draft_fab`` (``unverified`` +
    ``misattributed``). **Positive = the revision INTRODUCED fabrication** — the
    exact failure the loop was caught doing.
  * pass-flip and ``Δconfidence`` — secondary.

Each fired case is classified **helped / hurt / neutral** (adding fabrication is
always "hurt", even if correctness rose), and the run prints a net verdict: is
the second pass earning its cost, or should the material-gap gate be tighter?

Optional ``--judge`` (OFF by default) adds the holistic LLM score delta by
reusing ``eval_answer_quality``'s judge over the *pre-generated* draft/final
(``answer_override`` — no regeneration). That is the only paid path; reserve it
for milestone runs.

Usage::

    python scripts/eval_revision_delta.py --artifact artifact_vN.json
    python scripts/eval_revision_delta.py --artifact artifact_vN.json --all      # include non-fired cases
    python scripts/eval_revision_delta.py --artifact artifact_vN.json --judge     # + LLM score delta (paid)
    python scripts/eval_revision_delta.py --artifact artifact_vN.json --out revision_delta_vN.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.check_answer_keys import check_answer

QUALITY_SET_PATH = _PROJECT_ROOT / "eval" / "quality_set.json"


def _fab_total(fab: dict | None) -> int:
    """Fabrication count = absent-from-corpus (unverified) + displaced (misattributed)."""
    fab = fab or {}
    return int(fab.get("unverified", 0) or 0) + int(fab.get("misattributed", 0) or 0)


def _conf_score(conf: dict | None) -> float | None:
    if not conf:
        return None
    v = conf.get("confidence_score")
    return round(float(v), 3) if v is not None else None


def _classify(d_cite: float, d_state: float, d_fab: int) -> str:
    """Verdict for one fired case. Fabrication added dominates: a revision that
    injects a fabricated quote is 'hurt' even if it also cited one more article,
    because a fabricated quotation is the one defect a compliance answer cannot
    carry."""
    if d_fab > 0:
        return "hurt"          # revision introduced fabrication
    if d_cite < 0 or d_state < 0:
        return "hurt"          # revision dropped a decisive cite / key fact
    if d_cite > 0 or d_state > 0 or d_fab < 0:
        return "helped"        # more correct, or cleaned up a fabrication
    return "neutral"           # rewrote prose without moving any hard signal


def _score_case(r: dict, key: dict | None) -> dict:
    """Draft-vs-final deltas for one artifact result. Pure; no LLM."""
    draft, final = r.get("draft", ""), r.get("final", "")
    d_fab = _fab_total(r.get("final_fab")) - _fab_total(r.get("draft_fab"))

    out: dict = {
        "id": r["id"],
        "revised": bool(r.get("revised")),
        "d_fab": d_fab,
        "draft_fab": _fab_total(r.get("draft_fab")),
        "final_fab": _fab_total(r.get("final_fab")),
        "d_confidence": None,
        "d_cite": 0.0,
        "d_state": 0.0,
        "pass_flip": "n/a",
        "has_key": bool(key),
    }
    dc, fc = _conf_score(r.get("draft_confidence")), _conf_score(r.get("final_confidence"))
    if dc is not None and fc is not None:
        out["d_confidence"] = round(fc - dc, 3)

    if key:
        dk, fk = check_answer(draft, key), check_answer(final, key)
        out["d_cite"] = round(fk["cite_recall"] - dk["cite_recall"], 3)
        out["d_state"] = round(fk["state_recall"] - dk["state_recall"], 3)
        out["draft_cite_recall"] = dk["cite_recall"]
        out["final_cite_recall"] = fk["cite_recall"]
        if dk["passed"] != fk["passed"]:
            out["pass_flip"] = "gained" if fk["passed"] else "lost"
        else:
            out["pass_flip"] = "same"

    out["verdict"] = _classify(out["d_cite"], out["d_state"], d_fab)
    return out


def _judge_overlay(fired: list[dict], results_by_id: dict, cases_by_id: dict,
                   *, panel_spec: str | None, judge_model: str | None, runs: int) -> None:
    """OFF by default. Judge the pre-generated draft & final (no regeneration).

    Reuses eval_answer_quality's judge via ``answer_override`` so this stays the
    single source of the rubric prompt. Best-effort: if the judge can't be set
    up (no API key), it warns and leaves the deterministic verdict intact.
    """
    try:
        from scripts.eval_answer_quality import _run_case, _resolve_panel, RUBRIC_PATH
        rubric = RUBRIC_PATH.read_text()
        panel = _resolve_panel(panel_spec, judge_model)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ --judge overlay unavailable ({exc}); reporting deterministic deltas only.",
              file=sys.stderr)
        return
    print(f"\n  Judging {len(fired)} fired case(s) × 2 answers (paid) …", file=sys.stderr)
    for row in fired:
        r = results_by_id[row["id"]]
        case = cases_by_id.get(row["id"]) or {"id": row["id"], "question": r.get("question", "")}
        try:
            dj = _run_case(case, None, rubric, k=1, judge_panel=panel, judge_runs=runs,
                           answer_override=r.get("draft", ""))["score"]
            fj = _run_case(case, None, rubric, k=1, judge_panel=panel, judge_runs=runs,
                           answer_override=r.get("final", ""))["score"]
        except Exception as exc:  # noqa: BLE001 — one judge failure must not sink the report
            print(f"    {row['id']}: judge error ({exc})", file=sys.stderr)
            continue
        row["judge_draft"], row["judge_final"] = dj, fj
        row["judge_delta"] = round(fj - dj, 3) if (dj is not None and fj is not None) else None


def run(artifact_path: Path, *, include_all: bool, out: str | None,
        judge: bool, panel_spec: str | None, judge_model: str | None, judge_runs: int) -> dict:
    artifact = json.loads(artifact_path.read_text())
    results = artifact.get("results", [])
    results_by_id = {r["id"]: r for r in results}
    cases_by_id = {c["id"]: c for c in json.loads(QUALITY_SET_PATH.read_text())}
    keys_by_id = {cid: c.get("answer_key") for cid, c in cases_by_id.items()}

    errors = [r["id"] for r in results if r.get("error")]
    scored, not_fired = [], []
    for r in results:
        if r.get("error"):
            continue
        row = _score_case(r, keys_by_id.get(r["id"]))
        if row["revised"] or include_all:
            scored.append(row)
        else:
            not_fired.append(row)

    fired = [row for row in scored if row["revised"]]
    if judge and fired:
        _judge_overlay(fired, results_by_id, cases_by_id,
                       panel_spec=panel_spec, judge_model=judge_model, runs=judge_runs)

    tally = {"helped": 0, "hurt": 0, "neutral": 0}
    for row in scored:
        tally[row["verdict"]] += 1
    net_fab = sum(row["d_fab"] for row in scored)
    cite_deltas = [row["d_cite"] for row in scored if row["has_key"]]
    state_deltas = [row["d_state"] for row in scored if row["has_key"]]
    pass_gained = sum(1 for row in scored if row["pass_flip"] == "gained")
    pass_lost = sum(1 for row in scored if row["pass_flip"] == "lost")

    # ── report ────────────────────────────────────────────────────────────────
    scope = "all cases" if include_all else "cases where the revision fired"
    print(f"\n=== Revision delta (draft → final) — {scope} ===")
    print(f"  artifact               : {artifact_path.name}  "
          f"(n={len(results)}, ok={len(results) - len(errors)}, errors={len(errors)})")
    print(f"  revision fired on      : {len(fired)}/{len(results) - len(errors)} case(s)")
    print(f"  scored                 : {len(scored)}")
    print(f"  helped / hurt / neutral: {tally['helped']} / {tally['hurt']} / {tally['neutral']}")
    print(f"  net Δ fabrication      : {net_fab:+d}   "
          f"(> 0 ⇒ the revision loop ADDS fabrication overall)")
    if cite_deltas:
        print(f"  mean Δ cite_recall     : {statistics.mean(cite_deltas):+.3f}")
        print(f"  mean Δ state_recall    : {statistics.mean(state_deltas):+.3f}")
        print(f"  answer-key pass        : +{pass_gained} gained / -{pass_lost} lost")
    if any("judge_delta" in row for row in fired):
        jds = [row["judge_delta"] for row in fired if row.get("judge_delta") is not None]
        if jds:
            print(f"  mean Δ judge score     : {statistics.mean(jds):+.3f}  (LLM overlay)")

    # verdict
    verdict = ("NET-NEGATIVE — the loop hurts more than it helps; tighten the "
               "material-gap gate" if tally["hurt"] > tally["helped"] or net_fab > 0
               else "NET-POSITIVE — the loop earns its second generation"
               if tally["helped"] > tally["hurt"] and net_fab <= 0
               else "NEUTRAL — the loop mostly rewrites prose without moving hard signals")
    print(f"\n  VERDICT: {verdict}")

    if fired:
        print(f"\n  Per-fired-case (Δcite / Δstate / Δfab / verdict):")
        for row in sorted(fired, key=lambda x: (x["verdict"] != "hurt", x["id"])):
            j = f"  judgeΔ={row['judge_delta']:+.2f}" if row.get("judge_delta") is not None else ""
            print(f"    {row['id']:8}  {row['d_cite']:+.2f} / {row['d_state']:+.2f} / "
                  f"{row['d_fab']:+d}   {row['verdict']:8}{j}")
    if errors:
        print(f"\n  {len(errors)} errored case(s) skipped: {', '.join(errors[:12])}")

    summary = {
        "artifact": artifact_path.name,
        "n": len(results), "errors": errors, "fired": len(fired), "scored": len(scored),
        "tally": tally, "net_fab": net_fab,
        "mean_d_cite": round(statistics.mean(cite_deltas), 3) if cite_deltas else None,
        "mean_d_state": round(statistics.mean(state_deltas), 3) if state_deltas else None,
        "pass_gained": pass_gained, "pass_lost": pass_lost,
        "verdict": verdict, "cases": scored,
    }
    if out:
        out_path = Path(out)
        if not out_path.is_absolute() and out_path.parent == Path("."):
            out_path = _PROJECT_ROOT / "eval" / "runs" / out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\n  wrote {out_path}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Draft-vs-final revision-delta grader (deterministic; judge optional).")
    ap.add_argument("--artifact", required=True,
                    help="artifact JSON from generate_eval_artifact.py (bare name → eval/runs/)")
    ap.add_argument("--all", action="store_true",
                    help="score every case, not only those where the revision fired")
    ap.add_argument("--judge", action="store_true",
                    help="ALSO judge draft vs final with the LLM (paid; off by default)")
    ap.add_argument("--judge-panel", default=os.environ.get("CRSS_JUDGE_PANEL"),
                    help="judge panel spec (see eval_answer_quality); or CRSS_JUDGE_PANEL")
    ap.add_argument("--judge-model", default=os.environ.get("CRSS_JUDGE_MODEL", "mistral-large-latest"),
                    help="single judge model when no panel (default mistral-large-latest)")
    ap.add_argument("--judge-runs", type=int, default=1, help="judge samples per answer (default 1)")
    ap.add_argument("--out", default=None, help="write summary JSON (bare name → eval/runs/)")
    args = ap.parse_args()

    ap_path = Path(args.artifact)
    if not ap_path.is_absolute() and ap_path.parent == Path("."):
        ap_path = _PROJECT_ROOT / "eval" / "runs" / args.artifact
    if not ap_path.exists():
        print(f"Artifact not found: {ap_path}", file=sys.stderr)
        return 1

    run(ap_path, include_all=args.all, out=args.out, judge=args.judge,
        panel_spec=args.judge_panel, judge_model=args.judge_model, judge_runs=args.judge_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
