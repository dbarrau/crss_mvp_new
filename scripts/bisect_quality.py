#!/usr/bin/env python3
"""Regression bisect for answer quality.

Runs the LLM-judge answer-quality eval (scripts/eval_answer_quality.py) against a
series of git commits to localise where a quality regression was introduced.

The eval harness files (scripts/eval_answer_quality.py, eval/rubric_prompt.txt,
eval/quality_set.json) are untracked, so they survive `git checkout` and grade
each checked-out version of the agent code against a fixed question + fixed graph.

Safety: uncommitted tracked changes are stashed before the bisect and restored
in a finally block, along with the original branch — even on error or Ctrl-C.

Usage
-----
    python scripts/bisect_quality.py --case HQ_001
    python scripts/bisect_quality.py --case HQ_001 --commits 8431d15 2939a11 dd75b68
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Default sweep: newest -> oldest across the suspected regression window.
DEFAULT_COMMITS = [
    "8431d15",  # hybrid BM25+RRF retrieval, cross-encoder reranker, eval harness (HEAD)
    "6a54104",  # wire compute_confidence into ask_stream pipeline
    "2939a11",  # confidence scoring and legal force integration
    "7f2bfff",  # high-risk classification anchors
    "dd75b68",  # GDPR routing improvements
    "3df53d3",  # GDPR + legal reasoning chains (pre-confidence baseline)
]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _run_eval(case: str, label: str, judge_runs: int) -> dict:
    out_path = Path(tempfile.gettempdir()) / f"crss_q_{label}.json"
    cmd = [
        sys.executable, "scripts/eval_answer_quality.py",
        "--case", case, "--quiet", "--label", label,
        "--judge-runs", str(judge_runs), "--out", str(out_path),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        return {"error": proc.stderr.strip()[-500:] or "eval failed"}
    data = json.loads(out_path.read_text())
    r = data["results"][0]
    return {"score": r["score"], "reliance": r["reliance"],
            "gen_s": r["gen_s"], "judge_s": r["judge_s"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="HQ_001")
    ap.add_argument("--commits", nargs="+", default=DEFAULT_COMMITS)
    ap.add_argument("--judge-runs", type=int, default=1)
    args = ap.parse_args()

    orig_branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))

    rows: list[tuple[str, str, dict]] = []
    stashed = False
    try:
        if dirty:
            _git("stash", "push", "-m", "crss-bisect-quality")
            stashed = True
            print("Stashed uncommitted tracked changes.", flush=True)

        for sha in args.commits:
            subj = _git("log", "-1", "--format=%s", sha)[:55]
            print(f"\n>>> {sha}  {subj}", flush=True)
            _git("checkout", "-q", sha)
            res = _run_eval(args.case, sha, args.judge_runs)
            if "error" in res:
                print(f"    ERROR: {res['error']}", flush=True)
            else:
                print(f"    score={res['score']}/10  [{res['reliance']}]  "
                      f"(gen {res['gen_s']}s, judge {res['judge_s']}s)", flush=True)
            rows.append((sha, subj, res))
    finally:
        print(f"\nRestoring {orig_branch} …", flush=True)
        _git("checkout", "-q", orig_branch)
        if stashed:
            _git("stash", "pop")
            print("Restored stashed changes.", flush=True)

    # Summary table (newest -> oldest, as swept)
    print("\n" + "=" * 72)
    print(f"{'commit':<10}{'score':<8}{'subject'}")
    print("-" * 72)
    for sha, subj, res in rows:
        score = res.get("score")
        score_str = f"{score}/10" if score is not None else "ERROR"
        print(f"{sha:<10}{score_str:<8}{subj}")
    print("=" * 72)
    print("Tip: a downward step between two adjacent rows brackets the regressing commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
