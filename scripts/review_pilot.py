#!/usr/bin/env python3
"""Print a readable digest of a pilot session log.

The demo server appends one JSON line per question/answer turn and per tester
feedback vote to ``demo/pilot_logs/sessions.jsonl`` (see ``demo/logging_store.py``).
This script joins those records by ``interaction_id`` and prints a per-session
summary — questions asked, confidence CRSS reported, and the tester's rating +
comment — so you can review a supervised session without reading raw JSONL.

Usage::

    python scripts/review_pilot.py                 # digest of the default log
    python scripts/review_pilot.py --log path.jsonl
    python scripts/review_pilot.py --downvotes-only # only answers rated 👎

Honors ``CRSS_PILOT_LOG_DIR`` (same env var the server uses).
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def _default_log() -> Path:
    base = os.environ.get(
        "CRSS_PILOT_LOG_DIR",
        str(Path(__file__).resolve().parent.parent / "demo" / "pilot_logs"),
    )
    return Path(base) / "sessions.jsonl"


def _load(path: Path):
    """Return (interactions_by_id, latest_feedback_by_id) from the JSONL log."""
    interactions: dict[str, dict] = {}
    feedback: dict[str, dict] = {}  # last vote wins (vote → vote+comment)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = rec.get("interaction_id")
            if not iid:
                continue
            if rec.get("type") == "interaction":
                interactions[iid] = rec
            elif rec.get("type") == "feedback":
                feedback[iid] = rec
    return interactions, feedback


_RATING_ICON = {"up": "👍", "down": "👎"}


def _conf(rec: dict) -> str:
    level = rec.get("confidence_level")
    score = rec.get("confidence_score")
    if level is None and score is None:
        return "—"
    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
    return f"{level or '?'} ({score_str})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Digest a CRSS pilot session log")
    parser.add_argument("--log", type=Path, default=_default_log())
    parser.add_argument(
        "--downvotes-only",
        action="store_true",
        help="show only answers the tester rated 👎",
    )
    args = parser.parse_args()

    if not args.log.exists():
        print(f"No log found at {args.log}")
        print("Run a pilot session first (python demo/server.py), then re-run this.")
        return

    interactions, feedback = _load(args.log)
    if not interactions:
        print(f"Log {args.log} has no interactions yet.")
        return

    # Group by session, ordered chronologically within each session.
    by_session: dict[str, list[dict]] = defaultdict(list)
    for rec in interactions.values():
        by_session[rec.get("session_id") or "(no session id)"].append(rec)
    for recs in by_session.values():
        recs.sort(key=lambda r: r.get("ts", ""))

    up = down = rated = errored = shown = 0

    for session_id, recs in sorted(by_session.items(), key=lambda kv: kv[1][0].get("ts", "")):
        printed_header = False
        for i, rec in enumerate(recs, 1):
            iid = rec["interaction_id"]
            fb = feedback.get(iid)
            rating = (fb or {}).get("rating")
            if rating == "up":
                up += 1; rated += 1
            elif rating == "down":
                down += 1; rated += 1
            if rec.get("error"):
                errored += 1

            if args.downvotes_only and rating != "down":
                continue

            if not printed_header:
                print(f"\n═══ Session {session_id}  ·  {recs[0].get('ts', '')[:19]} ═══")
                printed_header = True
            shown += 1

            icon = _RATING_ICON.get(rating, "  ")
            print(f"\n  Q{i} {icon}  [{_conf(rec)}]")
            print(f"     {rec.get('question', '').strip()}")
            if rec.get("error"):
                print(f"     ⚠ ERROR: {rec['error']}")
            comment = (fb or {}).get("comment")
            if comment:
                print(f"     💬 {comment}")

    print("\n" + "─" * 60)
    total = len(interactions)
    print(
        f"Sessions: {len(by_session)}   Questions: {total}   "
        f"Rated: {rated} (👍 {up} / 👎 {down})   Errors: {errored}"
    )
    if args.downvotes_only:
        print(f"Shown (👎 only): {shown}")


if __name__ == "__main__":
    main()