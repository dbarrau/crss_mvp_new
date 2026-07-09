"""Append-only pilot session log (JSONL).

Captures every question/answer interaction and any tester feedback so a
supervised pilot produces a reviewable dataset instead of terminal-only traces.
Each line is a self-contained JSON record with a ``type`` discriminator
(``"interaction"`` or ``"feedback"``) linked by ``interaction_id``.

Design notes:
- Logging must never break the demo, so every public call swallows its own
  errors and degrades silently.
- The log directory is gitignored (see ``demo/.gitignore``); override the
  location with ``CRSS_PILOT_LOG_DIR``.
- Writes are serialized with a lock since the Flask dev server is threaded.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

_LOG = logging.getLogger("crss.pilot")
_LOCK = threading.Lock()


def _log_dir() -> Path:
    return Path(
        os.environ.get(
            "CRSS_PILOT_LOG_DIR",
            str(Path(__file__).parent / "pilot_logs"),
        )
    )


def _log_path() -> Path:
    return _log_dir() / "sessions.jsonl"


def new_interaction_id() -> str:
    return uuid.uuid4().hex


def _append(record: dict) -> None:
    """Serialize *record* as one JSON line. Never raises."""
    try:
        path = _log_path()
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # logging must never break a request
        _LOG.exception("Failed to write pilot log record")


def log_interaction(
    *,
    interaction_id: str,
    session_id: str | None,
    question: str,
    answer: str,
    k: int,
    confidence: dict | None = None,
    history_len: int = 0,
    error: str | None = None,
) -> None:
    """Record one completed (or failed) Q/A turn."""
    _append(
        {
            "type": "interaction",
            "ts": datetime.now(timezone.utc).isoformat(),
            "interaction_id": interaction_id,
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "k": k,
            "history_len": history_len,
            "confidence_score": (confidence or {}).get("score"),
            "confidence_level": (confidence or {}).get("level"),
            "error": error,
        }
    )


def log_feedback(
    *,
    interaction_id: str,
    session_id: str | None,
    rating: str,
    comment: str | None,
) -> None:
    """Record a tester's rating (``up``/``down``) and optional comment."""
    _append(
        {
            "type": "feedback",
            "ts": datetime.now(timezone.utc).isoformat(),
            "interaction_id": interaction_id,
            "session_id": session_id,
            "rating": rating,
            "comment": (comment or "").strip() or None,
        }
    )
