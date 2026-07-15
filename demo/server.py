#!/usr/bin/env python3
"""Lightweight Flask server wrapping the CRSS regulatory agent.

Usage::

    python demo/server.py          # http://localhost:5050
    python demo/server.py --port 8080
"""
import sys
from pathlib import Path

# Ensure project root is importable (same pattern as chat.py)
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # for export module

import argparse
import json
import logging
import os

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

# INFO-level so the terminal shows live pipeline progress (routing, retrieval,
# context size, confidence) while the LLM prefills.  The large generation model's
# time-to-first-token can be 10-40s on a big prompt; without these logs the
# terminal looks frozen even though the pipeline is working.  Set CRSS_LOG_LEVEL
# to override (e.g. WARNING for quiet, DEBUG for context-size traces).
logging.basicConfig(
    level=getattr(logging, os.environ.get("CRSS_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet the noisy HTTP client libraries that would otherwise flood INFO.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

from application.agent import ask_stream, ask_with_trace
from domain.legislation_catalog import LEGISLATION
from domain.mdcg_catalog import MDCG_DOCUMENTS
from export import generate_markdown
from logging_store import log_feedback, log_interaction, new_interaction_id
from retrieval.graph_retriever import GraphRetriever

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"
app = Flask(__name__)

# Single retriever instance, initialised once at startup
retriever: GraphRetriever | None = None


@app.route("/")
def index():
    # no-cache so an edited index.html (markdown renderer, CSS) always takes
    # effect on refresh — otherwise the browser serves a stale cached page and
    # front-end fixes appear not to work.
    resp = send_from_directory(STATIC_DIR, "index.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/ask", methods=["POST"])
def api_ask():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    k = body.get("k", 20)
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 20

    history = _parse_history(body)
    session_id = _session_id(body)
    interaction_id = new_interaction_id()

    try:
        result = ask_with_trace(question, retriever, k=k, history=history)
        log_interaction(
            interaction_id=interaction_id,
            session_id=session_id,
            question=question,
            answer=result["answer"],
            k=k,
            history_len=len(history or []),
        )
        return jsonify({
            "answer": result["answer"],
            "audit_trace": result.get("audit_trace"),
            "interaction_id": interaction_id,
        })
    except Exception as exc:
        logging.exception("Error in ask_with_trace()")
        log_interaction(
            interaction_id=interaction_id,
            session_id=session_id,
            question=question,
            answer="",
            k=k,
            history_len=len(history or []),
            error=str(exc),
        )
        return jsonify({"error": str(exc)}), 500


@app.route("/api/ask/stream", methods=["POST"])
def api_ask_stream():
    """Stream pipeline steps + LLM tokens as Server-Sent Events."""
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    k = body.get("k", 20)
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 20

    history = _parse_history(body)
    session_id = _session_id(body)
    interaction_id = new_interaction_id()

    def generate():
        # Capture what flows past so the turn can be logged for pilot review.
        answer = ""
        confidence: dict | None = None
        error: str | None = None
        try:
            for event in ask_stream(question, retriever, k=k, history=history):
                if event.get("type") == "confidence":
                    confidence = {"score": event.get("score"), "level": event.get("level")}
                elif event.get("type") == "done":
                    answer = event.get("answer") or answer
                    # Hand the client the id it needs to attach feedback.
                    event = {**event, "interaction_id": interaction_id}
                elif event.get("type") == "error":
                    error = event.get("message")
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logging.exception("Error in ask_stream()")
            error = str(exc)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            log_interaction(
                interaction_id=interaction_id,
                session_id=session_id,
                question=question,
                answer=answer,
                k=k,
                confidence=confidence,
                history_len=len(history or []),
                error=error,
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/debug", methods=["POST"])
def api_debug():
    """Return raw retrieval metadata (scores, refs) without calling the LLM."""
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    k = body.get("k", 20)
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 20

    try:
        provisions = retriever.retrieve(question, k=k)
        items = []
        for p in provisions:
            items.append({
                "ref": p.get("article_ref", "?"),
                "regulation": p.get("regulation", ""),
                "score": round(p.get("score", 0), 4),
                "children": len(p.get("children") or []),
                "cited": len(p.get("cited_provisions") or []),
            })
        return jsonify({"provisions": items})
    except Exception as exc:
        logging.exception("Error in retrieve()")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/legislation", methods=["GET"])
def api_legislation():
    """Return the corpus catalog, sourced from the domain catalogs.

    The demo's "covered corpus" used to be a hardcoded pill list in the frontend,
    which silently drifted from what is actually ingested (it was missing GDPR and
    the implementing regulation). Serving the catalog directly keeps the demo in
    sync with ``domain/legislation_catalog.py`` + ``domain/mdcg_catalog.py`` — add
    a document there and it appears here automatically.
    """
    legislation = [
        {
            "celex": celex,
            "name": meta.get("name", celex),
            "number": meta.get("number", ""),
            "type": meta.get("type", ""),
        }
        for celex, meta in LEGISLATION.items()
    ]
    guidance = [
        {
            "id": gid,
            "name": meta.get("name", gid),
            "title": meta.get("title", ""),
            "tier": meta.get("tier"),
        }
        for gid, meta in MDCG_DOCUMENTS.items()
    ]
    return jsonify({"legislation": legislation, "guidance": guidance})


# ---------------------------------------------------------------------------
# Export endpoints
# ---------------------------------------------------------------------------

def _session_id(body: dict) -> str | None:
    """A client-generated id grouping one tester's turns; optional."""
    sid = body.get("session_id")
    return sid.strip() if isinstance(sid, str) and sid.strip() else None


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Record a tester's thumbs up/down + optional comment for an answer."""
    body = request.get_json(silent=True) or {}
    interaction_id = (body.get("interaction_id") or "").strip()
    rating = (body.get("rating") or "").strip().lower()
    if not interaction_id or rating not in ("up", "down"):
        return jsonify({"error": "interaction_id and rating ('up'|'down') are required"}), 400
    log_feedback(
        interaction_id=interaction_id,
        session_id=_session_id(body),
        rating=rating,
        comment=body.get("comment"),
    )
    return jsonify({"ok": True})


def _parse_history(body: dict) -> list[dict[str, str]] | None:
    """Extract and validate the conversation history from a request body."""
    history = body.get("history")
    if not isinstance(history, list) or not history:
        return None
    valid = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in history
        if isinstance(turn, dict)
        and turn.get("role") in ("user", "agent", "assistant")
        and isinstance(turn.get("content"), str)
    ]
    return valid or None


def _parse_conversation(body: dict) -> list[dict] | None:
    """Validate and return the conversation list, or *None* on error."""
    conv = body.get("conversation")
    if not isinstance(conv, list) or not conv:
        return None
    # Keep only well-formed entries
    return [
        {"role": m["role"], "content": m["content"]}
        for m in conv
        if isinstance(m, dict)
        and m.get("role") in ("user", "agent")
        and isinstance(m.get("content"), str)
    ] or None


@app.route("/api/export/md", methods=["POST"])
def api_export_md():
    body = request.get_json(silent=True) or {}
    conversation = _parse_conversation(body)
    if not conversation:
        return jsonify({"error": "conversation is required (non-empty list)"}), 400
    try:
        md_text = generate_markdown(conversation)
        response = app.response_class(
            md_text,
            mimetype="text/markdown; charset=utf-8",
        )
        response.headers["Content-Disposition"] = (
            'attachment; filename="crss_demo_report.md"'
        )
        return response
    except Exception as exc:
        logging.exception("Error generating Markdown")
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global retriever

    parser = argparse.ArgumentParser(description="CRSS Demo Server")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print("Loading retriever (embeddings + Neo4j)...")
    retriever = GraphRetriever()
    print(f"Ready — opening http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
