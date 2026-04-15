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
import logging

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

logging.basicConfig(level=logging.WARNING)

from application.agent import ask
from export import generate_markdown
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
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    k = body.get("k", 5)
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 5

    try:
        answer = ask(question, retriever, k=k)
        return jsonify({"answer": answer})
    except Exception as exc:
        logging.exception("Error in ask()")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/debug", methods=["POST"])
def api_debug():
    """Return raw retrieval metadata (scores, refs) without calling the LLM."""
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    k = body.get("k", 5)
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 5

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


# ---------------------------------------------------------------------------
# Export endpoints
# ---------------------------------------------------------------------------

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
