"""Batch-embed all provisions with text_for_analysis and store in Neo4j.

Embeddings are stored as ``LIST<FLOAT>`` properties on :Provision nodes.
Similarity search is performed in-memory via numpy (fast for <10k nodes),
so no Neo4j Vector Index plugin is required.

Usage
-----
    python -m infrastructure.embeddings.batch_embedder

Or via the convenience script::

    python scripts/embed_provisions.py
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
logger = logging.getLogger(__name__)

PASSAGE_PREFIX = "passage: "
MODEL_NAME = "intfloat/multilingual-e5-base"
DIMENSIONS = 768

EMBED_KINDS = {
    "article", "paragraph", "subparagraph", "point", "roman_item",
    "recital", "section",
    "chapter", "annex_part",
    "annex", "annex_section", "annex_subsection",
    "annex_point", "annex_subpoint", "annex_bullet",
    # Guidance (MDCG)
    "guidance_section", "guidance_subsection",
    "guidance_paragraph", "guidance_chart",
}


def run(model_name: str = MODEL_NAME, batch_size: int = 64) -> int:
    """Embed all qualifying provisions and store vectors in Neo4j.

    Returns the number of nodes embedded.
    """
    model = SentenceTransformer(model_name)

    uri = _normalize_neo4j_uri(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    )
    driver = GraphDatabase.driver(
        uri,
        auth=(
            os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j")),
            os.environ.get("NEO4J_PASSWORD", "password"),
        ),
    )
    db = os.environ.get("NEO4J_DATABASE", "neo4j")

    # Fetch all provisions and guidance nodes that have analysable text
    with driver.session(database=db) as s:
        rows = s.run(
            "MATCH (n:Provision) "
            "WHERE n.text_for_analysis IS NOT NULL AND n.kind IN $kinds "
            "RETURN n.id AS id, n.text_for_analysis AS text, "
            "       n.display_path AS display_path "
            "UNION ALL "
            "MATCH (n:Guidance) "
            "WHERE n.text_for_analysis IS NOT NULL AND n.kind IN $kinds "
            "RETURN n.id AS id, n.text_for_analysis AS text, "
            "       n.display_path AS display_path",
            kinds=list(EMBED_KINDS),
        ).data()

    if not rows:
        logger.warning("No nodes found to embed. Is Neo4j loaded?")
        driver.close()
        return 0

    logger.info("Embedding %d provisions with %s …", len(rows), model_name)

    ids = [r["id"] for r in rows]
    texts = [
        PASSAGE_PREFIX + (r["display_path"] + ": " if r.get("display_path") else "") + r["text"]
        for r in rows
    ]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    batch = [
        {"id": id_, "emb": emb.tolist()}
        for id_, emb in zip(ids, embeddings)
    ]

    CHUNK = 500
    with driver.session(database=db) as s:
        for i in range(0, len(batch), CHUNK):
            s.run(
                "UNWIND $batch AS row "
                "OPTIONAL MATCH (p:Provision {id: row.id}) "
                "OPTIONAL MATCH (g:Guidance  {id: row.id}) "
                "WITH row, coalesce(p, g) AS n "
                "WHERE n IS NOT NULL "
                "SET n.embedding = row.emb",
                batch=batch[i : i + CHUNK],
            )
            logger.info("Stored %d / %d", min(i + CHUNK, len(batch)), len(batch))

    driver.close()
    logger.info("Done. %d nodes embedded.", len(ids))
    return len(ids)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run()
