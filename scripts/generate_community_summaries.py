#!/usr/bin/env python3
"""Generate LLM summaries for Community nodes and store embeddings.

For each :Community node that lacks a summary, this script:
1. Fetches up to *sample_size* member Provision texts.
2. Asks Mistral to produce a 2-3 sentence regulatory summary.
3. Encodes the summary with the same SentenceTransformer used for provisions.
4. Writes ``summary_text`` and ``summary_embedding`` back to the Community node.

This is an **offline / index-time** step — run it once after
``build_communities.py`` and re-run whenever community structure is rebuilt.

Quick start::

    python scripts/generate_community_summaries.py

Options::

    --rescan     Re-generate summaries for communities that already have one.
    --batch-size N  Number of provisions to sample per community (default 12).
    --dry-run    Fetch and print summaries without writing to Neo4j.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri
from retrieval._config import PASSAGE_PREFIX as _PASSAGE_PREFIX

logger = logging.getLogger(__name__)

_BATCH = 100
_DEFAULT_SAMPLE = 20


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def _fetch_communities(session, *, rescan: bool, level: int | None = None) -> list[dict]:
    conditions = []
    if not rescan:
        conditions.append("c.summary_text IS NULL")
    if level is not None:
        conditions.append(f"c.level = {level}")
    else:
        # By default skip Level-1 nodes — they need a separate summary pass
        conditions.append("(c.level IS NULL OR c.level = 0)")
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    return session.run(
        f"""
        MATCH (c:Community)
        {where}
        RETURN c.id AS id, c.member_count AS member_count,
               c.regulations AS regulations, c.level AS level
        ORDER BY c.member_count DESC
        """,
    ).data()


def _fetch_member_texts(session, community_id: str, sample_size: int) -> list[str]:
    rows = session.run(
        """
        MATCH (p:Provision)-[:MEMBER_OF]->(c:Community {id: $community_id})
        WHERE p.text_for_analysis IS NOT NULL AND p.text_for_analysis <> ''
          AND p.kind IN ['article', 'annex_section', 'annex_point',
                         'annex_part', 'recital', 'section',
                         'guidance_section', 'guidance_subsection',
                         'paragraph', 'point']
        RETURN p.text_for_analysis AS text, p.kind AS kind,
               coalesce(p.display_ref, p.id) AS ref
        ORDER BY p.hierarchy_depth ASC
        LIMIT $sample_size
        """,
        community_id=community_id,
        sample_size=sample_size,
    ).data()
    # Prefix each excerpt with its exact article reference so the LLM cannot
    # substitute numbers from its training knowledge.
    return [
        f"[{row['ref']}] {row['text']}"
        for row in rows
        if row.get("text")
    ]


def _write_community_summary(
    session,
    community_id: str,
    summary_text: str,
    embedding: list[float],
) -> None:
    session.run(
        """
        MATCH (c:Community {id: $community_id})
        SET c.summary_text      = $summary_text,
            c.summary_embedding = $embedding,
            c.summary_updated_at = datetime()
        """,
        community_id=community_id,
        summary_text=summary_text,
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM = (
    "You are a regulatory analyst specialising in EU law. "
    "You are given a set of regulatory provisions, each prefixed with its exact article reference "
    "in brackets, e.g. [Article 72] or [Article 13(1)]. "
    "Produce a structured summary in the following format:\n"
    "Actor roles: [comma-separated list of actor roles addressed]\n"
    "Tier 1 (<condition or 'all actors'>): <primary obligation, prohibition, or right>\n"
    "Tier 2 (<condition if applicable>): <additional obligation that applies under this condition>\n"
    "(Add more tiers only when genuinely distinct obligation levels exist in the provisions.)\n"
    "Covered provisions: [comma-separated article refs, verbatim from the bracketed references only]\n"
    "Rules: "
    "(1) Only cite article numbers that appear verbatim in the bracketed references — never infer or invent. "
    "(2) Use precise legal language. "
    "(3) If only one obligation tier exists, output Tier 1 only — do not fabricate tiers. "
    "(4) Output ONLY the structured summary, no headings, no preamble."
)

_L1_SUMMARY_SYSTEM = (
    "You are a regulatory analyst specialising in EU law. "
    "The following are structured summaries of individual community clusters within "
    "a single regulatory chapter. Each summary lists actor roles, obligation tiers, and covered provisions. "
    "Synthesise them into one paragraph (3-5 sentences) describing: "
    "(1) the chapter's overall scope; "
    "(2) which actor roles it addresses; "
    "(3) all distinct obligation tiers — explicitly naming any conditions that trigger "
    "higher-tier obligations (e.g. systemic-risk thresholds, high-risk AI classification, "
    "GPAI model designation). "
    "Use precise legal language. Output ONLY the synthesised summary, no headings."
)


def _fetch_l1_member_summaries(session, community_id: str) -> list[str]:
    """Fetch the summary_text values of the Level-0 communities that belong
    to this Level-1 community (via parent_community_id on Level-0 nodes)."""
    rows = session.run(
        """
        MATCH (l0:Community {parent_community_id: $l1_id})
        WHERE l0.summary_text IS NOT NULL AND l0.summary_text <> ''
        RETURN l0.summary_text AS summary_text
        ORDER BY l0.member_count DESC
        """,
        l1_id=community_id,
    ).data()
    return [r["summary_text"] for r in rows if r.get("summary_text")]


def _complete_with_retry(
    client, messages, *, max_tokens: int, on_rate_limit=None, on_success=None
) -> str:
    """Call Mistral with bounded retries that survive transient network faults.

    Retries *any* failure with exponential backoff (5, 10, 20, 40 s); rate limits
    additionally drive the caller's adaptive throttle via ``on_rate_limit``. After
    5 attempts the community is skipped (returns "") rather than crashing the
    batch — a single dropped connection mid-run (observed: httpx
    ``RemoteProtocolError`` / server-disconnect at community 90/132) must not
    discard the summaries already written or halt the remaining ones. The run is
    idempotent, so a skipped community is picked up on the next invocation.
    """
    for attempt in range(5):
        try:
            resp = client.chat.complete(
                model=os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            if on_success:
                on_success()
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if ("429" in str(exc) or "rate" in str(exc).lower()) and on_rate_limit:
                on_rate_limit()
            if attempt == 4:
                logger.error(
                    "Mistral call failed after 5 attempts (%s) — skipping community.",
                    type(exc).__name__,
                )
                return ""
            wait = 2 ** attempt * 5  # 5, 10, 20, 40 s
            logger.warning(
                "Transient Mistral error (%s); retrying in %ds (attempt %d/5).",
                type(exc).__name__, wait, attempt + 1,
            )
            time.sleep(wait)
    return ""


def _generate_l1_summary(member_summaries: list[str], client, on_rate_limit=None, on_success=None) -> str:
    """Generate a Level-1 summary from the Level-0 member summaries."""
    if not member_summaries:
        return ""
    joined = "\n---\n".join(member_summaries)
    return _complete_with_retry(
        client,
        [
            {"role": "system", "content": _L1_SUMMARY_SYSTEM},
            {"role": "user", "content": joined},
        ],
        max_tokens=280,
        on_rate_limit=on_rate_limit,
        on_success=on_success,
    )


def _generate_summary(texts: list[str], client, on_rate_limit=None, on_success=None) -> str:
    if not texts:
        return ""

    # Truncate each text to avoid token limits; 500 chars per provision is
    # enough for the model to understand the semantic theme and obligation tiers.
    joined = "\n---\n".join(t[:500] for t in texts[:_DEFAULT_SAMPLE])

    return _complete_with_retry(
        client,
        [
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user", "content": joined},
        ],
        max_tokens=350,
        on_rate_limit=on_rate_limit,
        on_success=on_success,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_summaries(
    *,
    rescan: bool,
    batch_size: int,
    dry_run: bool,
    level: int | None = None,
) -> dict[str, int]:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

    uri = _normalize_neo4j_uri(os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    user = os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j"))
    password = os.environ.get("NEO4J_PASSWORD", "password")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    from mistralai.client import Mistral
    # Bound every request so a stalled socket cannot freeze the whole batch (the
    # earlier run hung ~40 min on a timeout-less read before the server dropped
    # it); paired with _complete_with_retry, a timeout becomes a retry, not a hang.
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"], timeout_ms=60000)

    model = SentenceTransformer("intfloat/multilingual-e5-base")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    generated = 0
    skipped = 0
    try:
        with driver.session(database=database) as session:
            communities = _fetch_communities(session, rescan=rescan, level=level)
            total = len(communities)
            logger.info("Communities to summarise: %d (level=%s)", total, level)

            # Adaptive inter-request delay.  Starts at 1 s; doubles on each
            # 429 (up to 30 s max); decays 10 % toward a 1 s floor on each
            # successful call so a transient burst doesn't slow the whole run.
            inter_delay = [1.0]
            _DELAY_FLOOR = 1.0
            _DELAY_CEIL = 30.0

            def _on_rate_limit() -> None:
                inter_delay[0] = min(inter_delay[0] * 2.0, _DELAY_CEIL)
                logger.info(
                    "Rate limit hit; raising inter-request delay to %.1fs.",
                    inter_delay[0],
                )

            def _on_success() -> None:
                inter_delay[0] = max(inter_delay[0] * 0.9, _DELAY_FLOOR)

            for i, community in enumerate(communities, start=1):
                cid = community["id"]
                com_level = community.get("level") or 0

                if com_level == 1:
                    texts = _fetch_l1_member_summaries(session, cid)
                else:
                    texts = _fetch_member_texts(session, cid, batch_size)

                if not texts:
                    logger.warning("Skipping %s: no member provision texts found.", cid)
                    skipped += 1
                    continue

                summary = (
                    _generate_l1_summary(texts, client, on_rate_limit=_on_rate_limit, on_success=_on_success)
                    if com_level == 1
                    else _generate_summary(texts, client, on_rate_limit=_on_rate_limit, on_success=_on_success)
                )
                if not summary:
                    logger.warning("Skipping %s: LLM returned empty summary.", cid)
                    skipped += 1
                    continue

                embedding: list[float] = model.encode(
                    _PASSAGE_PREFIX + summary,
                    normalize_embeddings=True,
                ).astype(np.float32).tolist()

                if dry_run:
                    print(f"[{i}/{total}] {cid}: {summary[:120]}...")
                else:
                    _write_community_summary(session, cid, summary, embedding)
                    logger.info("  %d / %d done  delay=%.1fs  %s", i, total, inter_delay[0], cid)

                generated += 1
                time.sleep(inter_delay[0])

    finally:
        driver.close()

    return {"generated": generated, "skipped": skipped, "total": total}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and store LLM summaries for Community nodes.",
    )
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Re-generate summaries for communities that already have one.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_SAMPLE,
        help=f"Provision sample size per community (default: {_DEFAULT_SAMPLE}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summaries without writing to Neo4j.",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=[0, 1],
        default=None,
        help="Restrict to communities of a specific level (0=Louvain, 1=chapter). "
             "Omit to process Level-0 only (default). Use --level 1 after rebuilding "
             "communities to generate chapter-level summaries.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args()
    stats = generate_summaries(
        rescan=args.rescan,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        level=args.level,
    )
    print(
        f"\nSummaries: generated={stats['generated']} "
        f"skipped={stats['skipped']} total={stats['total']}"
    )
