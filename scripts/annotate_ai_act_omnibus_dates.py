"""EPHEMERAL: correct the AI Act Article 113 high-risk application date in-graph.

Systemic-but-temporary bridge for the consolidation-lag gap. The AI Act's
Article 113 dates were amended by Regulation (EU) 2026/1744 (Digital Omnibus),
but no consolidated AI Act exists on EUR-Lex yet, so the graph still holds the
pre-amendment text: Article 113(3)(c) reads "Article 6(1) ... shall apply from
2 August 2027" (Annex I / product-embedded high-risk AI, e.g. medical devices).

Rather than guess from the *question* whether to correct (the old three-layer
bridge did, and missed classification questions like "is this high-risk?" that
never mention a date but whose answer still surfaces one), this attaches the
correction to the stale *data*: it REPLACES every occurrence of the superseded
date in the Article 113 subtree's rendered text fields (`text` -> child
``raw_text``; ``text_for_analysis`` -> parent ``article_text``) with the amended
date followed by a bracketed editorial note. Wherever the stale date would reach
the LLM context, the operative clause now reads the amended date — no trigger, no
question-gate.

Why replace (not merely annotate): a model doing legal analysis reports the
black-letter operative clause and treats a bracketed gloss as an ignorable
footnote (observed: with an inserted note it still stated the 2027 date). Making
2028 the operative reading removes that dependence. This is the correction the
consolidated AI Act will make natively; the bracket records the provenance. Only
the Article 113 subtree is touched, so Article 111's unrelated "2 August 2027"
transition references are left intact.

A companion prompt directive (application/_prompts.py, gated on the "[AMENDED"
marker appearing in the assembled context — data-driven, not question-gated)
tells the model the amended value is controlling.

Display-only for the graph render; nothing else in the pipeline depends on it.
Idempotent (skips fields already carrying the marker). A re-ingest wipes it —
re-run afterwards. DELETE this script and the ``_flag_superseded_ai_act_dates``
backstop once the consolidated AI Act is ingested via ``source_celex`` (Article
113 will then carry 2 August 2028 natively).

    python scripts/annotate_ai_act_omnibus_dates.py            # apply
    python scripts/annotate_ai_act_omnibus_dates.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import os
import re

from dotenv import load_dotenv
from neo4j import GraphDatabase

_AI_ACT = "32024R1689"
_ART_113_PREFIX = "32024R1689_art_113"
_FIELDS = ("text", "text_for_analysis")

# The superseded date, tolerant of the non-breaking spaces EUR-Lex uses.
_STALE_DATE_RE = re.compile(r"2[\s ]+August[\s ]+2027")

# The whole date phrase is REPLACED (not merely annotated) so the operative clause
# reads the amended date — a model reporting the black-letter text then gets 2028
# even if it ignores the bracket. The marker doubles as the idempotency guard and is
# what the prompt-side directive keys on.
_MARKER = "[AMENDED by Reg (EU) 2026/1744"
_REPLACEMENT = (
    "2 August 2028 " + _MARKER + " (Digital Omnibus); the original 2 August 2027 "
    "is superseded. Current application dates: high-risk AI under Article 6(1)/Annex I "
    "(product-embedded, e.g. medical-device AI) = 2 August 2028; under Article 6(2)/"
    "Annex III = 2 December 2027]"
)
# Matches the whole editorial bracket (no nested ']'), for --revert.
_BRACKET_RE = re.compile(r" ?\[AMENDED by Reg \(EU\) 2026/1744[^\]]*\]")


def _annotate(value: str | None) -> tuple[str | None, int]:
    """Replace each stale-date occurrence with the amended date + note."""
    if not value or _MARKER in value or not _STALE_DATE_RE.search(value):
        return value, 0
    count = 0

    def _sub(m: re.Match) -> str:
        nonlocal count
        count += 1
        return _REPLACEMENT

    return _STALE_DATE_RE.sub(_sub, value), count


def _revert(value: str | None) -> tuple[str | None, int]:
    """Strip a previously-inserted note bracket (restores insert-style originals)."""
    if not value or _MARKER not in value:
        return value, 0
    new = _BRACKET_RE.sub("", value)
    return new, 1 if new != value else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only, no writes")
    ap.add_argument("--revert", action="store_true",
                    help="strip the editorial bracket instead of applying it")
    args = ap.parse_args()

    transform = _revert if args.revert else _annotate
    load_dotenv()
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    total_fields = total_hits = 0
    try:
        with driver.session() as session:
            rows = session.run(
                "MATCH (n:Provision {celex:$celex}) "
                "WHERE n.id STARTS WITH $prefix "
                "RETURN n.id AS id, n.text AS text, n.text_for_analysis AS text_for_analysis",
                celex=_AI_ACT, prefix=_ART_113_PREFIX,
            ).data()

            for row in rows:
                for field in _FIELDS:
                    new_val, count = transform(row.get(field))
                    if count == 0:
                        continue
                    total_fields += 1
                    total_hits += count
                    print(f"  {row['id']}.{field}: {count}")
                    if not args.dry_run:
                        session.run(
                            f"MATCH (n {{id:$id}}) SET n.`{field}` = $val",
                            id=row["id"], val=new_val,
                        )

        verb = "reverted" if args.revert else "annotated"
        if total_fields == 0:
            print(f"Nothing to {'revert' if args.revert else 'do'} "
                  f"(already in the target state).")
        else:
            print(f"{'would ' if args.dry_run else ''}{verb} {total_fields} field(s).")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
