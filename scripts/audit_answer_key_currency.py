#!/usr/bin/env python3
"""Answer-key currency audit — catch eval keys that encode superseded law.

When an amending act (the Digital Omnibus, Reg (EU) 2026/1744, and any future
one) enters the corpus, the consolidated graph moves to *current* law and the
runtime guards (``application/_date_currency``, ``application/_superseded``)
start rewriting/flagging pre-amendment dates and deleted-provision citations.
Any answer key in ``eval/quality_set.json`` that still asserts the OLD date, the
OLD in/out-of-force status, or penalises a NOW-REAL inserted provision as a
"fabrication" becomes a guaranteed false fail — the system correctly produces
current law and the stale key marks it wrong (this is exactly how HQ_038's high-
risk date and HQ_039's "Article 4a is fabricated" rotted after the Omnibus).

This is the standing sweep that makes that non-silent. It reuses the SAME
authoritative records the runtime guards read — no second source of truth:

  * ``domain.ontology.omnibus_date_changes.OMNIBUS_DATE_CHANGES``
        every superseded→current application-date move, with the scope regexes
        that pin it to the obligation it governs;
  * ``domain.ontology.superseded_provisions.SUPERSEDED``
        every provision an amending act DELETED;
  * inserted articles, derived from each amender's parsed amendment
        (``data/legislation/<amender>/EN/parsed.json``), driven by
        ``consolidation_plan()`` so a future amending act is covered for free.

Findings are two severities:
  STALE  — a hard conflict with current law (fails CI; exit 1).
  REVIEW — a soft signal a human should eyeball (does not fail CI).

Usage::

    python scripts/audit_answer_key_currency.py            # human report; exit 1 on STALE
    python scripts/audit_answer_key_currency.py --json      # machine-readable findings
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain.legislation_catalog import consolidation_plan  # noqa: E402
from domain.ontology.omnibus_date_changes import OMNIBUS_DATE_CHANGES  # noqa: E402
from domain.ontology.superseded_provisions import SUPERSEDED  # noqa: E402

QUALITY_SET = ROOT / "eval" / "quality_set.json"

STALE = "STALE"
REVIEW = "REVIEW"

# "Article 4a", "article 4A", "Art 4a" → normalised ref "article 4a".
_ARTICLE_RE = re.compile(r"\bart(?:icle)?\.?\s*(\d+[a-z]?)\b", re.IGNORECASE)


def _norm_article(text: str) -> str | None:
    m = _ARTICLE_RE.search(text)
    return f"article {m.group(1).lower()}" if m else None


def _flatten(node) -> list[str]:
    """All leaf strings in a nested must_cite / must_state / must_not_claim."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        out: list[str] = []
        for item in node:
            out.extend(_flatten(item))
        return out
    return []


def _inserted_articles() -> dict[str, set[str]]:
    """``{amended_base_celex: {'article 4a', ...}}`` — articles an amending act
    INSERTS into the base act, parsed from the amender's amendment text.

    EUR-Lex amendment instructions read "the following Article 4a is inserted".
    A newly-inserted article must not be treated as a fabrication by a key that
    was written before the amendment (HQ_039's "Article 4a"). Missing parsed data
    (``data/`` is gitignored) degrades to an empty set with a warning, not a crash.
    """
    inserted: dict[str, set[str]] = {}
    insert_re = re.compile(
        r"insert[^.]{0,80}?['‘’\"]?Article\s+(\d+[a-z])\b", re.IGNORECASE
    )
    for base_celex, amender_celex in consolidation_plan():
        parsed = ROOT / "data" / "legislation" / amender_celex / "EN" / "parsed.json"
        if not parsed.exists():
            print(
                f"  ! parsed amendment for {amender_celex} not found "
                f"({parsed.relative_to(ROOT)}); insertion check skipped for it. "
                f"Run the ingest pipeline to enable it.",
                file=sys.stderr,
            )
            continue
        data = json.loads(parsed.read_text(encoding="utf-8"))
        blob = " ".join((p.get("text") or "") for p in data.get("provisions", []))
        refs = {f"article {n.lower()}" for n in insert_re.findall(blob)}
        inserted.setdefault(base_celex, set()).update(refs)
    return inserted


def _haystack(q: dict) -> str:
    """Everything a scope regex may match against: question + notes + the key."""
    return " ".join(
        [
            q.get("question", ""),
            q.get("judge_notes", ""),
            json.dumps(q.get("answer_key", {})),
        ]
    ).lower()


def _finding(qid: str, severity: str, msg: str) -> dict:
    return {"id": qid, "severity": severity, "message": msg}


def audit(questions: list[dict], inserted: dict[str, set[str]] | None = None) -> list[dict]:
    """Findings for every stale answer key. ``inserted`` (amended-base-celex →
    inserted article refs) defaults to the parse of the amending acts on disk;
    inject it to test the detectors without the gitignored ``data/`` present."""
    findings: list[dict] = []
    if inserted is None:
        inserted = _inserted_articles()
    # Deleted-provision refs are act-specific, but must_cite/must_not_claim are
    # already scoped per question, so a global set is enough to detect a demand
    # to cite dead law or a stale "no longer exists" penalty on a live provision.
    deleted_refs = {e["deleted_ref"].lower() for e in SUPERSEDED}
    all_inserted = {ref for refs in inserted.values() for ref in refs}

    for q in questions:
        qid = q.get("id", "?")
        key = q.get("answer_key", {})
        hay = _haystack(q)
        must_cite = _flatten(key.get("must_cite", []))
        must_state = key.get("must_state", [])
        must_not = _flatten(key.get("must_not_claim", []))

        # --- Check A: a must_state fact requires a superseded date IN its scope ---
        for dc in OMNIBUS_DATE_CHANGES:
            in_scope = any(re.search(rx, hay) for rx in dc.scope_res)
            vetoed = any(re.search(rx, hay) for rx in dc.scope_exclude_res)
            if not in_scope or vetoed:
                continue
            for group in must_state:
                phrasings = _flatten(group)
                if any(dc.superseded.lower() in p.lower() for p in phrasings):
                    findings.append(
                        _finding(
                            qid,
                            STALE,
                            f"must_state requires the superseded date "
                            f"'{dc.superseded}' in a scope the Omnibus moved to "
                            f"'{dc.current}' ({dc.provision}). Update the fact to "
                            f"the current date.",
                        )
                    )

        # --- Check B: must_cite REQUIRES a deleted provision ---
        # Match only when the required citation IS the deleted ref or something
        # more specific (``dref in c``) — never the reverse. A deleted entry is a
        # sub-provision ("Article 10(5)", "Annex VIII, Section B, point 7") whose
        # PARENT ("Article 10", "Annex VIII") is still live; flagging the parent
        # would be wrong, and the reverse test also mis-hit "Annex VII" as a
        # substring of the deleted "Annex VIII, …".
        for cite in must_cite:
            c = cite.lower()
            for dref in deleted_refs:
                if dref in c:
                    findings.append(
                        _finding(
                            qid,
                            STALE,
                            f"must_cite requires '{cite}', a provision deleted by "
                            f"an amending act (see superseded_provisions). A "
                            f"correct current-law answer cannot cite it.",
                        )
                    )

        # --- Check C: must_not_claim penalises a NOW-REAL inserted provision ---
        for phrase in must_not:
            ref = _norm_article(phrase)
            if ref and ref in all_inserted:
                findings.append(
                    _finding(
                        qid,
                        STALE,
                        f"must_not_claim treats '{phrase}' as a fabrication, but "
                        f"an amending act INSERTED {ref.capitalize()} — it is now real "
                        f"law. Remove it from must_not_claim (keep genuinely "
                        f"non-existent draft refs).",
                    )
                )

        # --- Check D (soft): a superseded date appears anywhere in the key ---
        # Scope-independent, so REVIEW not STALE: it catches loose keys (e.g. a
        # date named without the high-risk marker) a human should eyeball.
        key_blob = json.dumps(key).lower()
        for dc in OMNIBUS_DATE_CHANGES:
            if dc.superseded.lower() in key_blob and not any(
                f["id"] == qid and dc.current in f["message"] for f in findings
            ):
                findings.append(
                    _finding(
                        qid,
                        REVIEW,
                        f"answer_key mentions '{dc.superseded}', which the Omnibus "
                        f"moved to '{dc.current}' for {dc.provision}. Confirm the "
                        f"context is one the change does NOT touch (e.g. the "
                        f"unchanged general application date).",
                    )
                )
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", type=Path, default=QUALITY_SET, help="quality set JSON")
    ap.add_argument("--json", action="store_true", help="emit findings as JSON")
    args = ap.parse_args()

    questions = json.loads(args.set.read_text(encoding="utf-8"))
    findings = audit(questions)

    if args.json:
        print(json.dumps(findings, indent=2))
    else:
        stale = [f for f in findings if f["severity"] == STALE]
        review = [f for f in findings if f["severity"] == REVIEW]
        try:
            shown = args.set.resolve().relative_to(ROOT)
        except ValueError:
            shown = args.set
        print(f"\nAnswer-key currency audit — {shown} "
              f"({len(questions)} questions)\n")
        if not findings:
            print("  ✓ no stale keys — every answer_key is consistent with "
                  "current consolidated law.\n")
        for f in stale:
            print(f"  ✗ [STALE]  {f['id']}: {f['message']}")
        if stale and review:
            print()
        for f in review:
            print(f"  ⚠ [REVIEW] {f['id']}: {f['message']}")
        print(f"\n  {len(stale)} stale, {len(review)} to review.\n")

    return 1 if any(f["severity"] == STALE for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
