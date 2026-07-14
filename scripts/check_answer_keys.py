#!/usr/bin/env python3
"""Deterministic answer-key checker — grades CRSS answers against a law-grounded
key instead of an LLM's holistic score.

The LLM judge in ``eval_answer_quality.py`` measures "does this sound right to a
model"; this measures objective, checkable correctness:

  * ``must_cite``      — the decisive provisions a correct answer MUST reference
                         (a missing one is a hard fail — this is the false-negative
                         weighting the holistic rubric cannot express).
  * ``must_state``     — the key facts it must get right (timelines, thresholds,
                         the correct route/status). Each fact is a list of accepted
                         phrasings; any one satisfies it.
  * ``must_not_claim`` — trap phrasings that betray an uncorrected false premise.
                         Surfaced as a soft flag (substring matching can't reason
                         about negation), not folded into the pass/fail score.

Keys live inline in ``eval/quality_set.json`` (``answer_key`` field), authored
from the regulations and version-controlled — no LLM, no human testers required.
They are the bridge to (and the seed for) human-verified ground truth.

Usage::

    python scripts/check_answer_keys.py --results eval/quality_postfixes.json
    python scripts/check_answer_keys.py --results <file> --out /tmp/keys.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
QUALITY_SET = ROOT / "eval" / "quality_set.json"

# A fact passes if ≥ this fraction of must_state facts are present (lets an answer
# omit a peripheral nuance while still failing on a missing decisive fact).
_STATE_PASS_THRESHOLD = 0.7


def _cite_pattern(ref: str) -> re.Pattern:
    """Whole-reference regex: 'Article 53' → \\bArticle\\s+53\\b (matches
    'Article 53(1)' but not 'Article 530'); 'Annex IX' → \\bAnnex\\s+IX\\b."""
    parts = ref.split(None, 1)
    if len(parts) == 2:
        keyword, num = parts
        return re.compile(rf"\b{re.escape(keyword)}\s+{re.escape(num)}\b", re.IGNORECASE)
    return re.compile(rf"\b{re.escape(ref)}\b", re.IGNORECASE)


def _fold(text: str) -> str:
    """Normalise orthographic noise that breaks literal substring matching.

    Two sources of false "missing fact" reports, both mirrored from the
    faithfulness checker's normalisation:
    - hyphen/dash family → space ("fundamental-rights" vs "fundamental rights");
    - markdown emphasis markers stripped — CRSS answers bold every provision
      reference by mandate, so "**Article 6(1)**" would otherwise never match a
      key phrase containing "article 6".
    """
    text = re.sub(r"[*_`]", "", text)          # markdown emphasis / code ticks
    text = re.sub(r"[-‐-―−]", " ", text)        # hyphen/dash family
    return re.sub(r"\s+", " ", text).strip()


def check_answer(answer: str, key: dict) -> dict:
    """Score a single answer against its key. Pure; no LLM."""
    low = answer.lower()
    folded = _fold(low)

    # A must_cite element is either a string (required) or a list of acceptable
    # alternatives (any one satisfies it) — e.g. an actor-status transition may be
    # anchored in Article 3(3) (developer = provider from inception) OR Article 25
    # (a deployer/distributor becoming a provider), and both are correct.
    cites = key.get("must_cite", [])
    found_cites: list[str] = []
    missed_cites: list[str] = []
    for req in cites:
        alts = req if isinstance(req, list) else [req]
        label = " or ".join(alts) if isinstance(req, list) else req
        (found_cites if any(_cite_pattern(a).search(answer) for a in alts)
         else missed_cites).append(label)

    states = key.get("must_state", [])
    missed_states: list[str] = []
    found_n = 0
    for fact in states:
        # A fact passes if any accepted phrasing appears, hyphen-folded so an
        # orthographic variant ("machine-readable" vs "machine readable") is
        # not a false miss.
        if any(_fold(alt.lower()) in folded for alt in fact):
            found_n += 1
        else:
            missed_states.append(fact[0])

    violations = [p for p in key.get("must_not_claim", []) if p.lower() in low]

    cite_recall = len(found_cites) / len(cites) if cites else 1.0
    state_recall = found_n / len(states) if states else 1.0
    return {
        "cite_recall": round(cite_recall, 3),
        "state_recall": round(state_recall, 3),
        "missed_cites": missed_cites,
        "missed_states": missed_states,
        "violations": violations,
        # A missing decisive provision is a hard fail; trap flags are advisory.
        "passed": cite_recall == 1.0 and state_recall >= _STATE_PASS_THRESHOLD,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic answer-key checker")
    ap.add_argument("--results", required=True, help="eval results JSON (with answers)")
    ap.add_argument("--out", help="write per-case JSON")
    args = ap.parse_args()

    keyed = {c["id"]: c for c in json.loads(QUALITY_SET.read_text()) if c.get("answer_key")}
    results = {r["id"]: r for r in json.loads(Path(args.results).read_text())["results"]}

    rows: list[dict] = []
    print(f"\nAnswer-key check — {len(keyed)} keyed case(s) vs {args.results}\n")
    for cid, case in keyed.items():
        r = results.get(cid)
        if not r or not r.get("answer"):
            print(f"  {cid}  — no answer in results, skipped")
            continue
        v = check_answer(r["answer"], case["answer_key"])
        rows.append({"id": cid, **v})
        bits = [f"cite={v['cite_recall']:.0%}", f"state={v['state_recall']:.0%}"]
        if v["missed_cites"]:
            bits.append(f"MISSED-CITE={v['missed_cites']}")
        if v["missed_states"]:
            bits.append(f"missed-fact={v['missed_states']}")
        if v["violations"]:
            bits.append(f"⚠TRAP={v['violations']}")
        print(f"  {cid}  {'PASS' if v['passed'] else 'FAIL'}  " + "  ".join(bits))

    if rows:
        print(f"\n{'─'*60}")
        print(f"  mean cite-recall:   {statistics.mean(r['cite_recall'] for r in rows):.0%}")
        print(f"  mean state-recall:  {statistics.mean(r['state_recall'] for r in rows):.0%}")
        print(f"  passing:            {sum(r['passed'] for r in rows)}/{len(rows)}")
        print(f"  trap flags:         {sum(1 for r in rows if r['violations'])}")
        print(f"{'─'*60}\n")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())