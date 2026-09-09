#!/usr/bin/env python3
"""Coverage audit — keep eval/quality_set.json a governed design, not a list.

The set is a coverage matrix on three axes carried in each question's
``coverage`` block: ``route`` (the router capability), ``regs`` (regulations in
scope) with a ``primary_reg`` bucket, and ``difficulty`` (E/M/H). This audit
reads the declared targets in ``eval/coverage_manifest.json`` and reports where
the live set sits against them, so the set cannot silently drift back into
accretion — the coherence analogue of ``audit_answer_key_currency.py``.

Rules (targets live in the manifest, not here):

  FAIL (exit 1)
    * a question's ``coverage`` block is missing or malformed;
    * a capability route in use has fewer than ``route_floor`` questions;
    * a regulation is below its ``reg_floors`` entry (counted by membership —
      a cross-reg question counts toward each reg it touches);
    * a (route x primary_reg x difficulty) cell exceeds ``difficulty_cell_ceiling``.

  REVIEW (soft, never fails)
    * a single-reg (route x primary_reg) cell reaches ``single_reg_aggregate_soft``
      across all difficulties — heavy but possibly all-distinct; a human decides.

Usage::

    python scripts/audit_eval_coverage.py            # human report; exit 1 on any FAIL
    python scripts/audit_eval_coverage.py --json      # machine-readable findings
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUALITY_SET = ROOT / "eval" / "quality_set.json"
MANIFEST = ROOT / "eval" / "coverage_manifest.json"

FAIL = "FAIL"
REVIEW = "REVIEW"


def _finding(sev: str, msg: str) -> dict:
    return {"severity": sev, "message": msg}


def audit(questions: list[dict], manifest: dict) -> list[dict]:
    findings: list[dict] = []
    routes = set(manifest["routes"])
    diffs = set(manifest["difficulties"])
    single_buckets = set(manifest["single_reg_buckets"])

    # --- validate every coverage block up front; a malformed one is a hard fail ---
    valid: list[tuple[str, dict]] = []
    for q in questions:
        qid = q.get("id", "?")
        cov = q.get("coverage")
        if not isinstance(cov, dict):
            findings.append(_finding(FAIL, f"{qid}: no coverage block — cannot govern it."))
            continue
        ok = True
        if cov.get("route") not in routes:
            findings.append(_finding(FAIL, f"{qid}: route {cov.get('route')!r} not in the manifest taxonomy."))
            ok = False
        if cov.get("difficulty") not in diffs:
            findings.append(_finding(FAIL, f"{qid}: difficulty {cov.get('difficulty')!r} not one of {sorted(diffs)}."))
            ok = False
        if not isinstance(cov.get("regs"), list) or not cov["regs"]:
            findings.append(_finding(FAIL, f"{qid}: regs must be a non-empty list."))
            ok = False
        if not cov.get("primary_reg"):
            findings.append(_finding(FAIL, f"{qid}: missing primary_reg bucket."))
            ok = False
        if ok:
            valid.append((qid, cov))

    # --- tallies ---
    route_ct: Counter = Counter(c["route"] for _, c in valid)
    reg_ct: Counter = Counter()          # membership (regs list)
    for _, c in valid:
        reg_ct.update(set(c["regs"]))
    diff_cell: dict = defaultdict(list)  # (route, primary_reg, difficulty) -> [ids]
    agg_cell: dict = defaultdict(list)   # (route, primary_reg) -> [ids]
    for qid, c in valid:
        diff_cell[(c["route"], c["primary_reg"], c["difficulty"])].append(qid)
        agg_cell[(c["route"], c["primary_reg"])].append(qid)

    # --- route floor ---
    for route in sorted(route_ct):
        if route_ct[route] < manifest["route_floor"]:
            findings.append(_finding(
                FAIL, f"route '{route}' has {route_ct[route]} question(s), "
                      f"below the floor of {manifest['route_floor']}."))

    # --- reg floor (membership) ---
    for reg, floor in manifest["reg_floors"].items():
        if reg_ct.get(reg, 0) < floor:
            findings.append(_finding(
                FAIL, f"regulation '{reg}' has {reg_ct.get(reg, 0)} question(s), "
                      f"below its floor of {floor}."))

    # --- difficulty-cell ceiling (single-reg buckets only) ---
    # The ceiling targets redundant identical-difficulty lookups in one regulation;
    # CROSS is where every cross-reg question lands by definition and each spans a
    # distinct reg pair, so it (and the CIR/MDCG/OTHER singletons) are exempt.
    ceil = manifest["difficulty_cell_ceiling"]
    for (route, bucket, d), ids in sorted(diff_cell.items()):
        if bucket in single_buckets and len(ids) > ceil:
            findings.append(_finding(
                FAIL, f"cell {route}x{bucket}x{d} has {len(ids)} questions "
                      f"({', '.join(sorted(ids))}) — over the ceiling of {ceil}."))

    # --- single-reg aggregate soft ceiling ---
    soft = manifest["single_reg_aggregate_soft"]
    for (route, bucket), ids in sorted(agg_cell.items()):
        if bucket in single_buckets and len(ids) >= soft:
            findings.append(_finding(
                REVIEW, f"cell {route}x{bucket} holds {len(ids)} cases "
                        f"({', '.join(sorted(ids))}) — heavy; confirm each tests something distinct."))

    return findings


def _print_report(questions, manifest, findings):
    fails = [f for f in findings if f["severity"] == FAIL]
    review = [f for f in findings if f["severity"] == REVIEW]
    route_ct = Counter(q["coverage"]["route"] for q in questions if isinstance(q.get("coverage"), dict))
    reg_ct: Counter = Counter()
    for q in questions:
        c = q.get("coverage")
        if isinstance(c, dict):
            reg_ct.update(set(c.get("regs", [])))

    print(f"\nEval coverage audit — {len(questions)} questions\n")
    print("  routes:", "  ".join(f"{r}={route_ct[r]}" for r in manifest["routes"]))
    print("  regs:  ", "  ".join(f"{r}={reg_ct.get(r, 0)}" for r in manifest["reg_floors"]))
    print()
    if not fails and not review:
        print("  ✓ the set matches its declared coverage design.\n")
    for f in fails:
        print(f"  ✗ [FAIL]   {f['message']}")
    if fails and review:
        print()
    for f in review:
        print(f"  ⚠ [REVIEW] {f['message']}")
    print(f"\n  {len(fails)} failing, {len(review)} to review.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", type=Path, default=QUALITY_SET)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    questions = json.loads(args.set.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    findings = audit(questions, manifest)

    if args.json:
        print(json.dumps(findings, indent=2))
    else:
        _print_report(questions, manifest, findings)

    return 1 if any(f["severity"] == FAIL for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
