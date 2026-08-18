"""Produce a base act's *consolidated* parsed.json from an amending act.

Reads the base regulation's ``parsed.json`` and the amending act's raw HTML,
applies the amendments (Stage 1 parse + Stage 2 apply), and writes a *separate*
``parsed.consolidated.json`` beside the source — which the loader can prefer via
``--consolidated``.  The source ``parsed.json`` is never touched, so a plain
rebuild resets everything; this is the interim bridge until EUR-Lex publishes the
official consolidated act (then ``source_celex`` supersedes it).

    python -m consolidation.build 32024R1689 --amender 32026R1744
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from consolidation.amendment_parser import parse_amending_regulation
from consolidation.applier import OpResult, consolidate

_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "legislation"


def _parsed_path(celex: str, lang: str, root: Path) -> Path:
    return root / celex / lang / "parsed.json"


def _consolidated_path(celex: str, lang: str, root: Path) -> Path:
    return root / celex / lang / "parsed.consolidated.json"


def build_consolidated(base_celex: str, amender_celex: str, lang: str = "EN",
                       data_root: Optional[Path] = None) -> tuple[dict, List[OpResult]]:
    """Return ``(consolidated_data, report)`` for the base act.

    ``consolidated_data`` mirrors the base ``parsed.json`` document (same
    top-level keys) with its ``provisions`` replaced by the consolidated tree and
    any relations dangling off deleted nodes dropped."""
    root = Path(data_root) if data_root else _DATA_ROOT
    data = json.loads(_parsed_path(base_celex, lang, root).read_text(encoding="utf-8"))

    instructions = parse_amending_regulation(amender_celex, lang, root)
    consolidated, report = consolidate(data["provisions"], instructions, amender_celex)

    live_ids = {n["id"] for n in consolidated}
    relations = [r for r in data.get("relations", [])
                 if r.get("source") in live_ids and (
                     r.get("target", "").startswith("ext_") or r.get("target") in live_ids)]

    out = dict(data)
    out["provisions"] = consolidated
    out["relations"] = relations
    out["consolidated_from"] = amender_celex
    return out, report


def write_consolidated(base_celex: str, amender_celex: str, lang: str = "EN",
                       data_root: Optional[Path] = None) -> tuple[Path, List[OpResult]]:
    root = Path(data_root) if data_root else _DATA_ROOT
    out, report = build_consolidated(base_celex, amender_celex, lang, root)
    path = _consolidated_path(base_celex, lang, root)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, report


def _summarise(report: List[OpResult]) -> str:
    from collections import Counter
    c = Counter(r.status for r in report)
    return " | ".join(f"{k}={v}" for k, v in sorted(c.items()))


if __name__ == "__main__":                                   # pragma: no cover
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Write a consolidated parsed.json for a base act.")
    ap.add_argument("base_celex", help="the act being amended, e.g. 32024R1689")
    ap.add_argument("--amender", required=True, help="the amending act, e.g. 32026R1744")
    ap.add_argument("--lang", default="EN")
    args = ap.parse_args()

    path, report = write_consolidated(args.base_celex, args.amender, args.lang)
    deferred = [r for r in report if r.status != "applied"]
    print(f"[consolidation] {args.base_celex} ← {args.amender}: {_summarise(report)}")
    for r in deferred:
        print(f"  [{r.status}] ({r.point_num}) {r.op} {r.target_ref}: {r.detail}", file=sys.stderr)
    print(f"[consolidation] wrote {path}")
