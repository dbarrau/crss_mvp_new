from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from canonicalization.annex_normalizer import extract_annex_blocks_from_provisions


def load_provisions(path: Path) -> List[Dict[str, Any]]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	return data.get("provisions", [])


def pick_default_annex_id(provisions: List[Dict[str, Any]]) -> Optional[str]:
	for p in provisions:
		if p.get("kind") == "annex":
			return p.get("id")
	return None


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(
		description="Linearize annex content from a parsed.json using AnnexBlock representation.",
	)
	parser.add_argument(
		"parsed_json",
		type=Path,
		help="Path to parsed.json produced by the ingestion pipeline.",
	)
	parser.add_argument(
		"--annex-id",
		dest="annex_id",
		type=str,
		help="Full provision id of the annex to linearize (e.g. 32024R1689_anx_XIII).",
	)

	args = parser.parse_args(argv)

	provisions = load_provisions(args.parsed_json)
	if not provisions:
		print("No provisions found in parsed JSON.", file=sys.stderr)
		return 1

	annex_id = args.annex_id or pick_default_annex_id(provisions)
	if not annex_id:
		print("No annex id provided and no annex nodes found.", file=sys.stderr)
		return 1

	blocks = extract_annex_blocks_from_provisions(provisions, annex_id)
	if not blocks:
		print(f"No blocks extracted for annex {annex_id}.", file=sys.stderr)
		return 1

	print(f"Annex: {annex_id}")
	print("=" * (7 + len(annex_id)))
	for idx, block in enumerate(blocks, start=1):
		ids = ",".join(block.source_ids)
		print(f"{idx:03d} [{block.kind}] label={block.label!r} depth={block.depth} ids={ids}")
		if block.text:
			print(f"    {block.text}")

	return 0


if __name__ == "__main__":  # pragma: no cover
	sys.exit(main())
