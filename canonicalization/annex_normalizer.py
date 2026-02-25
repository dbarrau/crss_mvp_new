from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal


BlockKind = Literal["heading", "paragraph", "point", "subpoint", "bullet"]


@dataclass
class AnnexBlock:
	"""Linearized representation of an annex fragment.

	This is a deterministic, model-agnostic view of annex content that can be
	fed to an LLM-based normalizer. It is intentionally simple and only
	captures what we can derive reliably from existing provisions.
	"""

	kind: BlockKind
	text: str
	label: Optional[str]
	depth: int
	source_ids: List[str]


def extract_annex_blocks_from_provisions(
	provisions: List[Dict], annex_id: str
) -> List[AnnexBlock]:
	"""Produce a linearized view of an annex from existing provisions.

	This function does **not** call any LLM. It simply walks the current
	provision list and extracts a flat sequence of blocks that describe the
	annex text and basic enumeration cues.

	Parameters
	----------
	provisions:
		Full list of provision nodes from parsed.json.
	annex_id:
		The full provision id of the annex node (e.g. "32024R1689_anx_XIII").

	Returns
	-------
	List[AnnexBlock]
		A list of blocks in document order, suitable as input to an
		LLM-based normalizer.
	"""

	# Index provisions by id and parent for quick traversal.
	by_id: Dict[str, Dict] = {p["id"]: p for p in provisions}
	children_by_parent: Dict[str, List[Dict]] = {}
	for p in provisions:
		parent_id = p.get("parent_id")
		if not parent_id:
			continue
		children_by_parent.setdefault(parent_id, []).append(p)

	annex = by_id.get(annex_id)
	if not annex:
		return []

	# We rely on the existing children list on the annex node for order.
	blocks: List[AnnexBlock] = []

	def visit(node: Dict, depth: int) -> None:
		kind = node.get("kind", "")
		text = node.get("text", "") or ""
		label = node.get("number")
		block_kind: BlockKind

		if kind == "annex_section":
			block_kind = "heading"
		elif kind == "annex_point":
			block_kind = "point"
		elif kind == "annex_subpoint":
			block_kind = "subpoint"
		elif kind == "annex_bullet":
			block_kind = "bullet"
		else:
			block_kind = "paragraph"

		blocks.append(
			AnnexBlock(
				kind=block_kind,
				text=text.strip(),
				label=str(label) if label is not None else None,
				depth=depth,
				source_ids=[node["id"]],
			)
		)

		for child_id in node.get("children", []):
			child = by_id.get(child_id)
			if not child:
				continue
			visit(child, depth + 1)

	visit(annex, depth=0)
	return blocks


def normalize_annex_with_llm(
	blocks: List[AnnexBlock],
	celex: str,
	annex_number: Optional[str] = None,
	model_name: Optional[str] = None,
) -> List[Dict]:
	"""Placeholder for LLM-based annex normalization.

	This function is intentionally a stub. It documents the intended
	interface for an LLM-backed normalizer but currently just returns an
	empty list so that it is safe to import and call in deterministic
	pipelines.

	Once wired to an LLM, this function should:

	- Take the linearized `blocks` representation.
	- Prompt a model to infer a clean annex tree using the project graph
	  schema (annex_section, annex_point, annex_subpoint, annex_bullet).
	- Return a list of normalized node dicts ready to be mapped into
	  graph_schema provisions or ParserContext nodes.
	"""

	_ = (blocks, celex, annex_number, model_name)  # suppress unused warnings
	return []
