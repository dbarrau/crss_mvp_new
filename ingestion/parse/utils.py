from __future__ import annotations

from typing import Dict, List, Optional


class ParserContext:
	def __init__(self, celex: str, lang: str = "EN") -> None:
		self.celex = celex
		self.lang = lang
		self.provisions: List[Dict] = []
		self.relations: List[Dict] = []
		self.nodes: Dict[str, Dict] = {}

	def child_path(self, parent: Optional[Dict]) -> List[str]:
		if not parent:
			return []
		return parent.get("path", []) + [parent["id"]]

	def add_node(self, node: Dict, parent_id: Optional[str]) -> Dict:
		self.provisions.append(node)
		self.nodes[node["id"]] = node
		if parent_id and parent_id in self.nodes:
			self.nodes[parent_id]["children"].append(node["id"])
		return node

	def make_node(
		self,
		kind: str,
		html_id: str,
		text: str,
		parent: Optional[Dict],
		title: Optional[str] = None,
		number: Optional[str] = None,
	) -> Dict:
		parent_id = parent["id"] if parent else None
		node_id = f"{self.celex}_{html_id}"
		node: Dict = {
			"id": node_id,
			"kind": kind,
			"text": text or "",
			"hierarchy_depth": (parent.get("hierarchy_depth", -1) + 1) if parent else 0,
			"path": self.child_path(parent),
			"parent_id": parent_id,
			"children": [],
			"lang": self.lang,
		}
		if title is not None:
			node["title"] = title
		if number is not None:
			node["number"] = number
		return self.add_node(node, parent_id)
