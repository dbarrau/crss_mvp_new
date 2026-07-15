from __future__ import annotations

import re
from typing import Dict, List

from bs4 import BeautifulSoup

from ..base.utils import ParserContext
from domain.ontology.eurlex_html import (
	ENACTING_TERMS_ID,
	CHAPTER_ID_RE,
	SECTION_ID_RE,
	ARTICLE_ID_RE,
	PARAGRAPH_ID_RE,
	ARTICLE_TITLE_ID_TEMPLATE,
	CLASS_OJ_NORMAL,
	TABLE_POINTS_WIDTH,
)


def parse_enacting_terms(soup, ctx: ParserContext, root: Dict) -> Dict:
	enc_root = soup.find("div", id=ENACTING_TERMS_ID)
	if not enc_root:
		return {}

	enc_node = ctx.make_node("enacting_terms", "enc_1", "", root)

	chapter_pattern = CHAPTER_ID_RE
	section_pattern = SECTION_ID_RE
	article_pattern = ARTICLE_ID_RE
	paragraph_pattern = PARAGRAPH_ID_RE

	def paragraph_text_without_tables(para_div) -> str:
		clone = BeautifulSoup(str(para_div), "html.parser")
		for tbl in clone.find_all("table"):
			tbl.decompose()
		return clone.get_text(" ", strip=True)

	def point_text_without_nested_tables(table) -> str:
		clone = BeautifulSoup(str(table), "html.parser")
		root_table = clone.find("table")
		if not root_table:
			return ""
		for nested in root_table.find_all("table"):
			nested.decompose()
		return root_table.get_text(" ", strip=True)

	# Dash markers EUR-Lex uses for unnumbered "indent" sub-items.
	_DASH_MARKERS = ("—", "–", "•")  # em-dash, en-dash, bullet

	def parse_points_from_tables(parent_node: Dict, tables: List) -> None:
		"""Turn a list of point <table>s into child nodes, recursively.

		Each item table is one enumerated unit — a lettered/roman/numeric point
		``(x)`` or an unnumbered ``—`` indent — whose own text (nested items
		stripped) becomes the node body, and whose immediate sub-item tables
		recurse into child nodes.  So a definition point's structure at any depth
		(point → letter → roman, or a ``—`` list) becomes real, referenceable
		provisions instead of a flattened blob.

		``tables`` may include sub-item tables nested inside others (a recursive
		``find_all`` from the caller, or the whole subtree on recursion); a table
		is processed only when no other table in the same list encloses it, so at
		each level only that level's direct items are created — this also stops a
		point's roman sub-item from being re-emitted as a phantom sibling.
		"""
		table_ids = {id(t) for t in tables}

		def _enclosed_by_sibling(table) -> bool:
			ancestor = table.find_parent("table")
			while ancestor is not None:
				if id(ancestor) in table_ids:
					return True
				ancestor = ancestor.find_parent("table")
			return False

		parent_kind = parent_node.get("kind", "")
		# First-level items under an article/paragraph are "point"s; anything
		# deeper (a point's lettered/roman sub-items) is a "roman_item" — the two
		# kinds the qualified-ref builder chains into "…, point (a)(i)".
		child_point_kind = (
			"point" if parent_kind in ("article", "paragraph", "subparagraph")
			else "roman_item"
		)
		parent_html_id = parent_node["id"].split(f"{ctx.celex}_", 1)[-1]
		indent_seq = 0

		for table in tables:
			if _enclosed_by_sibling(table):
				continue
			text = point_text_without_nested_tables(table)
			label_match = re.match(r"^\(([^)]+)\)\s*", text)
			if label_match:
				label = label_match.group(1)
				content = text[label_match.end():].strip()
				kind = child_point_kind
				html_id = f"{parent_html_id}_{'pt' if kind == 'point' else 'rm'}_{label}"
			elif text[:1] in _DASH_MARKERS:
				indent_seq += 1
				label = str(indent_seq)
				content = text[1:].strip()
				kind = "indent"
				html_id = f"{parent_html_id}_ind_{indent_seq}"
			else:
				continue
			node = ctx.make_node(kind, html_id, content, parent_node, number=label)
			parse_points_from_tables(
				node, table.find_all("table", width=TABLE_POINTS_WIDTH)
			)

	def collect_subparagraph_blocks(para_div):
		"""Group direct children into (p_element, [table_elements]) tuples."""
		blocks = []
		current_p = None
		current_tables = []
		for child in para_div.children:
			if not hasattr(child, 'name') or not child.name:
				continue
			if child.name == 'p' and CLASS_OJ_NORMAL in child.get('class', []):
				if current_p is not None:
					blocks.append((current_p, current_tables))
				current_p = child
				current_tables = []
			elif child.name == 'table':
				current_tables.append(child)
		if current_p is not None:
			blocks.append((current_p, current_tables))
		return blocks

	def parse_paragraph_div(para_div, parent_node: Dict) -> None:
		para_match = paragraph_pattern.match(para_div["id"])
		if not para_match:
			return
		_, para_num_raw = para_match.groups()
		# para_num_raw may be "003" or "003a" — strip leading zeros, keep suffix
		para_number = para_num_raw.lstrip("0") or "0"

		blocks = collect_subparagraph_blocks(para_div)

		if len(blocks) <= 1:
			# Single subparagraph — keep current behaviour
			paragraph = ctx.make_node(
				"paragraph",
				para_div["id"],
				paragraph_text_without_tables(para_div),
				parent_node,
				number=para_number,
			)
			parse_points_from_tables(paragraph, para_div.find_all("table", width=TABLE_POINTS_WIDTH))
		else:
			# Multiple subparagraphs
			paragraph = ctx.make_node(
				"paragraph",
				para_div["id"],
				"",
				parent_node,
				number=para_number,
			)
			for idx, (p_elem, tables) in enumerate(blocks, 1):
				sp_text = p_elem.get_text(" ", strip=True)
				sp_node = ctx.make_node(
					"subparagraph",
					f"{para_div['id']}_sp_{idx}",
					sp_text,
					paragraph,
					number=str(idx),
				)
				parse_points_from_tables(sp_node, tables)

	def parse_paragraphs(article_node: Dict, article_div) -> None:
		for para_div in article_div.find_all("div", id=paragraph_pattern, recursive=False):
			parse_paragraph_div(para_div, article_node)

	def parse_article_body_fallback(article_node: Dict, article_div) -> None:
		"""Parse article content when no numbered paragraph wrapper divs exist.

		Handles four EUR-Lex patterns:
		  1. Intro <p> + definition/point <table> elements  (art 3, 16, 108)
		  2. Single-body <p>                                (art 4, 32, 39)
		  3. Multi-paragraph <p> blocks                     (art 85)
		  4. Intro <p> + amendment <div> containers          (art 102-110)
		"""
		blocks = collect_subparagraph_blocks(article_div)

		if not blocks:
			# No <p class="oj-normal"> at all — extract all readable body text
			body = paragraph_text_without_tables(article_div)
			if body and body != article_node.get("text", ""):
				article_node["text"] = body
			return

		if len(blocks) == 1:
			p_elem, tables = blocks[0]
			body_text = p_elem.get_text(" ", strip=True)
			# For amendment articles, also grab text from child <div> siblings
			extra_parts = []
			capture = False
			for child in article_div.children:
				if child is p_elem:
					capture = True
					continue
				if not capture:
					continue
				if hasattr(child, "name") and child.name == "div" and not child.get("id"):
					extra_parts.append(child.get_text(" ", strip=True))
			if extra_parts:
				body_text = body_text + " " + " ".join(extra_parts)
			article_node["text"] = body_text
			parse_points_from_tables(article_node, tables)
		else:
			# Multiple subparagraph blocks
			for idx, (p_elem, tables) in enumerate(blocks, 1):
				sp_text = p_elem.get_text(" ", strip=True)
				sp_node = ctx.make_node(
					"subparagraph",
					f"{article_div['id']}_sp_{idx}",
					sp_text,
					article_node,
					number=str(idx),
				)
				parse_points_from_tables(sp_node, tables)

	def parse_articles(parent_node: Dict, parent_div) -> bool:
		found = False
		for article_div in parent_div.find_all("div", id=article_pattern, recursive=False):
			article_match = article_pattern.match(article_div["id"])
			if not article_match:
				continue
			found = True
			article_number = article_match.group(1)
			title = extract_title(article_div["id"])
			article_node = ctx.make_node(
				"article",
				article_div["id"],
				title or "",
				parent_node,
				number=article_number,
				title=title,
			)
			parse_paragraphs(article_node, article_div)
			if not article_node["children"]:
				parse_article_body_fallback(article_node, article_div)
		return found

	def parse_sections_or_articles(chapter_node: Dict, chapter_div) -> None:
		section_nodes = [div for div in chapter_div.find_all("div", id=section_pattern, recursive=False)]
		if section_nodes:
			for section_div in section_nodes:
				sec_match = section_pattern.match(section_div["id"])
				if not sec_match:
					continue
				section_number = sec_match.group(2)
				section_title = extract_title(section_div["id"])
				section_node = ctx.make_node(
					"section",
					section_div["id"],
					section_title or "",
					chapter_node,
					number=section_number,
					title=section_title,
				)
				articles_found = parse_articles(section_node, section_div)
				if not articles_found:
					group_paragraphs_as_articles(section_node, section_div)
		else:
			articles_found = parse_articles(chapter_node, chapter_div)
			if not articles_found:
				group_paragraphs_as_articles(chapter_node, chapter_div)

	def group_paragraphs_as_articles(parent_node: Dict, parent_div) -> None:
		buckets: Dict[str, List] = {}
		for para_div in parent_div.find_all("div", id=paragraph_pattern, recursive=False):
			match = paragraph_pattern.match(para_div["id"])
			if not match:
				continue
			art_num, _ = match.groups()
			buckets.setdefault(art_num, []).append(para_div)
		for art_num, para_list in buckets.items():
			article_node = ctx.make_node(
				"article",
				f"art_{int(art_num)}",
				extract_title(f"art_{int(art_num)}") or "",
				parent_node,
				number=str(int(art_num)),
			)
			for para_div in para_list:
				parse_paragraph_div(para_div, article_node)

	def extract_title(id_value: str):
		title_node = soup.find("div", id=ARTICLE_TITLE_ID_TEMPLATE.format(id=id_value))
		return title_node.get_text(" ", strip=True) if title_node else None

	found_chapters = False
	for chapter_div in enc_root.find_all("div", id=chapter_pattern, recursive=False):
		chapter_match = chapter_pattern.match(chapter_div["id"])
		if not chapter_match:
			continue
		found_chapters = True
		chapter_number = chapter_match.group(1)
		chapter_title = extract_title(chapter_div["id"])
		chapter_node = ctx.make_node(
			"chapter",
			chapter_div["id"],
			chapter_title or "",
			enc_node,
			number=chapter_number,
			title=chapter_title,
		)
		parse_sections_or_articles(chapter_node, chapter_div)

	if not found_chapters:
		# Regulations without chapters (e.g. short implementing regulations) have
		# articles sitting directly under enc_1 with no cpt_* wrapper.
		parse_sections_or_articles(enc_node, enc_root)

	return enc_node
