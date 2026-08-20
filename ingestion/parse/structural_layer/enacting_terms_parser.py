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
	CLASS_LIST,
	BODY_SUBPARA_CLASSES,
	BODY_TEXT_CLASSES,
	TABLE_POINTS_WIDTH,
)


def _classes(el) -> set:
	return set(el.get("class") or [])


def _opens_subparagraph(el) -> bool:
	"""A <p>/<div> that starts a new body subparagraph (oj-normal / normal)."""
	return getattr(el, "name", None) in ("p", "div") and bool(_classes(el) & BODY_SUBPARA_CLASSES)


def _is_continuation(el) -> bool:
	"""Body prose that continues the current subparagraph rather than opening a
	new one: a "list" element (a subparagraph following a point-list), or a bare
	unlabelled <div> (an amendment's quoted replacement block emitted between
	body paragraphs, e.g. MDR Art 117's ``'(12) …'``)."""
	if getattr(el, "name", None) not in ("p", "div"):
		return False
	classes = _classes(el)
	if CLASS_LIST in classes:
		return True
	return el.name == "div" and not el.get("id") and not classes


# Opening quotation marks EUR-Lex uses to delimit an amendment's quoted
# replacement text ("… is replaced by the following: '…'"). U+2018 is the OJ
# convention; the rest are defensive. A nested point-table whose content opens
# with one of these is the operative new text an amending point enacts — not an
# enumerated sub-item — so it is folded into the point's body instead of being
# dropped with the sub-item tables (which left amendments as husks like
# "point (c) is replaced by the following: ;").
_QUOTE_OPENERS = ("\u2018", "\u201c", "\u00ab", "\u0027", "\u0022")


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

	def _opens_with_quote(table) -> bool:
		"""True when a nested table holds an amendment's quoted replacement text
		(its content opens with a quotation mark) rather than an enumerated
		sub-item — the signal to fold it in rather than recurse into it."""
		return table.get_text(" ", strip=True)[:1] in _QUOTE_OPENERS

	def _direct_child_tables(root_table) -> List:
		"""Tables whose nearest enclosing table is root_table (its own level)."""
		return [t for t in root_table.find_all("table")
			if t.find_parent("table") is root_table]

	def point_text_without_nested_tables(table) -> str:
		"""Point body with enumerated sub-item tables stripped.

		Enumerated ``(x)``/``—`` sub-items become their own child nodes, so their
		text is removed here.  An amendment's quoted replacement block is *not* a
		sub-item but the operative new text the point enacts, so it is folded in
		(kept) — otherwise amending provisions parse to husks like
		``"point (c) is replaced by the following: ;"``.
		"""
		clone = BeautifulSoup(str(table), "html.parser")
		root_table = clone.find("table")
		if not root_table:
			return ""
		for nested in _direct_child_tables(root_table):
			if not _opens_with_quote(nested):
				nested.decompose()
		return root_table.get_text(" ", strip=True)

	def _sub_item_tables(table) -> List:
		"""Enumerated sub-item tables to recurse into: every descendant point
		table except those inside a quoted replacement block (already folded into
		the parent body by point_text_without_nested_tables)."""
		out = []
		for t in table.find_all("table", width=TABLE_POINTS_WIDTH):
			anc, quoted = t, False
			while anc is not None and anc is not table:
				if anc.name == "table" and _opens_with_quote(anc):
					quoted = True
					break
				anc = anc.find_parent("table")
			if not quoted:
				out.append(t)
		return out

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
			elif text[:1] in _QUOTE_OPENERS:
				# A quoted block at this level is amendment replacement text that
				# arrived without an enumerated label to become a child — an
				# article body "…the following point is added:" followed by the
				# quoted added point as a sibling table (AI Act Article 110 → the
				# added point '(68) …' inserted into Directive 2020/1828 Annex I).
				# Fold it into the parent's text so the added/replacement text is
				# not lost — the same rescue point_text_without_nested_tables gives
				# a quote nested inside a lead-in point, for the case where the
				# quote sits at this level with no point to fold it into.
				parent_node["text"] = (
					(parent_node.get("text", "") or "").rstrip() + " " + text
				).strip()
				continue
			else:
				continue
			node = ctx.make_node(kind, html_id, content, parent_node, number=label)
			parse_points_from_tables(node, _sub_item_tables(table))

	def group_blocks(elements) -> List:
		"""Group a flat sequence of body elements into ``(text, [table_elements])``
		tuples — one entry per subparagraph, with its own point-list tables
		attached.

		A body element that opens a subparagraph (``oj-normal``/``normal``) starts
		a new block; a ``<table>`` attaches its point-list to the open block; a
		continuation element (a ``list`` subparagraph, or a bare unlabelled
		amendment ``<div>``) appends its prose to the open block instead of being
		dropped. A point-list arriving before any prose opens an empty-text block
		so it still becomes referenceable points.
		"""
		blocks = []
		current_text = None
		current_tables = []

		def flush():
			nonlocal current_text, current_tables
			if current_text is not None:
				blocks.append((current_text.strip(), current_tables))
			current_text, current_tables = None, []

		for child in elements:
			if not getattr(child, "name", None):
				continue
			if _opens_subparagraph(child):
				flush()
				current_text = child.get_text(" ", strip=True)
			elif child.name == "table":
				if current_text is None:
					current_text = ""
				current_tables.append(child)
			elif _is_continuation(child):
				text = child.get_text(" ", strip=True)
				if not text:
					continue
				current_text = f"{current_text} {text}".strip() if current_text else text
		flush()
		return blocks

	def collect_subparagraph_blocks(para_div):
		"""Group para_div's direct children into (text, [table_elements]) blocks."""
		return group_blocks(para_div.children)

	def parse_paragraph_div(para_div, parent_node: Dict, orphans: List = ()) -> Dict:
		"""Build a paragraph node from para_div, plus any trailing *orphans* —
		bare <p class="oj-normal"> siblings EUR-Lex emitted OUTSIDE para_div for
		this paragraph's second-and-later subparagraphs (see parse_paragraphs).
		"""
		para_match = paragraph_pattern.match(para_div["id"])
		if not para_match:
			return None
		_, para_num_raw = para_match.groups()
		# para_num_raw may be "003" or "003a" — strip leading zeros, keep suffix
		para_number = para_num_raw.lstrip("0") or "0"

		own_blocks = collect_subparagraph_blocks(para_div)
		orphan_blocks = group_blocks(orphans)

		if len(own_blocks) <= 1 and not orphan_blocks:
			# Single subparagraph, nothing trailing — keep current behaviour.
			# Robust to however BeautifulSoup nests a malformed-but-common EUR-Lex
			# markup (a point-list's <table>s ending up as descendants of a
			# swallowed outer <p> rather than as para_div's direct children):
			# paragraph_text_without_tables / find_all("table", ...) both search
			# the WHOLE subtree, not just direct children.
			paragraph = ctx.make_node(
				"paragraph",
				para_div["id"],
				paragraph_text_without_tables(para_div),
				parent_node,
				number=para_number,
			)
			parse_points_from_tables(paragraph, para_div.find_all("table", width=TABLE_POINTS_WIDTH))
			return paragraph

		# Multiple subparagraphs — either para_div's own markup already has
		# them, or trailing orphans extend a single first subparagraph into a
		# real subparagraph-1..N structure. "first subparagraph"/"second
		# subparagraph" are themselves citable units in EU legislative
		# drafting (e.g. MDR Art 14(2)'s dropped postamble literally reads
		# "...referred to in the first subparagraph...").
		paragraph = ctx.make_node("paragraph", para_div["id"], "", parent_node, number=para_number)
		if len(own_blocks) <= 1:
			# subparagraph 1 uses the same whole-subtree extraction as the fast
			# path above, so a point-list nested under a swallowed outer <p>
			# is still found even though own_blocks' table list would miss it.
			sp_text = paragraph_text_without_tables(para_div)
			sp_node = ctx.make_node(
				"subparagraph", f"{para_div['id']}_sp_1", sp_text, paragraph, number="1"
			)
			parse_points_from_tables(sp_node, para_div.find_all("table", width=TABLE_POINTS_WIDTH))
			next_idx = 2
		else:
			for idx, (sp_text, tables) in enumerate(own_blocks, 1):
				sp_node = ctx.make_node(
					"subparagraph", f"{para_div['id']}_sp_{idx}", sp_text, paragraph, number=str(idx)
				)
				parse_points_from_tables(sp_node, tables)
			next_idx = len(own_blocks) + 1
		for offset, (sp_text, tables) in enumerate(orphan_blocks):
			idx = next_idx + offset
			sp_node = ctx.make_node(
				"subparagraph", f"{para_div['id']}_sp_{idx}", sp_text, paragraph, number=str(idx)
			)
			parse_points_from_tables(sp_node, tables)
		return paragraph

	def _detached_paragraph_number(child) -> str:
		"""If ``child`` is a bare body <div> standing in for a numbered paragraph
		whose <div id="NNN.MMM"> wrapper EUR-Lex omitted, return that paragraph
		number, else "". Such a paragraph is a body <div> with no wrapper id whose
		text opens with its own number (IVDR Art 42 para 7, Art 38 para 13); left
		unhandled it matches neither the paragraph-div test nor the orphan test
		below and is dropped whole."""
		if child.name != "div" or (child.get("id") or "") or not (_classes(child) & BODY_SUBPARA_CLASSES):
			return ""
		m = re.match(r"^(\d+)\s+(?=\S)", child.get_text(" ", strip=True))
		return m.group(1) if m else ""

	def parse_paragraphs(article_node: Dict, article_div) -> None:
		"""Walk article_div's direct children in document order, materialising
		numbered paragraphs and attaching trailing body content to the paragraph
		it follows.

		A numbered paragraph's <div id="NNN.MMM"> wraps only its FIRST
		subparagraph; EUR-Lex emits every subparagraph after the first as a bare
		body element (usually <p class="oj-normal">, but also class="normal"/"list"
		or a point-list <table>) SIBLING of that div, untagged, with no id linking
		it back. A find_all(..., recursive=False) scoped to <div id=...> tags can
		never see a <p>, so that text was silently dropped (confirmed on MDR
		Article 14 paragraphs 2 and 6, and IVDR Art 110's class="normal" orphan).
		A run of trailing body elements after a numbered div, up to the next
		numbered div, belongs to the paragraph it trails.

		Some consolidated layouts also emit a whole numbered paragraph as a bare
		body <div> with no wrapper id (see _detached_paragraph_number); such a div
		becomes its own paragraph node rather than being folded into its
		predecessor.
		"""
		art_num = article_node.get("number", "")

		pending_div = None
		pending_orphans: List = []

		def flush():
			nonlocal pending_div, pending_orphans
			if pending_div is not None:
				parse_paragraph_div(pending_div, article_node, pending_orphans)
			pending_div, pending_orphans = None, []

		for child in article_div.find_all(["div", "p", "table"], recursive=False):
			cid = child.get("id") if child.name == "div" else None
			if child.name == "div" and cid and paragraph_pattern.match(cid):
				flush()
				pending_div = child
				pending_orphans = []
			elif (detached := _detached_paragraph_number(child)) and art_num.isdigit():
				# A numbered paragraph EUR-Lex emitted without its wrapper div.
				flush()
				local_id = f"{int(art_num):03d}.{int(detached):03d}"
				body = re.sub(r"^\d+\s+", "", child.get_text(" ", strip=True))
				para = ctx.make_node("paragraph", local_id, body, article_node, number=detached)
				parse_points_from_tables(para, child.find_all("table", width=TABLE_POINTS_WIDTH))
			elif pending_div is not None and (
				child.name == "table" or _opens_subparagraph(child) or _is_continuation(child)
			):
				pending_orphans.append(child)
		flush()

	def parse_article_body_fallback(article_node: Dict, article_div) -> None:
		"""Parse article content when no numbered paragraph wrapper divs exist.

		Handles four EUR-Lex patterns:
		  1. Intro <p> + definition/point <table> elements  (art 3, 16, 108)
		  2. Single-body <p>                                (art 4, 32, 39)
		  3. Multi-paragraph <p> blocks                     (art 85)
		  4. Intro <p> + amendment <div>/quoted-block text   (art 102-110, 117)

		group_blocks handles the block-vs-continuation split — a bare amendment
		<div> or a "list" subparagraph is folded into the block it trails, so the
		single-block path no longer needs to rescue trailing <div> text by hand.
		"""
		blocks = collect_subparagraph_blocks(article_div)

		if not blocks:
			# No body prose at all — extract all readable body text
			body = paragraph_text_without_tables(article_div)
			if body and body != article_node.get("text", ""):
				article_node["text"] = body
			return

		if len(blocks) == 1:
			body_text, tables = blocks[0]
			article_node["text"] = body_text
			parse_points_from_tables(article_node, tables)
		else:
			# Multiple subparagraph blocks
			for idx, (sp_text, tables) in enumerate(blocks, 1):
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
