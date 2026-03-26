from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag

from ..base.utils import ParserContext
from domain.ontology.eurlex_html import (
	ANNEX_ID_RE,
	ANNEX_SKIP_ID,
	CLASS_ELI_CONTAINER,
	CLASS_OJ_DOC_TI,
	CLASS_OJ_TI_GRSEQ_1,
	CLASS_OJ_NORMAL,
	CLASS_OJ_ENUMERATION_SPACING,
)

# ── Number / marker extraction ────────────────────────────────────────
_DOTTED_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+")     # "1.1.1.   text"
_NUM_MARKER_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?$")       # "1." or "1.1." (cell marker)
_LETTER_RE     = re.compile(r"^\(([a-zA-Z]{1,2})\)$")      # "(a)", "(aa)"
_DASH_RE       = re.compile(r"^[—–\-•]$")                  # em-dash, en-dash, hyphen, bullet
_CHAPTER_RE    = re.compile(r"^Chapter\s+([IVXLCDM]+)", re.I)
_PART_RE       = re.compile(r"^PART\s+([A-Z])\b", re.I)
_SECTION_RE    = re.compile(r"^Section\s+([A-Z0-9]+)\.?\s*(.*)", re.I)
_WS_RE         = re.compile(r"[\xa0\s]+")


def _norm(text: str) -> str:
	"""Collapse &nbsp; and whitespace runs into single spaces."""
	return _WS_RE.sub(" ", text).strip()


def _parse_dotted(text: str) -> Optional[Tuple[str, str, int]]:
	"""Extract a dotted-number prefix.

	Returns (number, remaining_text, depth) or None.
	"1.1.1.   text" → ("1.1.1", "text", 3)
	"10.   HEADING" → ("10", "HEADING", 1)
	"""
	m = _DOTTED_NUM_RE.match(text)
	if not m:
		return None
	num = m.group(1)
	return num, text[m.end():].strip(), num.count(".") + 1


def _cell_text(cell: Tag) -> str:
	"""Text of a cell, stripping nested tables."""
	clone = BeautifulSoup(str(cell), "html.parser")
	for tbl in clone.find_all("table"):
		tbl.decompose()
	return clone.get_text(" ", strip=True)


def _id_of(parent: Dict, number: str, celex: str) -> str:
	"""Build a node id rooted at the immediate parent: anx_VI_part_A_1.1"""
	return f"{_suffix(parent, celex)}_{number}"


def _suffix(node: Dict, celex: str) -> str:
	"""Node-id part after the CELEX prefix."""
	return node["id"].split(f"{celex}_", 1)[-1]


# ── Public entry point ────────────────────────────────────────────────
def parse_annexes(soup, ctx: ParserContext, root: Dict) -> Optional[Dict]:
	annex_divs = soup.find_all(
		"div", id=ANNEX_ID_RE, class_=CLASS_ELI_CONTAINER,
	)
	if not annex_divs:
		return None

	annexes_root = ctx.make_node("annexes", "annexes", "", root)

	for div in annex_divs:
		html_id = div.get("id", "")
		if html_id == ANNEX_SKIP_ID:
			continue
		_parse_single_annex(div, html_id, ctx, annexes_root, soup)

	return annexes_root


# ── Per-annex parser ──────────────────────────────────────────────────
def _parse_single_annex(
	annex_div: Tag, html_id: str, ctx: ParserContext,
	annexes_root: Dict, soup,
) -> None:
	# ── title ──
	title_ps = annex_div.find_all("p", class_=CLASS_OJ_DOC_TI, recursive=False)
	titles = [_norm(t.get_text(" ", strip=True)) for t in title_ps if t.get_text(strip=True)]
	annex_title = titles[1] if len(titles) > 1 else (titles[0] if titles else _fallback_title(soup, html_id) or html_id)

	num_m = re.match(r"^anx_([A-Za-z0-9]+)$", html_id)
	annex_node = ctx.make_node(
		"annex", html_id, annex_title, annexes_root,
		title=annex_title,
		number=num_m.group(1) if num_m else None,
	)

	# Collect direct child elements (p, table, div) in document order.
	elements: List[Tag] = [
		el for el in annex_div.children
		if isinstance(el, Tag) and el.name in ("p", "table", "div")
		and CLASS_OJ_DOC_TI not in (el.get("class") or [])
	]

	# ── Stack: (depth, node) ──
	# depth -1 = annex root
	# depth  0 = chapters / parts / named sections
	# depth  1+ = numbered content (depth = segment count)
	stack: List[Tuple[int, Dict]] = [(-1, annex_node)]
	blt_cnt: Dict[str, int] = {}          # parent_id → bullet counter

	i = 0
	while i < len(elements):
		el = elements[i]
		cls = el.get("class") or []

		# ── <p class="oj-ti-grseq-1"> : heading ──
		if el.name == "p" and CLASS_OJ_TI_GRSEQ_1 in cls:
			i = _on_heading(elements, i, stack, annex_node, html_id, ctx, blt_cnt)
			continue

		# ── <p class="oj-normal"> : body paragraph ──
		if el.name == "p" and CLASS_OJ_NORMAL in cls:
			i = _on_paragraph(elements, i, stack, html_id, ctx)
			continue

		# ── <p> with other/no class : treat as continuation ──
		if el.name == "p":
			text = _norm(el.get_text(" ", strip=True))
			if text:
				stack[-1][1]["text"] = f"{stack[-1][1]['text']} {text}".strip()
			i += 1
			continue

		# ── <table> : list item ──
		if el.name == "table":
			_on_table(el, stack, html_id, ctx, blt_cnt)
			i += 1
			continue

		# ── <div class="oj-enumeration-spacing"> ──
		if el.name == "div" and CLASS_OJ_ENUMERATION_SPACING in cls:
			_on_enum_spacing(el, stack, html_id, ctx)
			i += 1
			continue

		i += 1


# ── Stack helper ──────────────────────────────────────────────────────
def _parent_for(stack: List[Tuple[int, Dict]], depth: int) -> Dict:
	"""Pop entries with depth >= *depth*; return the new top as parent."""
	while len(stack) > 1 and stack[-1][0] >= depth:
		stack.pop()
	return stack[-1][1]


# ── Heading handler (oj-ti-grseq-1) ──────────────────────────────────
def _on_heading(
	elements: List[Tag], i: int,
	stack: List[Tuple[int, Dict]], annex_node: Dict,
	html_id: str, ctx: ParserContext, blt_cnt: Dict[str, int],
) -> int:
	text = _norm(elements[i].get_text(" ", strip=True))
	if not text:
		return i + 1

	# ── Chapter ──
	m = _CHAPTER_RE.match(text)
	if m:
		parent = _parent_for(stack, 0)
		roman = m.group(1)
		title = _peek_title(elements, i + 1)
		if title is not None:
			i += 1  # consumed lookahead
		node = ctx.make_node(
			"annex_chapter", f"{html_id}_chp_{roman}",
			title or text, parent,
			title=title or text, number=roman,
		)
		stack.append((0, node))
		return i + 1

	# ── Part ──
	m = _PART_RE.match(text)
	if m:
		parent = _parent_for(stack, 0)
		letter = m.group(1).upper()
		title = _peek_title(elements, i + 1)
		if title is not None:
			i += 1
		node = ctx.make_node(
			"annex_part", f"{html_id}_part_{letter}",
			title or text, parent,
			title=title or text, number=letter,
		)
		stack.append((0, node))
		return i + 1

	# ── Named section ("Section A. …") ──
	m = _SECTION_RE.match(text)
	if m:
		parent = _parent_for(stack, 0)
		label = m.group(1)
		content = m.group(2).strip() or text
		node = ctx.make_node(
			"annex_section", f"{html_id}_sec_{label}",
			content, parent,
			title=content, number=label,
		)
		stack.append((0, node))
		return i + 1

	# ── Numbered heading ("1.   HEADING" / "1.1.   Sub") ──
	parsed = _parse_dotted(text)
	if parsed:
		number, content, depth = parsed
		parent = _parent_for(stack, depth)
		# Depth 1 = broad section (e.g. "1. ORGANISATIONAL REQUIREMENTS")
		# Depth 2+ = subsection — the ideal retrieval anchor
		kind = "annex_section" if depth <= 1 else "annex_subsection"
		node = ctx.make_node(
			kind, _id_of(parent, number, ctx.celex),
			content, parent,
			title=content, number=number,
		)
		stack.append((depth, node))
		return i + 1

	# ── Unnumbered heading ──
	# Before any numbered content → annex subtitle.
	# After numbered content → title annotation on current parent.
	if len(stack) == 1 and stack[0][0] == -1:
		annex_node["text"] = text
		annex_node["title"] = text
	else:
		top = stack[-1][1]
		kind = top.get("kind", "")
		if kind in ("annex_chapter", "annex_part") and not top.get("children"):
			top["title"] = text
			top["text"] = text
		else:
			top["text"] = f"{top['text']} {text}".strip()
	return i + 1


def _peek_title(elements: List[Tag], idx: int) -> Optional[str]:
	"""Look ahead for an unnumbered oj-ti-grseq-1 that serves as title."""
	if idx >= len(elements):
		return None
	nxt = elements[idx]
	if nxt.name != "p" or CLASS_OJ_TI_GRSEQ_1 not in (nxt.get("class") or []):
		return None
	t = _norm(nxt.get_text(" ", strip=True))
	if _parse_dotted(t) or _CHAPTER_RE.match(t) or _PART_RE.match(t) or _SECTION_RE.match(t):
		return None  # it's a numbered heading, not a title
	return t or None


# ── Body paragraph handler (oj-normal) ────────────────────────────────
def _on_paragraph(
	elements: List[Tag], i: int,
	stack: List[Tuple[int, Dict]], html_id: str, ctx: ParserContext,
) -> int:
	text = _norm(elements[i].get_text(" ", strip=True))
	if not text:
		return i + 1

	parsed = _parse_dotted(text)
	if parsed:
		number, content, depth = parsed
		parent = _parent_for(stack, depth)
		node = ctx.make_node(
			"annex_point", _id_of(parent, number, ctx.celex),
			content, parent, number=number,
		)
		stack.append((depth, node))
		return i + 1

	# Continuation text
	stack[-1][1]["text"] = f"{stack[-1][1]['text']} {text}".strip()
	return i + 1


# ── Table handler ─────────────────────────────────────────────────────
def _on_table(
	table: Tag, stack: List[Tuple[int, Dict]],
	html_id: str, ctx: ParserContext, blt_cnt: Dict[str, int],
) -> None:
	container = table.find("tbody", recursive=False) or table
	for row in container.find_all("tr", recursive=False):
		cells = row.find_all("td", recursive=False)
		if not cells:
			continue

		# Determine marker & body depending on column count
		if len(cells) >= 3:
			marker = _norm(cells[1].get_text(" ", strip=True))
			body = _cell_text(cells[2])
			content_cell = cells[2]
		elif len(cells) == 2:
			marker = _norm(cells[0].get_text(" ", strip=True))
			body = _cell_text(cells[1])
			content_cell = cells[1]
		else:
			body = _cell_text(cells[0])
			stack[-1][1]["text"] = f"{stack[-1][1]['text']} {body}".strip()
			continue

		# ── Dotted-number marker ──
		nm = _NUM_MARKER_RE.match(marker)
		if nm:
			number = nm.group(1)
			depth = number.count(".") + 1
			parent = _parent_for(stack, depth)
			node = ctx.make_node(
				"annex_point", _id_of(parent, number, ctx.celex),
				body, parent, number=number,
			)
			stack.append((depth, node))
			_process_nested_tables(content_cell, stack, html_id, ctx, blt_cnt)
			continue

		# ── Letter marker ──
		lm = _LETTER_RE.match(marker)
		if lm:
			letter = lm.group(1).lower()
			parent = stack[-1][1]
			ctx.make_node(
				"annex_subpoint",
				f"{_suffix(parent, ctx.celex)}_{letter}",
				body, parent, number=letter,
			)
			_process_nested_tables(content_cell, stack, html_id, ctx, blt_cnt)
			continue

		# ── Dash / bullet marker ──
		if _DASH_RE.match(marker):
			parent = stack[-1][1]
			blt_cnt[parent["id"]] = blt_cnt.get(parent["id"], 0) + 1
			ctx.make_node(
				"annex_bullet",
				f"{_suffix(parent, ctx.celex)}_blt_{blt_cnt[parent['id']]}",
				body, parent,
			)
			_process_nested_tables(content_cell, stack, html_id, ctx, blt_cnt)
			continue

		# ── Fallback: append as continuation ──
		combined = f"{marker} {body}".strip() if marker else body
		stack[-1][1]["text"] = f"{stack[-1][1]['text']} {combined}".strip()


def _process_nested_tables(
	cell: Tag, stack: List[Tuple[int, Dict]],
	html_id: str, ctx: ParserContext, blt_cnt: Dict[str, int],
) -> None:
	for nested in cell.find_all("table", recursive=False):
		_on_table(nested, stack, html_id, ctx, blt_cnt)


# ── oj-enumeration-spacing handler ────────────────────────────────────
def _on_enum_spacing(
	div: Tag, stack: List[Tuple[int, Dict]],
	html_id: str, ctx: ParserContext,
) -> None:
	"""Handle <div class="oj-enumeration-spacing"> blocks.

	These contain inline <p> elements: first with the number, rest with text.
	"""
	ps = div.find_all("p", recursive=False)
	if len(ps) < 2:
		return
	num_text = _norm(ps[0].get_text(" ", strip=True))
	body = _norm(" ".join(p.get_text(" ", strip=True) for p in ps[1:]))

	parsed = _parse_dotted(num_text + " ")  # add space so regex matches
	if parsed:
		number, _, depth = parsed
		parent = _parent_for(stack, depth)
		node = ctx.make_node(
			"annex_point", _id_of(parent, number, ctx.celex),
			body, parent, number=number,
		)
		stack.append((depth, node))
		return

	# Plain number marker (e.g. "1." in cell)
	nm = _NUM_MARKER_RE.match(num_text)
	if nm:
		number = nm.group(1)
		depth = number.count(".") + 1
		parent = _parent_for(stack, depth)
		node = ctx.make_node(
			"annex_point", _id_of(parent, number, ctx.celex),
			body, parent, number=number,
		)
		stack.append((depth, node))
		return

	# Fallback: continuation
	stack[-1][1]["text"] = f"{stack[-1][1]['text']} {num_text} {body}".strip()


# ── Utility ───────────────────────────────────────────────────────────
def _fallback_title(soup, html_id: str) -> Optional[str]:
	node = soup.find("div", id=f"{html_id}.tit_1")
	return node.get_text(" ", strip=True) if node else None
