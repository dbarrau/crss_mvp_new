from __future__ import annotations

import re
from typing import Dict, Optional

from bs4 import BeautifulSoup

from .utils import ParserContext


def parse_annexes(soup, ctx: ParserContext, root: Dict) -> Optional[Dict]:
	annex_divs = soup.find_all("div", id=re.compile(r"^anx_[A-Za-z0-9]+$"), class_="eli-container")
	if not annex_divs:
		return None

	annexes_root = ctx.make_node("annexes", "annexes", "", root)

	number_token = re.compile(r"^(\d+(?:\.\d+)*)\.?$")
	letter_token = re.compile(r"^\(([a-zA-Z]+)\)$")
	dash_token = re.compile(r"^[-–—]$")
	section_heading_class = "oj-ti-grseq-1"

	def cell_text_without_tables(cell) -> str:
		clone = BeautifulSoup(str(cell), "html.parser")
		for tbl in clone.find_all("table"):
			tbl.decompose()
		return clone.get_text(" ", strip=True)

	def parse_table(table, current_parent: Dict, current_point: Optional[Dict], counters: Dict[str, int]) -> Optional[Dict]:
		rows = table.find_all("tr", recursive=False)
		last_point = current_point
		for row in rows:
			cells = row.find_all("td", recursive=False)
			if not cells:
				continue
			if len(cells) >= 3:
				num_text = cells[1].get_text(" ", strip=True)
				body_text = cell_text_without_tables(cells[2])
				num_match = number_token.match(num_text)
				counters["point"] += 1
				last_point = ctx.make_node(
					"annex_point",
					f"{current_parent['id'].split(f'{ctx.celex}_', 1)[-1]}_pt_{counters['point']}",
					body_text,
					current_parent,
					number=num_match.group(1) if num_match else num_text or str(counters["point"]),
				)
				nested = cells[2].find_all("table", recursive=False)
				for nested_table in nested:
					last_point = parse_table(nested_table, last_point, last_point, counters) or last_point
				continue
			elif len(cells) == 2:
				token_text = cells[0].get_text(" ", strip=True)
				body_text = cell_text_without_tables(cells[1])
				letter_match = letter_token.match(token_text)
				dash_match = dash_token.match(token_text)
				if letter_match:
					counters["subpoint"] += 1
					host = last_point or current_parent
					ctx.make_node(
						"annex_subpoint",
						f"{host['id'].split(f'{ctx.celex}_', 1)[-1]}_ltr_{counters['subpoint']}",
						body_text,
						host,
						number=letter_match.group(1),
					)
					nested = cells[1].find_all("table", recursive=False)
					for nested_table in nested:
						parse_table(nested_table, host, last_point, counters)
					continue
				if dash_match:
					counters["bullet"] += 1
					host = last_point or current_parent
					ctx.make_node(
						"annex_bullet",
						f"{host['id'].split(f'{ctx.celex}_', 1)[-1]}_blt_{counters['bullet']}",
						body_text,
						host,
					)
					nested = cells[1].find_all("table", recursive=False)
					for nested_table in nested:
						parse_table(nested_table, host, last_point, counters)
					continue
			counters["bullet"] += 1
			host = last_point or current_parent
			ctx.make_node(
				"annex_bullet",
				f"{host['id'].split(f'{ctx.celex}_', 1)[-1]}_blt_{counters['bullet']}",
				cell_text_without_tables(cells[-1]),
				host,
			)
		return last_point

	def parse_enumeration_spacing(block, current_parent: Dict, counters: Dict[str, int]) -> None:
		items = block.find_all("p", recursive=False) or block.find_all("p")
		for item in items:
			text = item.get_text(" ", strip=True)
			if not text:
				continue
			parts = text.split(None, 1)
			if len(parts) == 2 and number_token.match(parts[0].rstrip(".")):
				counters["point"] += 1
				ctx.make_node(
					"annex_point",
					f"{current_parent['id'].split(f'{ctx.celex}_', 1)[-1]}_pt_{counters['point']}",
					parts[1],
					current_parent,
					number=parts[0].rstrip("."),
				)
			else:
				counters["para"] += 1
				ctx.make_node(
					"annex_paragraph",
					f"{current_parent['id'].split(f'{ctx.celex}_', 1)[-1]}_p_{counters['para']}",
					text,
					current_parent,
				)

	def parse_annex_body(annex_node: Dict, annex_div) -> None:
		counters = {"section": 0, "point": 0, "subpoint": 0, "bullet": 0, "para": 0}
		current_parent: Dict = annex_node
		last_point: Optional[Dict] = None

		blocks = annex_div.find_all(["p", "div", "table"], recursive=True)
		for block in blocks:
			if block.name == "p" and "oj-doc-ti" in (block.get("class") or []):
				continue
			if block.name == "p" and section_heading_class in (block.get("class") or []):
				counters["section"] += 1
				text = block.get_text(" ", strip=True)
				number = None
				section_match = re.match(r"Section\s+([A-Za-z0-9]+)", text, re.IGNORECASE)
				if section_match:
					number = section_match.group(1)
				current_parent = ctx.make_node(
					"annex_section",
					f"{annex_node['id'].split(f'{ctx.celex}_', 1)[-1]}_sec_{counters['section']}",
					text,
					annex_node,
					number=number,
					title=text,
				)
				last_point = None
				continue
			if block.name == "div" and "oj-enumeration-spacing" in (block.get("class") or []):
				parse_enumeration_spacing(block, current_parent, counters)
				last_point = None
				continue
			if block.name == "table":
				last_point = parse_table(block, current_parent, last_point, counters) or last_point
				continue
			if block.name == "p":
				text = block.get_text(" ", strip=True)
				if not text:
					continue
				lead_parts = text.split(None, 1)
				if len(lead_parts) == 2 and number_token.match(lead_parts[0].rstrip(".")):
					counters["point"] += 1
					last_point = ctx.make_node(
						"annex_point",
						f"{current_parent['id'].split(f'{ctx.celex}_', 1)[-1]}_pt_{counters['point']}",
						lead_parts[1],
						current_parent,
						number=lead_parts[0].rstrip("."),
					)
					continue
				counters["para"] += 1
				ctx.make_node(
					"annex_paragraph",
					f"{current_parent['id'].split(f'{ctx.celex}_', 1)[-1]}_p_{counters['para']}",
					text,
					current_parent,
				)

		# Fallback: flat row-based extraction over all table rows to ensure items are captured
		rows = annex_div.find_all("tr")
		for row in rows:
			cells = row.find_all("td")
			if len(cells) == 0:
				continue
			if len(cells) >= 2:
				num_text = cells[0].get_text(" ", strip=True)
				body_text = cell_text_without_tables(cells[-1])
				counters["point"] += 1
				ctx.make_node(
					"annex_point",
					f"{annex_node['id'].split(f'{ctx.celex}_', 1)[-1]}_pt_fb_{counters['point']}",
					body_text,
					annex_node,
					number=num_text or str(counters["point"]),
				)

	for annex in annex_divs:
		annex_id = annex.get("id", "")
		if annex_id == "anx_ES":
			continue
		title_divs = annex.find_all("p", class_="oj-doc-ti")
		title_texts = [t.get_text(" ", strip=True) for t in title_divs if t.get_text(" ", strip=True)]
		title = title_texts[1] if len(title_texts) > 1 else (title_texts[0] if title_texts else extract_title_from_soup(soup, annex_id))
		number_match = re.match(r"^anx_([A-Za-z0-9]+)$", annex_id)
		number = number_match.group(1) if number_match else None
		annex_node = ctx.make_node(
			"annex",
			annex_id,
			title or annex_id,
			annexes_root,
			title=title,
			number=number,
		)
		parse_annex_body(annex_node, annex)

	return annexes_root


def extract_title_from_soup(soup, id_value: str) -> Optional[str]:
	title_node = soup.find("div", id=f"{id_value}.tit_1")
	return title_node.get_text(" ", strip=True) if title_node else None
