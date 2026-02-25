from __future__ import annotations

import re
from typing import Dict, Optional

from bs4 import BeautifulSoup

from ..utils import ParserContext


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
		rows = table.find_all("tr")
		last_point = current_point
		for row in rows:
			cells = row.find_all("td")
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

	def parse_enumeration_spacing(block, current_parent: Dict, counters: Dict[str, int], last_host: Optional[Dict]) -> Optional[Dict]:
		items = block.find_all("p", recursive=False) or block.find_all("p")
		for item in items:
			text = item.get_text(" ", strip=True)
			if not text:
				continue
			parts = text.split(None, 1)
			if len(parts) == 2 and number_token.match(parts[0].rstrip(".")):
				counters["point"] += 1
				last_host = ctx.make_node(
					"annex_point",
					f"{current_parent['id'].split(f'{ctx.celex}_', 1)[-1]}_pt_{counters['point']}",
					parts[1],
					current_parent,
					number=parts[0].rstrip("."),
				)
			else:
				if last_host:
					last_host["text"] = f"{last_host['text']} {text}".strip()
				else:
					current_parent["text"] = f"{current_parent.get('text', '')} {text}".strip()
		return last_host

	def parse_annex_body(annex_node: Dict, annex_div) -> None:
		counters = {"section": 0, "point": 0, "subpoint": 0, "bullet": 0}
		current_parent = annex_node
		current_point: Optional[Dict] = None
		current_list_type = None  # "numbered", "lettered", "bullet"

		# Get all top-level blocks in order
		blocks = list(annex_div.find_all(["p", "div", "table"], recursive=False))

		for block in blocks:
			if block.name == "table":
				current_point = parse_table(block, current_parent, current_point, counters) or current_point
				continue

			text = block.get_text(" ", strip=True)
			if not text:
				continue

			# Section headings
			if block.name == "p" and "oj-ti-grseq-1" in (block.get("class") or []):
				counters["section"] += 1
				number = re.search(r"Section\s+([A-Za-z0-9]+)", text, re.I)
				current_parent = ctx.make_node(
					"annex_section",
					f"{annex_node['id'].split(f'{ctx.celex}_', 1)[-1]}_sec_{counters['section']}",
					text,
					annex_node,
					number=number.group(1) if number else None,
					title=text,
				)
				current_point = None
				current_list_type = None
				continue

			# Detect new list item
			lead_match = re.match(r"^(\d+|[a-zA-Z])\.?\s+", text)
			if lead_match:
				label = lead_match.group(1)
				content = text[lead_match.end():].strip()

				if label.isdigit():
					counters["point"] += 1
					current_point = ctx.make_node(
						"annex_point",
						f"{current_parent['id'].split(f'{ctx.celex}_', 1)[-1]}_pt_{counters['point']}",
						content,
						current_parent,
						number=label,
					)
					current_list_type = "numbered"
				else:  # letter
					counters["subpoint"] += 1
					host = current_point or current_parent
					ctx.make_node(
						"annex_subpoint",
						f"{host['id'].split(f'{ctx.celex}_', 1)[-1]}_ltr_{counters['subpoint']}",
						content,
						host,
						number=label.lower(),
					)
					current_list_type = "lettered"
				continue

			# Bullet / continuation
			if current_point and (text.startswith("—") or text.startswith("-") or text.startswith("•")):
				counters["bullet"] += 1
				host = current_point
				ctx.make_node(
					"annex_bullet",
					f"{host['id'].split(f'{ctx.celex}_', 1)[-1]}_blt_{counters['bullet']}",
					text.lstrip("—-• ").strip(),
					host,
				)
				continue

			# Continuation text (append to current point or section)
			if current_point:
				current_point["text"] = f"{current_point['text']} {text}".strip()
			else:
				current_parent["text"] = f"{current_parent.get('text', '')} {text}".strip()

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
