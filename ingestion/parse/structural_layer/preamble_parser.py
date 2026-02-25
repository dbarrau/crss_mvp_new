from __future__ import annotations

import re
from typing import Dict, Optional

from ..utils import ParserContext


def parse_preamble(soup, ctx: ParserContext, root: Dict) -> Optional[Dict]:
	preamble_div = soup.find("div", id="pbl_1")
	if not preamble_div:
		return None

	preamble_node = ctx.make_node("preamble", "preamble", "", root)

	for cit_div in preamble_div.find_all("div", id=re.compile(r"^cit_\d+")):
		number = cit_div.get("id", "").split("_")[-1]
		ctx.make_node(
			"citation",
			cit_div["id"],
			cit_div.get_text(" ", strip=True),
			preamble_node,
			number=number,
		)

	recital_divs = preamble_div.find_all("div", id=re.compile(r"^rct_\d+"))
	if recital_divs:
		for rec_div in recital_divs:
			number = rec_div.get("id", "").split("_")[-1]
			ctx.make_node(
				"recital",
				rec_div["id"],
				rec_div.get_text(" ", strip=True),
				preamble_node,
				number=number,
			)
	else:
		for tr in preamble_div.find_all("tr"):
			cells = tr.find_all("td")
			if len(cells) < 2:
				continue
			number = cells[0].get_text(strip=True)
			if not number or not number[0].isdigit():
				continue
			text = cells[1].get_text(" ", strip=True)
			ctx.make_node("recital", f"rct_{number}", text, preamble_node, number=number)

	return preamble_node
