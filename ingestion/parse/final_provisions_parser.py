from __future__ import annotations

from typing import Dict, Optional

from .utils import ParserContext


def parse_final_provisions(soup, ctx: ParserContext, root: Dict) -> Optional[Dict]:
	final_div = soup.find("div", id="fnp_1")
	if not final_div:
		return None
	return ctx.make_node("final_provisions", "fnp_1", final_div.get_text(" ", strip=True), root)
