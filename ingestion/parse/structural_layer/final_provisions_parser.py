from __future__ import annotations

from typing import Dict, Optional

from ..base.utils import ParserContext
from domain.ontology.eurlex_html import FINAL_PROVISIONS_ID


def parse_final_provisions(soup, ctx: ParserContext, root: Dict) -> Optional[Dict]:
	final_div = soup.find("div", id=FINAL_PROVISIONS_ID)
	if not final_div:
		return None
	return ctx.make_node("final_provisions", "fnp_1", final_div.get_text(" ", strip=True), root)
