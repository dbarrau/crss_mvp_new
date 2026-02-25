# ingestion/parse/universal_eurlex_parser.py

"""EUR-Lex universal parser aligned with CRSS GraphRAG shape."""

from __future__ import annotations

from typing import Dict

from bs4 import BeautifulSoup

from .structural_layer.annex_parser import parse_annexes
from .structural_layer.enacting_terms_parser import parse_enacting_terms
from .final_provisions_parser import parse_final_provisions
from .structural_layer.preamble_parser import parse_preamble
from .utils import ParserContext


# Public entry point used by the registry
def parse_eurlex_html(html_content: str, celex: str, regulation_id: str, lang: str = "EN") -> Dict:
	soup = BeautifulSoup(html_content, "html.parser")
	ctx = ParserContext(celex=celex, lang=lang)

	# Root document node
	main_title = None
	main_title_div = soup.find("div", class_="eli-main-title", id="tit_1")
	if main_title_div:
		main_title = main_title_div.get_text(" ", strip=True)
	root = ctx.make_node("document", "document", main_title or regulation_id or celex, None)

	parse_preamble(soup, ctx, root)
	parse_enacting_terms(soup, ctx, root)
	parse_final_provisions(soup, ctx, root)
	parse_annexes(soup, ctx, root)

	return {"provisions": ctx.provisions, "relations": ctx.relations}
