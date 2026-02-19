from ....domain.ontology.lang_keywords import LANG_KEYWORDS
from typing import List
import re

# ============================================================
# Cross-reference extraction
# ============================================================
def extract_references(text: str, lang: str) -> List[str]:
    """Extract cross-references to Articles and Annexes from text.

    The function works in a multilingual way by leveraging
    :data:`LANG_KEYWORDS` for the requested language.

    :param text: Text possibly containing references (e.g. "see Article 5").
    :param lang: Language code ("EN", "FR", "DE").
    :return: List of normalized reference identifiers, e.g. "Article_5", "Annex_IV".
    """
    keywords = LANG_KEYWORDS.get(lang.upper(), LANG_KEYWORDS["EN"])
    article_kw = keywords["article"]
    annex_kw = keywords["annex"]

    refs: List[str] = []
    refs += [f"Article_{m}" for m in re.findall(rf"{article_kw}\s+(\d+)", text, re.I)]
    refs += [f"Annex_{m}" for m in re.findall(rf"{annex_kw}\s+([IVXLCDM]+)", text, re.I)]
    return refs
