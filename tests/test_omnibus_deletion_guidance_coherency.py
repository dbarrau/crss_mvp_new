"""Guidance-prose coherency guard for Omnibus-deleted provisions.

The runtime graph holds *current* (consolidated) law, but hand-authored guidance
— curated legal-reasoning rationales and the answer prompt's worked examples —
can still teach a *deleted* provision as if it were live. That is exactly how a
correct-looking answer omits the surviving lex specialis: the model faithfully
follows stale curated guidance (real case: the special-category bias-detection
question routed through GDPR Article 9(2) via a rationale/prompt that still cited
the deleted AI Act Article 10(5), never surfacing its successor Article 4a).

The superseded runtime guard is a backstop on *output*; it never fires when our
own guidance steers the model *around* citing the deleted ref. This test closes
that upstream hole: a recorded-deleted provision may still be *named* in guidance,
but only in an explicitly historical context (a deletion marker nearby) — never
presented as current law.

Pure-static: reads the superseded record + two source surfaces; no Neo4j / LLM.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from domain.ontology.legal_reasoning_chains import _ALL_EDGES
from domain.ontology.superseded_provisions import SUPERSEDED

_REPO = Path(__file__).resolve().parent.parent

# A mention of a deleted provision is legitimate only when flagged as historical.
_DELETION_MARKERS = ("delet", "supersed", "former", "no longer", "repeal", "removed")

# Article-paragraph deletions only (e.g. "Article 10(5)"); annex-point refs are not
# cited in this prose and their shape is noisier to match.
_ART_PARA_RE = re.compile(r"^Article\s+(\d+[a-z]?)\((\d+[a-z]?)\)$")


def _deleted_art_para_refs() -> list[str]:
    refs = []
    for rec in SUPERSEDED:
        m = _ART_PARA_RE.match((rec.get("deleted_ref") or "").strip())
        if m:
            refs.append((m.group(1), m.group(2)))
    return refs


def _citation_pattern(art: str, para: str) -> re.Pattern:
    # Tolerant of internal whitespace: "Article 10(5)", "Article 10 ( 5 )".
    return re.compile(
        rf"\bArticle\s+{re.escape(art)}\s*\(\s*{re.escape(para)}\s*\)",
        re.IGNORECASE,
    )


def _presented_as_current(text: str, pat: re.Pattern) -> list[str]:
    """Return offending snippets: an occurrence of the deleted ref with no
    deletion marker within a ±160-char window."""
    bad = []
    for m in pat.finditer(text):
        window = text[max(0, m.start() - 160): m.end() + 160].lower()
        if not any(mark in window for mark in _DELETION_MARKERS):
            bad.append(text[max(0, m.start() - 60): m.end() + 60].strip())
    return bad


# The live guidance surfaces the model actually consumes.
_RATIONALES = "\n\n".join(e.rationale for e in _ALL_EDGES if e.rationale)
_PROMPT_SRC = (_REPO / "application" / "_prompts.py").read_text(encoding="utf-8")
_CHAINS_SRC = (_REPO / "domain" / "ontology" / "legal_reasoning_chains.py").read_text(
    encoding="utf-8"
)

_SURFACES = {
    "legal-reasoning rationales": _RATIONALES,
    "application/_prompts.py": _PROMPT_SRC,
    "legal_reasoning_chains.py (source)": _CHAINS_SRC,
}


@pytest.mark.parametrize("art,para", _deleted_art_para_refs())
@pytest.mark.parametrize("surface_name", list(_SURFACES))
def test_deleted_provision_not_taught_as_current(surface_name, art, para):
    text = _SURFACES[surface_name]
    pat = _citation_pattern(art, para)
    offenders = _presented_as_current(text, pat)
    assert not offenders, (
        f"{surface_name} cites deleted AI Act Article {art}({para}) as current law "
        f"(no deletion marker nearby). Point it at the surviving successor instead, "
        f"or mark the mention as historical. Offending context:\n  "
        + "\n  ".join(offenders)
    )


def test_guard_is_actually_exercised():
    """Fail loudly if the record ever stops yielding an article-paragraph deletion,
    which would make the parametrized test silently vacuous."""
    assert _deleted_art_para_refs(), (
        "No article-paragraph deletions in the superseded record — the coherency "
        "parametrization would be empty. Verify domain/ontology/superseded_provisions.py."
    )
