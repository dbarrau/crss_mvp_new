"""Guidance parse profiles.

A *parse profile* bundles the three publisher-specific inputs the otherwise
family-agnostic guidance engine (:func:`..mdcg_parser.parse_guidance_pdf`)
needs:

- ``custom_prompt`` — the LlamaParse agentic instruction block;
- ``clean_fn`` — post-processing of the raw markdown;
- ``extract_fn`` — optional structured extraction (e.g. MDCG flowcharts).

Adding a new guidance family is a new :data:`GuidanceProfile` here plus a
catalog entry naming it via ``parse_profile`` — no new code path downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from .mdcg_parser import (
    MDCG_CUSTOM_PROMPT,
    MDCG_HEADER_PATTERNS,
    CleanFn,
    ExtractFn,
    clean_guidance_markdown,
    extract_flowcharts,
)


@dataclass(frozen=True)
class GuidanceProfile:
    """Publisher-specific parse behaviour, injected into the parse engine.

    ``section_scheme`` selects how the structurer reads section numbering:
    ``"dotted_decimal"`` (MDCG: ``4.3.2``) or ``"roman_arabic"`` (AI Office:
    Roman top-level ``I., II.`` with Arabic subsections ``1., 2.``).
    """

    name: str
    custom_prompt: str
    clean_fn: CleanFn
    extract_fn: ExtractFn | None = None
    tier: str = "agentic"
    section_scheme: str = "dotted_decimal"


# ── AI Office prompt ────────────────────────────────────────────────────────
# AI Office guidelines are long (100+ pp.) interpretive texts on the AI Act.
# Unlike MDCG PDFs they have no "Medical Devices" running header and no
# lettered decision-tree flowcharts, so the profile drops those steps and the
# prompt is retargeted to the AI Office layout.

AI_OFFICE_CUSTOM_PROMPT = """
You are parsing an official guidance document published by the European
Commission's AI Office under Regulation (EU) 2024/1689 (the AI Act). These are
long interpretive guidance PDFs with numbered sections, footnotes, recital and
Article references, and worked examples.

CRITICAL INSTRUCTIONS:

1. CONTINUOUS DOCUMENT — Treat the entire PDF as ONE continuous document.
   Do NOT restart headings or numbering at page boundaries. The document may
   run to well over a hundred pages; keep one coherent heading hierarchy
   throughout.

2. REMOVE REPEATED HEADERS/FOOTERS — Running headers/footers (the document
   title, "AI Office", "European Commission", and bare page numbers) repeat on
   nearly every page. REMOVE every such repeated line. Keep the first, real
   occurrence of the document title as the level-1 heading.

3. SECTION HIERARCHY — Preserve the original numbered section structure exactly.
   Map section numbers to markdown heading levels:
   - Document title → # (level 1)
   - Top sections (1, 2, 3) → ## (level 2)
   - Subsections (3.1, 3.2) → ### (level 3)
   - Sub-subsections (3.1.2) → #### (level 4)
   - Deeper (3.1.2.1) → ##### (level 5)
   Unnumbered but clearly titled sections (e.g. "Executive summary",
   "Scope", "Annex") map to ## unless their size clearly makes them a subsection.

4. FOOTNOTES — Collect ALL footnotes from the entire document into a SINGLE
   section titled "## Footnotes" at the very end. Preserve the original footnote
   numbering (superscript numbers become plain "1.", "2.", …). Do NOT scatter
   footnotes at the end of each page — consolidate them ALL at the end.

5. LEGAL REFERENCES — Keep every legal reference exactly as written: Article
   and paragraph numbers, recital numbers, Annex references, and references to
   other EU acts (Regulations, Directives). Do not renumber or "correct" them.

6. FORMATTING — Preserve bold (**text**) and italic (*text*). Keep bullet and
   numbered lists as markdown lists, preserving their nesting.

7. EXAMPLES & BOXES — Worked examples, "Example" call-outs, and shaded boxes
   are substantive content: preserve them in full under a clear heading or as a
   blockquote, not as stray fragments.

8. TABLES — Render any tables as markdown tables, preserving all rows/columns.

9. CONTENT PRESERVATION — Do NOT omit, summarize, or paraphrase any content.
   Include every sentence, every example, every footnote from the original.
"""

# AI Office running headers/footers to dedupe (heading- and plain-line forms).
AI_OFFICE_HEADER_PATTERNS = [
    r"^#{1,3}\s*AI Office\s*$",
    r"^AI Office\s*$",
    r"^#{1,3}\s*European Commission\s*$",
    r"^European Commission\s*$",
]


# ── Profile registry ────────────────────────────────────────────────────────

_PROFILES: dict[str, GuidanceProfile] = {
    "mdcg": GuidanceProfile(
        name="mdcg",
        custom_prompt=MDCG_CUSTOM_PROMPT,
        clean_fn=partial(clean_guidance_markdown, header_patterns=MDCG_HEADER_PATTERNS),
        extract_fn=extract_flowcharts,
    ),
    "ai_office": GuidanceProfile(
        name="ai_office",
        custom_prompt=AI_OFFICE_CUSTOM_PROMPT,
        clean_fn=partial(
            clean_guidance_markdown, header_patterns=AI_OFFICE_HEADER_PATTERNS
        ),
        extract_fn=None,  # no lettered decision-tree flowcharts in AI Office docs
        # AI Office docs are not uniform: the high-risk trio uses Roman
        # top-level sections, the transparency guidance uses dotted-decimal.
        # "auto" lets the structurer detect per document.
        section_scheme="auto",
    ),
}


def get_profile(name: str) -> GuidanceProfile:
    """Return the parse profile for *name* (e.g. ``"mdcg"``, ``"ai_office"``)."""
    try:
        return _PROFILES[name]
    except KeyError:
        raise ValueError(
            f"Unknown guidance parse profile {name!r}. "
            f"Known profiles: {sorted(_PROFILES)}"
        ) from None
