"""EUR-Lex HTML structure knowledge.

This module is the single source of truth for all HTML element IDs, CSS class
names, and structural ID patterns used to navigate EUR-Lex documents published
on the Official Journal (OJ) format.

Grouping:
  - SECTION IDS:   Static document-level IDs (one per document).
  - ID PATTERNS:   Compiled regex patterns for repeating structural elements.
  - CSS CLASSES:   OJ/ELI typography and layout class names.
  - HTML ATTRS:    Other HTML attribute values used to locate content.
"""

from __future__ import annotations

import re

# ── Section IDs (static, one-per-document) ───────────────────────────────────

MAIN_TITLE_ID = "tit_1"
PREAMBLE_ID = "pbl_1"
ENACTING_TERMS_ID = "enc_1"
FINAL_PROVISIONS_ID = "fnp_1"

# Template: substitute {id} with the parent element's id value.
ARTICLE_TITLE_ID_TEMPLATE = "{id}.tit_1"

# Annex whose content is the entry-into-force signature block — skip it.
ANNEX_SKIP_ID = "anx_ES"


# ── ID patterns (repeating structural elements) ───────────────────────────────

CITATION_ID_RE = re.compile(r"^cit_\d+")
RECITAL_ID_RE = re.compile(r"^rct_\d+")
CHAPTER_ID_RE = re.compile(r"^cpt_([IVXLCDM]+)$")
SECTION_ID_RE = re.compile(r"^cpt_([IVXLCDM]+)\.sct_(\d+)$")
ARTICLE_ID_RE = re.compile(r"^art_(\d+[a-z]?)$")
PARAGRAPH_ID_RE = re.compile(r"^(\d{3}[a-z]?)\.(\d{3}[a-z]?)$")
ANNEX_ID_RE = re.compile(r"^anx_[A-Za-z0-9]+$")


# ── CSS class names (OJ / ELI typography) ────────────────────────────────────

# ELI (European Legislation Identifier) layout classes
CLASS_ELI_MAIN_TITLE = "eli-main-title"
CLASS_ELI_CONTAINER = "eli-container"

# OJ typography classes
CLASS_OJ_DOC_TI = "oj-doc-ti"             # Annex document title
CLASS_OJ_TI_GRSEQ_1 = "oj-ti-grseq-1"   # Group/section heading
CLASS_OJ_NORMAL = "oj-normal"             # Normal body paragraph
CLASS_OJ_ENUMERATION_SPACING = "oj-enumeration-spacing"  # Enumeration block

# Two body-prose classes some consolidated OJ layouts use instead of "oj-normal":
CLASS_NORMAL = "normal"                   # body paragraph without the oj- prefix
CLASS_LIST = "list"                       # continuation subparagraph after a point-list

# Classes EUR-Lex uses to carry article/paragraph body prose. Treating only
# "oj-normal" as body silently dropped real obligations where a consolidated
# layout dropped the prefix ("normal", IVDR Art 112 intro / Art 110 orphan) or
# tagged a post-point-list continuation subparagraph "list" (MDR Art 117). A
# "normal"/"oj-normal" element opens a subparagraph; a "list" element continues
# the current one (see BODY_SUBPARA_CLASSES).
BODY_SUBPARA_CLASSES = frozenset({CLASS_OJ_NORMAL, CLASS_NORMAL})   # open a subparagraph
BODY_TEXT_CLASSES = frozenset({CLASS_OJ_NORMAL, CLASS_NORMAL, CLASS_LIST})  # any body prose


# ── HTML attribute values ─────────────────────────────────────────────────────

# <table width="100%"> is the EUR-Lex convention for point/sub-item tables
# inside enacting-terms article paragraphs.
TABLE_POINTS_WIDTH = "100%"
