"""Normalize consolidated EUR-Lex HTML to match expected legal-basis structure.

Consolidated EUR-Lex documents (CELEX starting with ``0``) use a different HTML
vocabulary than the original Official Journal (OJ) format that the universal
parser was designed for. This module transforms the consolidated HTML *before*
it reaches the parser, so the parser itself requires no changes.

Key transformations
-------------------
1. CSS class renaming (``norm`` → ``oj-normal``, ``title-gr-seq-level-*`` →
   ``oj-ti-grseq-1``, etc.).
2. **Paragraph reconstruction** — consolidated docs put paragraphs as plain
   ``<div class="norm">`` with a ``<span class="no-parag">N.  </span>`` for
   the number.  We wrap these in ``<div id="ART.PAR">`` containers with the
   ``NNN.NNN`` format the parser expects.
3. **Grid-list → table** — points use CSS-grid ``<div class="grid-container
   grid-list">`` instead of ``<table width="100%">``.
4. **Amendment marker removal** — ``<p class="modref">``, ``<p class="arrow">``
   and other consolidated-only annotations are stripped.
5. **Article ID extension** — ``art_10a``-style IDs added by amendments are
   handled by extending ``ARTICLE_ID_RE``.
6. **Annex class mapping** — ``title-doc-first`` / ``title-doc-last`` →
   ``oj-doc-ti``, ``title-annex-1`` → ``oj-doc-ti``, etc.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString, Tag


# ── Detection ─────────────────────────────────────────────────────────────────

# Consolidated HTML class names that never appear in the OJ format.
_CONSOL_ONLY_CLASSES = {"norm", "modref", "grid-container", "grid-list", "arrow"}


def is_consolidated_html(html: str) -> bool:
    """Quick heuristic: does the HTML look like a consolidated EUR-Lex doc?"""
    # Check for a handful of consolidated-only markers.
    return ('class="norm"' in html or 'class="modref"' in html
            or 'class="grid-container' in html)


# ── Class mapping ─────────────────────────────────────────────────────────────

_CLASS_MAP: dict[str, str] = {
    "norm": "oj-normal",
    "boldface": "oj-bold",
    "italics": "oj-italic",
    "tbl-norm": "oj-table",
    "footnote": "oj-note",
    "superscript": "oj-super",
    "separator": "oj-separator",
    "separator-annex": "oj-doc-sep",
    "separator-short": "oj-separator",
    "stitle-article-norm": "oj-sti-art",
    "title-article-norm": "oj-ti-art",
}

# Multiple consolidated heading classes → single oj-ti-grseq-1
_HEADING_CLASSES = {
    "title-gr-seq-level-1",
    "title-gr-seq-level-2",
    "title-gr-seq-level-3",
    "title-gr-seq-level-4",
    "title-gr-seq-level-5",
    "title-division-1",
    "title-division-2",
}

# Annex title classes → oj-doc-ti
_ANNEX_TITLE_CLASSES = {
    "title-doc-first",
    "title-doc-last",
    "title-annex-1",
}

# Classes to strip entirely (amendment markers / ToC / navigation)
_STRIP_CLASSES = {
    "modref",
    "arrow",
    "hd-modifiers",
    "disclaimer",
    "hd-toc-1",
    "hd-toc-2",
    "hd-toc-3",
    "toc-1",
    "toc-2",
    "toc-item",
    "anchorarrow",
}


def _remap_classes(tag: Tag) -> None:
    """Replace consolidated CSS classes with their OJ equivalents in-place."""
    classes = tag.get("class")
    if not classes:
        return
    new_classes: list[str] = []
    for cls in classes:
        if cls in _CLASS_MAP:
            new_classes.append(_CLASS_MAP[cls])
        elif cls in _HEADING_CLASSES:
            new_classes.append("oj-ti-grseq-1")
        elif cls in _ANNEX_TITLE_CLASSES:
            new_classes.append("oj-doc-ti")
        else:
            new_classes.append(cls)
    tag["class"] = new_classes


# ── Amendment marker removal ──────────────────────────────────────────────────

_AMENDMENT_TEXT_RE = re.compile(r"^[▼►][A-Z0-9]+")
_INLINE_MARKER_RE = re.compile(r"[►◄]")


def _strip_amendment_markers(soup: BeautifulSoup) -> None:
    """Remove amendment source markers (▼B, ►M1, etc.) and their containers.

    This includes:
    - Block-level ``<p class="modref">`` / ``<p class="arrow">`` elements
    - Inline ``<span class="boldface">`` containing ► or ◄ characters
    - Parent ``<a>`` elements and ``<span>`` wrappers left empty after removal
    """
    # 1. Remove block-level markers by class
    for cls in _STRIP_CLASSES:
        for el in soup.find_all(class_=cls):
            el.decompose()

    # 2. Remove inline boldface spans containing ► or ◄ markers
    for bf in soup.find_all("span", class_="boldface"):
        if _INLINE_MARKER_RE.search(bf.get_text()):
            bf.decompose()

    # 3. Remove <a> tags that linked to amendment acts and are now empty
    #    (their boldface child was just decomposed)
    for a_tag in soup.find_all("a"):
        if not a_tag.get_text(strip=True):
            a_tag.decompose()

    # 4. Remove <span> wrappers that are now empty after child removal
    for span in soup.find_all("span"):
        if not span.get_text(strip=True) and not span.find(True):
            span.decompose()

    # 5. Merge adjacent NavigableString nodes that were separated by markers
    _merge_text_nodes(soup)


def _merge_text_nodes(soup: BeautifulSoup) -> None:
    """Merge adjacent NavigableString nodes within the same parent.

    After stripping inline amendment markers, text like
    ``"From " + "26 May 2021" + ", any publication..."``
    needs to be combined into a single text node so the parser
    does not create spurious subparagraphs.
    """
    for tag in soup.find_all(True):
        children = list(tag.children)
        i = 0
        while i < len(children) - 1:
            if isinstance(children[i], NavigableString) and isinstance(children[i + 1], NavigableString):
                merged = NavigableString(str(children[i]) + str(children[i + 1]))
                children[i].replace_with(merged)
                children[i + 1].extract()
                children = list(tag.children)
            else:
                i += 1


# ── Grid-list → table conversion ─────────────────────────────────────────────

def _convert_grid_lists(soup: BeautifulSoup) -> None:
    """Convert ``<div class="grid-container grid-list">`` into
    ``<table width="100%"><tr><td>marker</td><td>body</td></tr></table>``."""
    for grid in soup.find_all("div", class_="grid-list"):
        col1 = grid.find(class_="grid-list-column-1")
        col2 = grid.find(class_="grid-list-column-2")
        if not col1 or not col2:
            continue

        marker_text = col1.get_text(strip=True)
        # Build the body — prefer inner <p> content, otherwise full text
        body_ps = col2.find_all("p")

        table = soup.new_tag("table", border="0", cellpadding="0",
                             cellspacing="0", width="100%")
        tbody = soup.new_tag("tbody")
        tr = soup.new_tag("tr")

        td_marker = soup.new_tag("td", valign="top")
        p_marker = soup.new_tag("p")
        p_marker["class"] = ["oj-normal"]
        p_marker.string = marker_text
        td_marker.append(p_marker)

        td_body = soup.new_tag("td", valign="top")
        if body_ps:
            for p in body_ps:
                _remap_classes(p)
                td_body.append(p.extract())
        else:
            p_body = soup.new_tag("p")
            p_body["class"] = ["oj-normal"]
            p_body.string = col2.get_text(" ", strip=True)
            td_body.append(p_body)

        # Also convert any nested grid-lists that are still inside col2
        # (they will have been moved into td_body)
        for nested_grid in td_body.find_all("div", class_="grid-list"):
            _convert_single_grid_to_table(nested_grid, soup)

        tr.append(td_marker)
        tr.append(td_body)
        tbody.append(tr)
        table.append(tbody)

        grid.replace_with(table)


def _convert_single_grid_to_table(grid: Tag, soup: BeautifulSoup) -> None:
    """Convert a single nested grid-list to table (for roman items etc.)."""
    col1 = grid.find(class_="grid-list-column-1")
    col2 = grid.find(class_="grid-list-column-2")
    if not col1 or not col2:
        return

    marker_text = col1.get_text(strip=True)
    table = soup.new_tag("table", border="0", cellpadding="0",
                         cellspacing="0", width="100%")
    tbody = soup.new_tag("tbody")
    tr = soup.new_tag("tr")

    td_marker = soup.new_tag("td", valign="top")
    p_marker = soup.new_tag("p")
    p_marker["class"] = ["oj-normal"]
    p_marker.string = marker_text
    td_marker.append(p_marker)

    td_body = soup.new_tag("td", valign="top")
    body_ps = col2.find_all("p")
    if body_ps:
        for p in body_ps:
            _remap_classes(p)
            td_body.append(p.extract())
    else:
        p_body = soup.new_tag("p")
        p_body["class"] = ["oj-normal"]
        p_body.string = col2.get_text(" ", strip=True)
        td_body.append(p_body)

    tr.append(td_marker)
    tr.append(td_body)
    tbody.append(tr)
    table.append(tbody)
    grid.replace_with(table)


# ── Paragraph reconstruction ─────────────────────────────────────────────────

_PARA_NUM_RE = re.compile(r"^(\d+[a-z]?)\.\s*$")


def _reconstruct_paragraphs(soup: BeautifulSoup) -> None:
    """Wrap article paragraph ``<div class="norm">`` blocks with proper
    ``<div id="ART.PAR">`` containers carrying the ``NNN.NNN`` format."""
    article_re = re.compile(r"^art_(\d+[a-z]?)$")

    for article_div in soup.find_all("div", id=article_re):
        art_match = article_re.match(article_div["id"])
        if not art_match:
            continue
        art_num_raw = art_match.group(1)
        # Numeric article number for the NNN prefix (zero-padded to 3 digits)
        art_num = int(re.match(r"(\d+)", art_num_raw).group(1))

        # Collect direct child <div class="oj-normal"> (after class remap)
        # that contain a <span class="no-parag"> with a paragraph number.
        para_counter = 0
        for child in list(article_div.children):
            if not isinstance(child, Tag):
                continue
            # After class remap: look for <div class="oj-normal"> or <div class="norm">
            child_classes = child.get("class", [])
            if child.name != "div":
                continue
            if "oj-normal" not in child_classes and "norm" not in child_classes:
                continue

            # Look for <span class="no-parag">N.  </span>
            span = child.find("span", class_="no-parag")
            if not span:
                continue
            span_text = span.get_text(strip=True)
            m = _PARA_NUM_RE.match(span_text)
            if not m:
                continue

            para_label = m.group(1)  # e.g. "3" or "3a"
            # For numeric-only labels use the number; for "3a" keep as-is
            try:
                para_num_int = int(para_label)
                para_id = f"{art_num:03d}.{para_num_int:03d}"
            except ValueError:
                # e.g. "3a" → "120.003a"
                num_part = int(re.match(r"(\d+)", para_label).group(1))
                suffix = re.search(r"([a-z]+)$", para_label).group(1)
                para_id = f"{art_num:03d}.{num_part:03d}{suffix}"
            para_counter += 1

            # Build a new wrapper div
            wrapper = soup.new_tag("div", id=para_id)

            # Remove the span (the parser extracts numbers from the ID, not text)
            span.decompose()

            # Check if there's an inline-element div wrapping the actual content
            inline_div = child.find("div", class_="inline-element", recursive=False)
            if inline_div:
                # Flatten all content (including inline tags) into a single <p class="oj-normal">
                p = soup.new_tag("p")
                p["class"] = ["oj-normal"]
                # Use decode_contents() to preserve all inline tags (e.g. <span class="italics">)
                p.append(BeautifulSoup(inline_div.decode_contents(), "html.parser"))
                wrapper.append(p)
                # Also move any tables or block elements after the inline_div
                for sib in list(child.children):
                    if isinstance(sib, Tag) and sib is not inline_div:
                        if sib.name == "table":
                            wrapper.append(sib.extract())
            else:
                # No inline-element wrapper — content is directly in the div
                # Move tables out, flatten the rest into a single <p class="oj-normal">
                for ic in list(child.children):
                    if isinstance(ic, Tag) and ic.name == "table":
                        wrapper.append(ic.extract())
                # After extracting tables, flatten the rest
                text_html = child.decode_contents(formatter="html").strip()
                if text_html:
                    p = soup.new_tag("p")
                    p["class"] = ["oj-normal"]
                    p.append(BeautifulSoup(text_html, "html.parser"))
                    wrapper.append(p)

            child.replace_with(wrapper)


# ── Annex normalization ───────────────────────────────────────────────────────

def _normalize_annexes(soup: BeautifulSoup) -> None:
    """Ensure annex containers use the ``eli-container`` class."""
    annex_re = re.compile(r"^anx_[A-Za-z0-9]+$")
    for anx in soup.find_all("div", id=annex_re):
        classes = anx.get("class", [])
        if "eli-container" not in classes:
            anx["class"] = classes + ["eli-container"]


# ── Enumeration spacing ──────────────────────────────────────────────────────

def _convert_enumeration_divs(soup: BeautifulSoup) -> None:
    """Handle ``<div class="oj-enumeration-spacing">`` from either format."""
    # These are already handled by the annex parser, nothing extra needed.
    pass


# ── Master entry point ────────────────────────────────────────────────────────

def normalize_consolidated_html(html: str) -> str:
    """Transform consolidated EUR-Lex HTML to match the OJ format the parser
    expects. Returns the HTML unchanged if it already looks like OJ format.

    This is called by the dispatcher *before* the parser sees the HTML.
    """
    if not is_consolidated_html(html):
        return html

    soup = BeautifulSoup(html, "html.parser")

    # 1. Strip amendment markers (must come first — before class remap)
    _strip_amendment_markers(soup)

    # 2. Convert grid-lists → tables (before class remap, since we need
    #    the grid-list classes to identify them)
    _convert_grid_lists(soup)

    # 3. Remap all CSS classes
    for tag in soup.find_all(True):
        _remap_classes(tag)

    # 4. Reconstruct paragraph wrapper divs with NNN.NNN IDs
    _reconstruct_paragraphs(soup)

    # 5. Normalize annex containers
    _normalize_annexes(soup)

    return str(soup)
