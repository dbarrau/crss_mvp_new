"""Legal-Markdown normalization for CRSS answers.

The generation model emits Markdown that renders well *enough* in a lenient
viewer but breaks in strict CommonMark: nested list items indented with a single
space (CommonMark needs >=2 to nest), point-lists run inline inside one line
("… which: (a) … (b) …"), a list that hugs the preceding paragraph with no blank
line (rendered as running text — the "list in plain text" symptom), and citation
runs where only the first reference is bold ("**Articles 26**, 27, 49").

This module is the SERVER/EXPORT-side twin of ``normalizeLegalLists`` in
demo/static/index.html: the browser normalizes for the live view, this normalizes
the same answer for the exported .md so both surfaces read identically. Keep the
two in sync when either changes.

Entry point: :func:`normalize_legal_markdown`.
"""
from __future__ import annotations

import re

_ROMAN = "ivxlcdm"
_IS_LIST_LINE = re.compile(r"^\s*(?:[-*+]\s|\d+\.\s)")
_CODE_FENCE = re.compile(r"^\s*```")


# ── Bold-reference runs ──────────────────────────────────────────────────────
# "**Articles 26**, 27, and 49(3)" → "**Articles 26, 27 and 49(3)**": extend a
# bolded Article/Annex/Recital lead over the comma/"and"-separated numeric
# enumeration that follows, so the whole citation run is bold — not just the
# first item. Stops at the first token that is not a reference number, so
# trailing prose ("… (if a public entity)") is left outside the bold.
# A reference number: a digit run (article/recital/paragraph, "10a", "26(3)")
# OR an UPPERCASE roman numeral (annex — "VIII"). Roman is uppercase-only, and
# the keyword is the only case-insensitive part (via (?i:…)), so a lowercase word
# whose letters happen to be roman ("did", "mix") can never match as a reference.
_REF_NUM = r"(?:\d+[a-z]?|[IVXLCDM]+)(?:\((?:\d+[a-z]?|[ivxlcdm]+|[a-z])\))*"
_BOLD_REF_RUN = re.compile(
    r"\*\*(?P<kw>(?i:Articles?|Annexe?s?|Recitals?|Paragraphs?|Points?))\s+"
    r"(?P<first>" + _REF_NUM + r")\*\*"
    r"(?P<rest>(?:\s*(?:,\s*and|,|and|or)\s*" + _REF_NUM + r")+)"
)


def _bold_reference_runs(md: str) -> str:
    return _BOLD_REF_RUN.sub(
        lambda m: f"**{m.group('kw')} {m.group('first')}{m.group('rest')}**", md
    )


# ── Heading glued to a run-in body part ──────────────────────────────────────
# The model occasionally omits the newline after a heading, running the next part
# straight into it, so the renderer swallows a whole sentence into the <h3>. Two
# NO-SPACE glue shapes are split back apart:
#   (1) enumerator run-in: a lowercase letter + single-capital/number + ". "
#       ("### Final AnswerA. The R&D exemption …");
#   (2) punctuation run-in: a "?" or ")" immediately followed by a capitalised
#       word ("… the AI Act?Key provision: …", "(Article 5)Banned outright …").
# Both are guarded to the NO-SPACE boundary so legitimate titles are never split:
# "### A. Scope" / "### Section A. Overview" (space before the letter), or a
# heading ending in ")" or "?" followed by a space or end-of-line.
_HEADING_GLUE_ENUM = re.compile(
    r"^(#{1,6}[ \t]+.*[a-z])((?:[A-Z]|\d{1,3})\.\s.+)$", re.MULTILINE
)
_HEADING_GLUE_PUNCT = re.compile(
    r"^(#{1,6}[ \t]+.*?[?)])([A-Z][a-z].+)$", re.MULTILINE
)


def _split_glued_headings(md: str) -> str:
    md = _HEADING_GLUE_ENUM.sub(r"\1\n\n\2", md)
    md = _HEADING_GLUE_PUNCT.sub(r"\1\n\n\2", md)
    return md


# ── Marker depth (roman-vs-letter continuity) ────────────────────────────────


def _marker_depth(inner: str, state: dict) -> int:
    """Depth of a parenthetical marker: 0 = (a)/(1) top, 1 = (i)/(ii) roman sub.

    ``state['alpha']`` carries the last letter so a lone ``(i)`` right after
    ``(h)`` reads as a continuing letter, not a new roman sub-list.
    """
    if inner.isdigit():
        return 0
    low = inner.lower()
    if len(low) > 1:
        return 1 if all(c in _ROMAN for c in low) else 0
    if low not in _ROMAN:
        state["alpha"] = ord(low)
        return 0
    if state.get("alpha") and ord(low) == state["alpha"] + 1:
        state["alpha"] = ord(low)
        return 0
    return 1


# ── Inline enumerator run → nested list ──────────────────────────────────────

_INLINE_SPLIT = re.compile(r"(?:^|[\s;:])(\((?:[A-Za-z]|[ivxlcdm]{2,4}|\d{1,3})\))\s", re.IGNORECASE)
_BQ_PREFIX = re.compile(r"^(\s*>[>\s]*)([\s\S]*)$")


def _expand_inline(line: str) -> list[str] | None:
    """Expand a line holding an INLINE run of legal enumerators into a nested
    list, preserving a leading blockquote prefix. Returns replacement lines, or
    ``None`` when there is no genuine run.

    Guarded so an isolated reference is never split: needs >=2 markers, at least
    one substantial (>40-char) item, AND every item's own text must be
    substantial — so "points (a) and (b)", "Directive (EU) 2019/790" and a
    reference chain ("points (a), (b) and (d) of the first subparagraph") all
    stay untouched (their items include a bare connector like "and"/"of").
    """
    bq = _BQ_PREFIX.match(line)
    prefix = bq.group(1) if bq else ""
    body = bq.group(2) if bq else line

    parts = _INLINE_SPLIT.split(body)
    if len(parts) < 3:
        return None
    items = []
    i = 1
    while i + 1 < len(parts):
        items.append({"marker": parts[i], "text": (parts[i + 1] or "").strip()})
        i += 2
    if len(items) < 2 or not any(len(it["text"]) > 40 for it in items):
        return None
    if any(len(re.sub(r"[.,;:]+$", "", it["text"]).strip()) < 8 for it in items):
        return None

    res: list[str] = []
    lead = parts[0].strip()
    if lead:
        res.append(prefix + lead)
        if prefix:
            res.append(prefix.rstrip())
    state = {"alpha": 0}
    for it in items:
        depth = _marker_depth(it["marker"][1:-1], state)
        res.append(prefix + "   " * depth + "- " + it["marker"] + " " + it["text"])
    return res


# ── Reindent native 1-space bullets to real nesting ──────────────────────────

_LEADING_MARKER = re.compile(r"^( +)((?:[-*+]|\d+\.)\s)")


def _reindent_list_line(s: str) -> str:
    """Re-expand each leading space of a native bullet/ordered marker into the
    3-space step marked needs to nest (0->0, 1->3, 2->6, …). The model signals a
    nested item with a SINGLE leading space, which CommonMark renders as a
    sibling (the "hierarchy lost" flatten)."""
    m = _LEADING_MARKER.match(s)
    return "   " * len(m.group(1)) + s[len(m.group(1)):] if m else s


# ── Parenthetical / dash marker lines ────────────────────────────────────────

_ORDERED = re.compile(r"^\s*\d+\.\s")
_MARKER_LINE = re.compile(r"^(\s*)(\((?:\d+|[A-Za-z]+)\)|[—–•‣·])\s+(.+)$")


def normalize_legal_markdown(md: str) -> str:
    """Rewrite legal-enumeration Markdown into strict, well-nested CommonMark.

    Faithful port of ``normalizeLegalLists`` (demo/static/index.html) plus the
    bold-reference-run pass, so the exported .md renders identically to the live
    view. Idempotent enough for export use; not intended to be applied twice.
    """
    if not md:
        return md
    md = _bold_reference_runs(md)
    md = _split_glued_headings(md)

    out: list[str] = []
    in_code = False
    last_alpha = 0
    for raw in md.split("\n"):
        if _CODE_FENCE.match(raw):
            in_code = not in_code
            out.append(raw)
            continue
        if in_code:
            out.append(raw)
            continue

        expanded = _expand_inline(raw)
        if expanded:
            out.extend(expanded)
            last_alpha = 0
            continue

        if _ORDERED.match(raw):
            last_alpha = 0
            out.append(_reindent_list_line(raw))
            continue

        m = _MARKER_LINE.match(raw)
        if not m:
            last_alpha = 0
            out.append(_reindent_list_line(raw))
            continue

        marker = m.group(2)
        label = marker + " "
        if re.match(r"^[—–•‣·]$", marker):
            depth, label = 2, ""            # indent bullet: drop the glyph
        else:
            inner = marker[1:-1]
            if inner.isdigit():
                depth = 0                    # (1) definition / top-level
            else:
                low = inner.lower()
                romanish = all(c in _ROMAN for c in low)
                if len(low) > 1:
                    depth = 2 if romanish else 1
                elif low not in _ROMAN:
                    depth = 1
                    last_alpha = ord(low)    # plain letter (a),(b)…
                elif last_alpha and ord(low) == last_alpha + 1:
                    depth = 1
                    last_alpha = ord(low)    # (i) continuing (h) → letter
                else:
                    depth = 2                # (i) opening a roman sub-list

        # Separate a new list from a preceding paragraph so marked starts it.
        prev = out[-1] if out else ""
        if prev != "" and not _IS_LIST_LINE.match(prev):
            out.append("")
        out.append("   " * depth + "- " + label + m.group(3))

    return "\n".join(out)
