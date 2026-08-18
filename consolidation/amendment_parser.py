"""Stage 1 — parse an amending act's raw HTML into structured instructions.

An amending regulation's operative Article ("Amendments to Regulation (EU) N")
is a numbered list of surgical instructions.  EUR-Lex encodes them as nested
two-column tables interleaving *directive* prose ("in Article 6, the following
paragraphs are inserted:") with the *content* it enacts (a curly-quoted block of
ordinary provision markup).  The flat ``parsed.json`` folds those quoted blocks
away; this parser goes back to the raw HTML and recovers the full structure.

The grammar is small and recursive:

  * **amend-as-follows** — a container ("Article 5 is amended as follows:")
    whose sub-parts (a),(b),… are *themselves* directives (and may nest again,
    e.g. Article 113's "the third paragraph is amended as follows: point (c) is
    replaced by the following: …").
  * **replace / insert / add** — carry a curly-quoted content payload of one or
    more provision units (paragraphs 1a/1b, points (ba)/(bb), roman (i)/(ii)).
  * **delete** — names a target and carries no content.

Output is a list of :class:`AmendmentInstruction` (one per top-level numbered
point), each flattened to a list of concrete :class:`Operation`s.  It is pure
structure — no graph ids yet; Stage 2 resolves each ``target_ref`` onto the base
regulation's nodes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from bs4 import BeautifulSoup


# ── serializable model ───────────────────────────────────────────────────────

@dataclass
class ContentNode:
    """One unit of enacted text: a paragraph, point, roman sub-item, etc.

    ``enumerator`` is the bare label as it will read in the consolidated act
    ("1a", "ba", "a", "i", "" for an unlabelled chapeau/body).  Children are the
    unit's own nested sub-items, in document order.
    """
    enumerator: str
    kind: str
    text: str
    children: List["ContentNode"] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"enumerator": self.enumerator, "kind": self.kind, "text": self.text}
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class Operation:
    """A single surgical change to one target in the base regulation."""
    op: str                     # "replace" | "insert" | "add" | "delete"
    item_kind: str              # article | paragraph | subparagraph | point | annex | section
    target_ref: str             # human-readable locator into the BASE act
    content: List[ContentNode] = field(default_factory=list)
    directive: str = ""         # the raw directive prose (audit trail)
    target: dict = field(default_factory=dict)   # structured locator for the applier

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "item_kind": self.item_kind,
            "target_ref": self.target_ref,
            "target": self.target,
            "directive": self.directive,
            "content": [c.to_dict() for c in self.content],
        }


@dataclass
class AmendmentInstruction:
    """One top-level numbered point of the amending Article, flattened to ops."""
    point_num: str
    target_celex: str
    operations: List[Operation] = field(default_factory=list)
    raw_lead: str = ""

    def to_dict(self) -> dict:
        return {
            "point_num": self.point_num,
            "target_celex": self.target_celex,
            "raw_lead": self.raw_lead,
            "operations": [o.to_dict() for o in self.operations],
        }


# ── HTML helpers (tbody-tolerant; EUR-Lex two-column tables) ──────────────────

# Quotation marks EUR-Lex uses to delimit an amendment's enacted content.
# U+2018/U+2019 are the OJ convention; the rest are defensive.
_QUOTE_OPENERS = ("‘", "“", "«", "'")
_QUOTE_CLOSERS = ("’", "”", "»", "'")


def _norm(s: Optional[str]) -> str:
    """Collapse all whitespace (incl. EUR-Lex non-breaking spaces) to single spaces."""
    return re.sub(r"\s+", " ", s or "").strip()


def _rows(table) -> List:
    body = table.find("tbody", recursive=False) or table
    return body.find_all("tr", recursive=False)


def _cells(tr) -> List:
    return tr.find_all("td", recursive=False)


def _direct_child_tables(el) -> List:
    """Tables whose nearest enclosing table/div is ``el`` itself."""
    return el.find_all("table", recursive=False)


def _direct_paras(td) -> str:
    """A cell's own body text — its direct ``<p>`` children, or (when the cell
    holds bare text with no ``<p>``, as annex points do) its direct text with
    nested tables excluded."""
    paras = _norm(" ".join(p.get_text(" ", strip=True) for p in td.find_all("p", recursive=False)))
    if paras:
        return paras
    parts = []
    for ch in td.children:
        name = getattr(ch, "name", None)
        if name == "table":
            continue
        parts.append(ch.get_text(" ", strip=True) if name else str(ch))
    return _norm(" ".join(parts))


def _element_children(el) -> List:
    return [c for c in el.children if getattr(c, "name", None)]


def _strip_quotes(s: str) -> str:
    """Peel the amendment block's *trailing* delimiter (closing quote and/or the
    ';' EUR-Lex places between sub-parts) off a unit's text.  The *leading*
    delimiter is already removed by :func:`_split_enumerator`, so any leading
    quote here belongs to the text (e.g. a defined term) and is kept."""
    s = s.strip()
    while s and (s[-1] in _QUOTE_CLOSERS or s[-1] == ";"):
        s = s[:-1].rstrip()
    return s.strip()


def _is_blank_unit(enum: str, text: str, children: list) -> bool:
    """A unit carrying no enumerator, no text, and no children — the leftover of
    a bare ';' separator between amendment sub-parts."""
    return not enum and not text and not children


# ── directive classification ─────────────────────────────────────────────────

# The five operative verbs.  Order matters: "amended as follows" is a container
# that must win over the bare verb inside its quoted content.
_RE_AMEND = re.compile(r"\bis amended as follows\b", re.I)
_RE_REPLACE = re.compile(r"\b(?:is|are)\s+replaced by the following\b", re.I)
_RE_INSERT = re.compile(r"\b(?:is|are)\s+inserted\b", re.I)
_RE_ADD = re.compile(r"\b(?:is|are)\s+added\b", re.I)
_RE_DELETE = re.compile(r"\b(?:is|are)\s+deleted\b", re.I)


def _classify_verb(directive: str) -> str:
    d = _norm(directive)
    if _RE_AMEND.search(d):
        return "amend"
    if _RE_REPLACE.search(d):
        return "replace"
    if _RE_INSERT.search(d):
        return "insert"
    if _RE_ADD.search(d):
        return "add"
    if _RE_DELETE.search(d):
        return "delete"
    return "unknown"


# The noun an insert/add enacts ("the following <KIND>s are inserted").
_NEW_ITEM_RE = re.compile(
    r"the following\s+(sub-?paragraph|paragraph|point|article|annex|section)s?\b", re.I)


def _new_item_kind(directive: str) -> str:
    m = _NEW_ITEM_RE.search(_norm(directive))
    return (m.group(1).replace("-", "").lower() if m else "provision")


# ── structured target locator ────────────────────────────────────────────────

_ARTICLE_RE = re.compile(r"\bArticle\s+(\d+[a-z]?)\b", re.I)
_ARTICLE_PARA_RE = re.compile(r"\bArticle\s+(\d+[a-z]?)\((\d+[a-z]?)\)", re.I)
_ANNEX_RE = re.compile(r"\bAnnex\s+([IVXLCDM]+|\d+)\b", re.I)
_ORDINALS = ("first", "second", "third", "fourth", "fifth",
             "sixth", "seventh", "eighth", "ninth", "tenth")


@dataclass
class Locator:
    """A structured path into the base act, built up across the recursion.

    A nested directive contributes only the segments it names ("point (a)");
    :meth:`merged_under` fills the rest from the inherited container scope, so
    "point (a)" inside "Article 113, the third paragraph is amended as follows"
    resolves to *Article 113, third paragraph, point (a)* — no context lost.
    """
    article: str = ""
    annex: str = ""
    para: str = ""          # numbered paragraph → rendered as (N)
    para_ord: str = ""      # ordinal paragraph ("third") when unnumbered
    subpara: str = ""       # ordinal ("first") or number
    section: str = ""       # annex section ("A"/"B")
    intro: bool = False     # the introductory part (chapeau)
    heading: bool = False   # the article heading (title), not its body
    point: str = ""         # "a" / "ba" / "14" / "1"

    def merged_under(self, parent: "Locator") -> "Locator":
        sets_para = bool(self.para or self.para_ord)
        return Locator(
            article=self.article or parent.article,
            annex=self.annex or parent.annex,
            para=self.para if sets_para else parent.para,
            para_ord=self.para_ord if sets_para else parent.para_ord,
            subpara=self.subpara or parent.subpara,
            section=self.section or parent.section,
            intro=self.intro or parent.intro,
            heading=self.heading or parent.heading,
            point=self.point or parent.point,
        )

    def with_leaf(self, kind: str, value: str) -> "Locator":
        d = Locator(**vars(self))
        if kind == "point":
            d.point = value
        elif kind == "paragraph":
            d.para, d.para_ord = value, ""
        return d

    def leaf_kind(self) -> str:
        if self.heading:
            return "heading"
        if self.point:
            return "point"
        if self.intro:
            return "chapeau"
        if self.subpara:
            return "subparagraph"
        if self.para or self.para_ord:
            return "paragraph"
        if self.section:
            return "section"
        if self.article:
            return "article"
        if self.annex:
            return "annex"
        return "provision"

    def render(self) -> str:
        if self.article:
            head = f"Article {self.article}"
            if self.para:
                head += f"({self.para})"
        elif self.annex:
            head = f"Annex {self.annex}"
        else:
            head = ""
        segs: List[str] = [head] if head else []
        if self.para_ord:
            segs.append(f"{self.para_ord} paragraph")
        if self.section:
            segs.append(f"Section {self.section}")
        if self.subpara:
            segs.append(f"subparagraph {self.subpara}" if self.subpara.isdigit()
                        else f"{self.subpara} subparagraph")
        if self.intro:
            segs.append("introductory part")
        if self.heading:
            segs.append("heading")
        if self.point:
            segs.append(f"point {self.point}" if self.annex else f"point ({self.point})")
        return ", ".join(segs)

    def to_fields(self) -> dict:
        """The non-empty segments, for the applier to resolve without re-parsing."""
        return {k: v for k, v in vars(self).items() if v}


def _parse_locator(directive: str) -> Locator:
    """Extract every structural segment a directive names, EXCEPT the new item
    an insert/add enacts ("the following point …" is a kind, not a locator)."""
    d = _norm(directive)
    d = re.sub(r",?\s*the following\b.*$", "", d, flags=re.I)   # drop the new-item phrase
    loc = Locator()
    m = _ARTICLE_PARA_RE.search(d)
    if m:
        loc.article, loc.para = m.group(1), m.group(2)
    else:
        m = _ARTICLE_RE.search(d)
        if m:
            loc.article = m.group(1)
    m = _ANNEX_RE.search(d)
    if m:
        loc.annex = m.group(1)
    if not loc.para:
        m = re.search(r"\bparagraph\s+(\d+[a-z]?)\b", d, re.I)
        if m:
            loc.para = m.group(1)
    m = re.search(r"\bthe\s+(\w+)\s+paragraph\b", d, re.I)
    if m and m.group(1).lower() in _ORDINALS and not loc.para:
        loc.para_ord = m.group(1).lower()
    m = re.search(r"\bthe\s+(\w+)\s+subparagraph\b", d, re.I)
    if m and m.group(1).lower() in _ORDINALS:
        loc.subpara = m.group(1).lower()
    else:
        m = re.search(r"\bsubparagraph\s+(\d+)\b", d, re.I)
        if m:
            loc.subpara = m.group(1)
    m = re.search(r"\bSection\s+([A-Z0-9]+)\b", d, re.I)
    if m:
        loc.section = m.group(1).upper()
    if re.search(r"\bintroductory part\b", d, re.I):
        loc.intro = True
    if re.search(r"\bheading\b", d, re.I):
        loc.heading = True
    m = re.search(r"\bpoint\s+\(([^)]+)\)", d, re.I)
    if m:
        loc.point = m.group(1)
    elif re.search(r"\bpoint\s+\d", d, re.I):
        m = re.search(r"\bpoint\s+(\d+(?:\.\d+)*)\b", d, re.I)
        if m:
            loc.point = m.group(1)
    return loc


def _subject_multi(directive: str) -> tuple[Optional[str], List[str]]:
    """A multi-target subject ('paragraphs 2 and 3', 'points 7 and 9') → the
    kind + the list of labels; ``(None, [])`` when the subject is singular."""
    d = _norm(directive)
    m = re.search(r"\bparagraphs\s+(\d+[a-z]?)\s+and\s+(\d+[a-z]?)\b", d, re.I)
    if m:
        return "paragraph", [m.group(1), m.group(2)]
    m = re.search(r"\bpoints\s+(\d+)\s+and\s+(\d+)\b", d, re.I)
    if m:
        return "point", [m.group(1), m.group(2)]
    m = re.search(r"\bpoints\s+((?:\([^)]+\)[,\s]*(?:and\s+)?)+)", d, re.I)
    if m:
        labels = re.findall(r"\(([^)]+)\)", m.group(1))
        if len(labels) > 1:
            return "point", labels
    return None, []


# ── content payload parsing ──────────────────────────────────────────────────

_CHILD_KIND = {
    "article": "paragraph",
    "paragraph": "point",
    "subparagraph": "point",
    "point": "roman_item",
    "roman_item": "indent",
    "indent": "indent",
    "annex": "section",
    "section": "point",
}


def _child_kind(kind: str) -> str:
    return _CHILD_KIND.get(kind, "indent")


def _split_enumerator(text: str) -> tuple[str, str]:
    """Peel a leading enumerator off a chapeau/body string.

    Handles paragraph numbers ("1a. …", "2. …"), parenthesised points/romans
    ("(ba) …", "(i) …"), and inserted-article headers ("Article 4a …").  Returns
    ``(enumerator, remaining_text)``; enumerator is "" when none is present.
    """
    s = text.lstrip()
    while s and s[0] in _QUOTE_OPENERS:
        s = s[1:].lstrip()
    m = re.match(r"^\((\w+)\)\s*", s)                    # (ba) (a) (i)
    if m:
        return m.group(1), s[m.end():].strip()
    m = re.match(r"^(\d+[a-z]?)\.\s+", s)                # 1a.  2.
    if m:
        return m.group(1), s[m.end():].strip()
    m = re.fullmatch(r"(\d+[a-z]?)\.", s)                # standalone "21." (annex enum cell); not "6.1"
    if m:
        return m.group(1), ""
    m = re.match(r"^Article\s+(\d+[a-z]?)\b\.?\s*", s, re.I)  # inserted article header
    if m:
        return f"Article {m.group(1)}", s[m.end():].strip()
    m = re.match(r"^Annex\s+([IVXLCDM]+|\d+)\b\.?\s*", s, re.I)  # added annex header
    if m:
        return f"Annex {m.group(1)}", s[m.end():].strip()
    return "", s.strip()


def _parse_point_table(table, kind: str) -> List[ContentNode]:
    """Parse a two-column point ``<table>`` into ContentNodes, recursively.

    Each ``<tr>`` is ``[enumerator-cell, content-cell]``; the content cell's own
    ``<p>`` text is the body, and its nested tables recurse as children.
    """
    out: List[ContentNode] = []
    for tr in _rows(table):
        tds = _cells(tr)
        if len(tds) < 2:
            # single-cell row — descend into any inner tables at this level
            for sub in tds[0].find_all("table", recursive=False) if tds else []:
                out.extend(_parse_point_table(sub, kind))
            continue
        enum, _ = _split_enumerator(_norm(tds[0].get_text()))
        body = _strip_quotes(_direct_paras(tds[1]))
        node = ContentNode(enum, kind, body, [])
        node.children.extend(_parse_payload(_element_children(tds[1]), _child_kind(kind),
                                            skip_first_p=True))
        if not _is_blank_unit(enum, body, node.children):
            out.append(node)
    return out


def _parse_payload(elements: List, kind: str, skip_first_p: bool = False) -> List[ContentNode]:
    """Parse an ordered list of payload elements into ContentNodes.

    ``elements`` are the direct children of a content cell (or ``<div>``) after
    the directive.  ``kind`` is the kind for units created at THIS level; nested
    point tables step down via :func:`_child_kind`.  ``skip_first_p`` drops the
    directive/body ``<p>`` already consumed by the caller.
    """
    nodes: List[ContentNode] = []
    pending: Optional[ContentNode] = None    # a paragraph awaiting its child tables
    seen_p = False
    for el in elements:
        name = el.name
        if name == "p":
            if skip_first_p and not seen_p:
                seen_p = True
                continue
            seen_p = True
            enum, body = _split_enumerator(_norm(el.get_text()))
            body = _strip_quotes(body)
            if not enum and not body:
                continue
            node = ContentNode(enum, kind, body, [])
            nodes.append(node)
            pending = node
        elif name == "table":
            pts = _parse_point_table(el, _child_kind(kind) if pending else kind)
            if pending is not None:
                pending.children.extend(pts)
            else:
                nodes.extend(pts)
        elif name == "div":
            # a div-wrapped provision group: <p>enum chapeau</p> + sibling tables
            sub = _parse_payload(_element_children(el), kind)
            nodes.extend(sub)
            pending = sub[-1] if sub else pending
    return nodes


# A standalone article header line ("‘Article 4a"), NOT a directive ("Article 4
# is replaced …") nor a cross-reference ("Article 6(2)").
_ARTICLE_HEADER = re.compile(r"^[‘'\"\s]*Article\s+(\d+[a-z]?)\s*$", re.I)
_PARA_ENUM = re.compile(r"^[‘'\"\s]*\d+[a-z]?\.\s")


def _article_stream(elements: List) -> List[tuple]:
    """Flatten a whole-article payload into ordered ``(role, element)`` items —
    ``role`` ∈ {'header','p','table','para'}.  Unwraps the quoted-block wrapper
    ``<div>`` but keeps a paragraph ``<div>`` (its first ``<p>`` is enumerated
    "N.") intact as one unit, so an article's paragraphs survive as units."""
    out: List[tuple] = []
    for el in elements:
        name = el.name
        if name == "p":
            out.append(("header" if _ARTICLE_HEADER.match(_norm(el.get_text())) else "p", el))
        elif name == "table":
            out.append(("table", el))
        elif name == "div":
            first_p = el.find("p", recursive=False)
            if first_p and _PARA_ENUM.match(_norm(first_p.get_text())):
                out.append(("para", el))                    # a paragraph unit
            else:
                out.extend(_article_stream(_element_children(el)))   # unwrap
    return out


def _parse_article_payload(elements: List) -> Optional[List[ContentNode]]:
    """Parse a whole-article (or multi-article) quoted payload into article
    ContentNodes with nested paragraphs/points.  The payload is a mini-document:
    ``<p>Article N</p><p>Title</p>`` then paragraph units (each with its points).
    ``None`` when no article header is present (caller falls back)."""
    stream = _article_stream(elements)
    header_idx = [i for i, (role, _) in enumerate(stream) if role == "header"]
    if not header_idx:
        return None
    articles: List[ContentNode] = []
    for k, start in enumerate(header_idx):
        end = header_idx[k + 1] if k + 1 < len(header_idx) else len(stream)
        number = _ARTICLE_HEADER.match(_norm(stream[start][1].get_text())).group(1)
        rest = stream[start + 1:end]
        title = ""
        if rest and rest[0][0] == "p":
            t_enum, t_text = _split_enumerator(_norm(rest[0][1].get_text()))
            if not t_enum:                                  # the untitled heading line
                title = _strip_quotes(t_text)
                rest = rest[1:]
        paragraphs = _parse_payload([el for _, el in rest], "paragraph")
        articles.append(ContentNode(f"Article {number}", "article", title, paragraphs))
    return articles


def _payload_elements(content_td) -> List:
    """Direct children of a content cell that make up the quoted payload
    (everything after the leading directive ``<p>``)."""
    return _element_children(content_td)


def _subpart_cells(content_td) -> List:
    """The content cells of an amend-as-follows container's sub-parts (a),(b),…"""
    cells = []
    for tbl in content_td.find_all("table", recursive=False):
        for tr in _rows(tbl):
            tds = _cells(tr)
            if len(tds) >= 2:
                cells.append(tds[1])
    return cells


def _first_para_text(content_td) -> str:
    p = content_td.find("p", recursive=False)
    return _norm(p.get_text(" ", strip=True)) if p else ""


# ── recursive directive walker ───────────────────────────────────────────────

def _item_kind(op: str, directive: str, own: Locator) -> str:
    """The kind of thing an op operates on: the new item for insert/add, the
    subject leaf for replace/delete."""
    if op in ("insert", "add"):
        return _new_item_kind(directive)
    return own.leaf_kind()


def _target_from_content(content: List[ContentNode]) -> str:
    """Locator for a NEW article/annex insert, taken from its enacted header
    (content opens with 'Article 4a …' / 'Annex XIV …')."""
    for c in content:
        if c.enumerator.startswith(("Article ", "Annex ")):
            return c.enumerator
    return ""


def _is_directive_cell(cell) -> bool:
    """True when a sub-cell's lead prose is itself an amendment directive
    (a known verb), as opposed to enacted quoted content."""
    return _classify_verb(_first_para_text(cell)) != "unknown"


def _parse_unit(content_td, scope: Locator) -> List[Operation]:
    """Parse one directive unit (a content cell) into concrete Operations.

    amend-as-follows recurses into its sub-parts under the accumulated scope;
    every other verb yields Operation(s) — replace/delete may fan out over a
    multi-target subject ('paragraphs 2 and 3', 'points 7 and 9').
    """
    directive = _first_para_text(content_td)
    op = _classify_verb(directive)
    own = _parse_locator(directive)
    here = own.merged_under(scope)

    if op == "amend":
        subcells = _subpart_cells(content_td)
        # Normally "X is amended as follows:" introduces enumerated sub-directives
        # (a),(b),…  But sometimes it is followed directly by the whole quoted
        # replacement of X ("point (14) is amended as follows: '(14) …'") with no
        # sub-directives — that is semantically a replace, so fall through.
        if subcells and all(_is_directive_cell(c) for c in subcells):
            ops: List[Operation] = []
            for sub_td in subcells:
                ops.extend(_parse_unit(sub_td, here))
            return ops
        op = "replace"

    if op == "unknown":
        return [Operation("unknown", "provision", here.render(), [], directive, here.to_fields())]

    kind = _item_kind(op, directive, own)
    if op == "delete":
        content = []
    elif kind == "article":
        # A whole-article replace/insert carries a mini-article document (header,
        # title, paragraphs) that must nest — not flatten like paragraph content.
        content = _parse_article_payload(_payload_elements(content_td)) or []
    else:
        content = _parse_payload(_payload_elements(content_td), kind, skip_first_p=True)

    multi_kind, labels = _subject_multi(directive)
    if multi_kind and op in ("replace", "delete"):
        ops = []
        for label in labels:
            tgt = here.with_leaf(multi_kind, label)
            unit = [c for c in content if c.enumerator == label]
            ops.append(Operation(op, multi_kind, tgt.render(), unit, directive, tgt.to_fields()))
        return ops

    target = here.render()
    if op in ("insert", "add") and kind in ("article", "annex") and not target:
        target = _target_from_content(content)
    return [Operation(op, kind, target, content, directive, here.to_fields())]


# ── entry points ─────────────────────────────────────────────────────────────

from pathlib import Path

from domain.legislation_catalog import LEGISLATION

# {"2024/1689": "32024R1689", …} — the acts we have loaded, keyed by OJ number.
CELEX_BY_NUMBER: dict = {meta["number"]: celex for celex, meta in LEGISLATION.items()}
_DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "legislation"

_AMENDS_TITLE_RE = re.compile(r"Amendments?\s+to\s+Regulation\s+\(EU\)\s*(\d{4}/\d+)", re.I)


def _article_title(soup, article_id: str) -> str:
    t = soup.find("div", id=f"{article_id}.tit_1")
    return _norm(t.get_text()) if t else ""


def _parse_amending_article(art_div, target_celex: str) -> List[AmendmentInstruction]:
    """Parse one amending Article's numbered points into instructions."""
    instructions: List[AmendmentInstruction] = []
    for tbl in art_div.find_all("table", recursive=False):
        for tr in _rows(tbl):
            tds = _cells(tr)
            if len(tds) < 2:
                continue
            m = re.fullmatch(r"\((\d{1,3})\)", _norm(tds[0].get_text()))
            if not m:
                continue
            ops = _parse_unit(tds[1], scope=Locator())
            instructions.append(AmendmentInstruction(
                point_num=m.group(1),
                target_celex=target_celex,
                operations=ops,
                raw_lead=_first_para_text(tds[1]),
            ))
    return instructions


def parse_amendments_html(html: str, target_celex: str,
                          amending_article_id: str = "art_1") -> List[AmendmentInstruction]:
    """Parse a single amending Article of an act's raw HTML (explicit target)."""
    soup = BeautifulSoup(html, "lxml")
    art = soup.find("div", id=amending_article_id)
    return _parse_amending_article(art, target_celex) if art else []


def find_amending_articles(soup) -> List[tuple]:
    """Every enacting Article titled "Amendments to Regulation (EU) N" whose N is
    a loaded act → ``(article_id, target_celex)``."""
    out: List[tuple] = []
    enc = soup.find("div", id="enc_1")
    for art in (enc.find_all("div", id=re.compile(r"^art_\d+[a-z]?$")) if enc else []):
        m = _AMENDS_TITLE_RE.search(_article_title(soup, art["id"]))
        if m and CELEX_BY_NUMBER.get(m.group(1)):
            out.append((art["id"], CELEX_BY_NUMBER[m.group(1)]))
    return out


def parse_amending_regulation(celex: str, lang: str = "EN",
                              data_root: Optional[Path] = None) -> List[AmendmentInstruction]:
    """Parse every amending Article of a loaded amending act (by CELEX)."""
    root = Path(data_root) if data_root else _DEFAULT_DATA_ROOT
    html = (root / celex / lang / "raw" / "raw.html").read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    instructions: List[AmendmentInstruction] = []
    for art_id, target_celex in find_amending_articles(soup):
        art = soup.find("div", id=art_id)
        instructions.extend(_parse_amending_article(art, target_celex))
    return instructions


def instructions_to_dict(instructions: List[AmendmentInstruction]) -> dict:
    return {"instructions": [i.to_dict() for i in instructions]}


if __name__ == "__main__":                                   # pragma: no cover
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description="Parse an amending act into structured instructions.")
    ap.add_argument("celex", help="CELEX of the amending act, e.g. 32026R1744")
    ap.add_argument("--lang", default="EN")
    ap.add_argument("--out", help="write amendments.json here (default: stdout)")
    args = ap.parse_args()

    ins = parse_amending_regulation(args.celex, args.lang)
    payload = instructions_to_dict(ins)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    n_ops = sum(len(i.operations) for i in ins)
    print(f"[amendment_parser] {args.celex}: {len(ins)} points, {n_ops} operations",
          file=sys.stderr)
