"""Stage 2 — apply parsed amendment instructions onto the base regulation.

Consumes the structured :class:`~consolidation.amendment_parser.AmendmentInstruction`s
and mutates a COPY of the base act's ``parsed.json`` provisions so it holds
consolidated (current) law: inserted paragraphs/points become real nodes in the
right position, replaced provisions get the new text + subtree, deleted ones are
removed.  The result flows through the *existing* load → embed → canonicalize
pipeline unchanged — this module never reimplements id assignment, ref building,
or embedding; it only produces well-formed base-format nodes.

Pure and reversible: the source ``parsed.json`` is untouched (the caller writes
the consolidated tree to a separate file), and every node this module creates or
rewrites is tagged ``amended_by`` / ``amendment_op`` so the change is auditable
and removable.  Targets are resolved by *traversing* the base tree (robust to the
id-format differences between numbered-paragraph and subparagraph articles), not
by computing ids.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import List, Optional

from consolidation.amendment_parser import AmendmentInstruction, ContentNode, Operation

# 3 non-breaking spaces — EUR-Lex's enumerator gap in a paragraph's body text.
_ENUM_GAP = "   "
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
             "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}


@dataclass
class OpResult:
    """What became of one operation — an auditable trail over the mutation."""
    point_num: str
    op: str
    target_ref: str
    status: str                       # "applied" | "skipped"
    detail: str = ""
    node_ids: List[str] = field(default_factory=list)


# ── indexing / small helpers ─────────────────────────────────────────────────

def _index(provisions: List[dict]) -> dict:
    return {n["id"]: n for n in provisions}


def _children(byid: dict, node: dict) -> List[dict]:
    return [byid[c] for c in node.get("children", []) if c in byid]


def _pad3(num: str) -> str:
    """"6" → "006", "113" → "113", "4a" → "004a" (numeric part zero-padded)."""
    m = re.match(r"(\d+)([a-z]?)", num)
    return (m.group(1).zfill(3) + m.group(2)) if m else num


def _art_num(text: str) -> str:
    """"Article 4a" / "4a" → "4a"."""
    m = re.search(r"(\d+[a-z]?)", text)
    return m.group(1) if m else text


def _anchor_number(num: str) -> str:
    """The existing sibling an inserted label extends: "1a"→"1", "ba"→"b"."""
    return num[:-1] if (num and num[-1].isalpha() and num[:-1]) else num


# ── target resolution (traversal, not id computation) ────────────────────────

def _find_article(byid: dict, celex: str, number: str) -> Optional[dict]:
    return byid.get(f"{celex}_art_{number}")


def _find_annex(byid: dict, celex: str, number: str) -> Optional[dict]:
    for suffix in (number, number.upper(), number.lower()):
        n = byid.get(f"{celex}_anx_{suffix}")
        if n:
            return n
    return None


def _child_by_number(byid: dict, node: dict, kinds: tuple, number: str) -> Optional[dict]:
    for c in _children(byid, node):
        if c.get("kind") in kinds and (c.get("number") or "") == number:
            return c
    return None


def _nth_of_kind(byid: dict, node: dict, kinds: tuple, n: int) -> Optional[dict]:
    matches = [c for c in _children(byid, node) if c.get("kind") in kinds]
    return matches[n - 1] if 0 < n <= len(matches) else None


def _paragraph(byid: dict, article: dict, number: str) -> Optional[dict]:
    """A numbered paragraph — stored either as a :paragraph or (for articles
    without numbered paragraphs) matched loosely by number."""
    return (_child_by_number(byid, article, ("paragraph",), number)
            or _child_by_number(byid, article, ("paragraph", "subparagraph"), number))


def _nth_paragraph(byid: dict, article: dict, n: int) -> Optional[dict]:
    """"the third paragraph" — the nth paragraph OR subparagraph child (Article
    113's unnumbered paragraphs are stored as subparagraphs)."""
    return _nth_of_kind(byid, article, ("paragraph", "subparagraph"), n)


def _nth_subparagraph(byid: dict, node: dict, ordinal: str) -> Optional[dict]:
    """"the first subparagraph" of a paragraph.  A single-subparagraph paragraph
    keeps its text on the paragraph itself, so "first subparagraph" == node."""
    n = _ORDINALS.get(ordinal, 1) if not ordinal.isdigit() else int(ordinal)
    subs = [c for c in _children(byid, node) if c.get("kind") == "subparagraph"]
    if subs:
        return subs[n - 1] if 0 < n <= len(subs) else None
    return node if n == 1 else None


_POINT_KINDS = ("point", "indent", "annex_point")


def _point(byid: dict, node: dict, label: str) -> Optional[dict]:
    """A point by label, at this level or (fallback) anywhere beneath it."""
    direct = _child_by_number(byid, node, _POINT_KINDS, label)
    if direct:
        return direct
    stack = list(_children(byid, node))
    while stack:
        c = stack.pop()
        if c.get("kind") in _POINT_KINDS and (c.get("number") or "") == label:
            return c
        stack.extend(_children(byid, c))
    return None


def resolve(byid: dict, celex: str, tf: dict) -> Optional[dict]:
    """Resolve a structured target locator to its base node (for replace/delete)
    or its container (for insert/add — the leaf item isn't in ``tf``)."""
    if tf.get("article"):
        node = _find_article(byid, celex, tf["article"])
    elif tf.get("annex"):
        node = _find_annex(byid, celex, tf["annex"])
    else:
        return None
    if node is None:
        return None
    if tf.get("para"):
        node = _paragraph(byid, node, tf["para"])
    if node and tf.get("para_ord"):
        node = _nth_paragraph(byid, node, _ORDINALS.get(tf["para_ord"], 0))
    if node and tf.get("section"):
        node = _child_by_number(byid, node, ("section", "annex_section"), tf["section"])
    if node and tf.get("subpara"):
        node = _nth_subparagraph(byid, node, tf["subpara"])
    if node and tf.get("intro"):
        subs = [c for c in _children(byid, node) if c.get("kind") == "subparagraph"]
        node = subs[0] if subs else node
    if node and tf.get("point"):
        node = _point(byid, node, tf["point"])
    return node


# ── node building (ContentNode → base-format node dicts) ─────────────────────

# An annex's descendants use their own kinds/ids, not the article scheme.
_ANNEX_CHILD_KIND = {"annexes": "annex", "annex": "annex_section",
                     "annex_section": "annex_point", "annex_point": "annex_point"}


def _effective_kind(parent_kind: str, cn_kind: str) -> str:
    """Under an annex, a ContentNode's generic kind maps to the annex family."""
    return _ANNEX_CHILD_KIND.get(parent_kind, cn_kind)


def _derive_id(celex: str, parent: dict, kind: str, number: str) -> str:
    pid = parent["id"]
    if kind == "article":
        return f"{celex}_art_{_art_num(number)}"
    if kind == "paragraph":
        return f"{celex}_{_pad3(parent.get('number', ''))}.{_pad3(number)}"
    if kind == "annex":
        return f"{celex}_anx_{number}"
    if kind == "annex_section":
        return f"{pid}_sec_{number}"
    if kind == "annex_point":
        return f"{pid}_{number}"
    suffix = {"subparagraph": "sp", "point": "pt", "roman_item": "rm", "indent": "ind"}
    return f"{pid}_{suffix.get(kind, 'x')}_{number}"


def _node_text(cn: ContentNode) -> str:
    """Base convention: paragraphs/subparagraphs carry the enumerator in the body
    ("1a.   …"); points/romans/articles do not."""
    if cn.kind in ("paragraph", "subparagraph") and cn.text:
        return f"{cn.enumerator}.{_ENUM_GAP}{cn.text}"
    return cn.text


def _build_subtree(cn: ContentNode, parent: dict, celex: str,
                   amending_celex: str) -> tuple[dict, List[dict]]:
    """Build a base-format node (+ all descendants) from a ContentNode."""
    kind = _effective_kind(parent.get("kind", ""), cn.kind)
    node = {
        "id": _derive_id(celex, parent, kind, cn.enumerator),
        "kind": kind,
        "text": _node_text(cn),
        "hierarchy_depth": parent.get("hierarchy_depth", 0) + 1,
        "path": list(parent.get("path", [])) + [parent["id"]],
        "parent_id": parent["id"],
        "children": [],
        "lang": parent.get("lang", "EN"),
        "number": cn.enumerator if kind != "article" else _art_num(cn.enumerator),
        "binding_force": parent.get("binding_force", "binding"),
        "source_type": parent.get("source_type", "regulation"),
        "text_for_analysis": cn.text,
        "amended_by": amending_celex,
        "amendment_op": "new",
    }
    if kind == "article":
        node["title"] = cn.text
    all_nodes = [node]
    for child in cn.children:
        child_node, descendants = _build_subtree(child, node, celex, amending_celex)
        node["children"].append(child_node["id"])
        all_nodes.extend(descendants)
    return node, all_nodes


def _build_forest(content: List[ContentNode], parent: dict, celex: str,
                  amending_celex: str) -> tuple[List[dict], List[dict]]:
    """Build the top-level new nodes (to splice into ``parent``) and the full
    flat list of every node created (tops + descendants)."""
    tops, everything = [], []
    for cn in content:
        top, descendants = _build_subtree(cn, parent, celex, amending_celex)
        tops.append(top)
        everything.extend(descendants)
    return tops, everything


# ── tree editors ─────────────────────────────────────────────────────────────

def _remove_subtree(provisions: List[dict], byid: dict, node: dict) -> None:
    ids = set()
    stack = [node]
    while stack:
        n = stack.pop()
        ids.add(n["id"])
        stack.extend(_children(byid, n))
    provisions[:] = [p for p in provisions if p["id"] not in ids]
    for i in ids:
        byid.pop(i, None)


def _detach_from_parent(byid: dict, node: dict) -> None:
    parent = byid.get(node.get("parent_id"))
    if parent and node["id"] in parent.get("children", []):
        parent["children"].remove(node["id"])


def _splice(parent: dict, new_ids: List[str], anchor_id: Optional[str]) -> None:
    """Insert ``new_ids`` into ``parent.children`` after ``anchor_id`` (or append)."""
    kids = parent.setdefault("children", [])
    if anchor_id and anchor_id in kids:
        at = kids.index(anchor_id) + 1
    else:
        at = len(kids)
    parent["children"][at:at] = new_ids


def _apply_insert(provisions, byid, parent, op, base_celex, amending_celex, mode):
    tops, everything = _build_forest(op.content, parent, base_celex, amending_celex)
    if not tops:
        return "skipped", "no content parsed", []
    anchor_id = None
    if mode == "insert":
        anchor_base = _anchor_number(tops[0]["number"])
        anchor = _child_by_number(byid, parent, (tops[0]["kind"],), anchor_base)
        anchor_id = anchor["id"] if anchor else None
    _splice(parent, [t["id"] for t in tops], anchor_id)
    provisions.extend(everything)
    for n in everything:
        byid[n["id"]] = n
    return "applied", f"{mode} into {parent['id']}", [t["id"] for t in tops]


def _apply_replace(provisions, byid, node, op, base_celex, amending_celex):
    # "the heading is replaced by the following:" swaps only the title, not the
    # article body — do NOT wipe the paragraphs.
    if op.item_kind == "heading" or (op.target or {}).get("heading"):
        new_heading = op.content[0].text if op.content else node.get("text", "")
        node["text"] = new_heading
        if "title" in node or node.get("kind") == "article":
            node["title"] = new_heading
        node["amended_by"] = amending_celex
        node["amendment_op"] = "replace-heading"
        return "applied", f"replaced heading of {node['id']}", [node["id"]]
    for c in list(_children(byid, node)):
        _remove_subtree(provisions, byid, c)
    node["children"] = []
    unit = op.content[0] if op.content else None
    if unit is not None:
        node["text"] = _node_text(unit)
        node["text_for_analysis"] = unit.text
        for child in unit.children:
            child_node, descendants = _build_subtree(child, node, base_celex, amending_celex)
            node["children"].append(child_node["id"])
            provisions.extend(descendants)
            for n in descendants:
                byid[n["id"]] = n
    node["amended_by"] = amending_celex
    node["amendment_op"] = "replace"
    return "applied", f"replaced {node['id']}", [node["id"]]


def _apply_insert_article(provisions, byid, op, base_celex, amending_celex):
    """A new article ("the following Article is inserted: 'Article 4a …'") — its
    parent is the chapter of the article it follows (anchor 4a → after 4)."""
    if not op.content:
        return "skipped", "no article content", []
    art_cn = op.content[0]
    new_num = _art_num(art_cn.enumerator)
    anchor = _find_article(byid, base_celex, _anchor_number(new_num))
    if anchor is None:
        return "skipped", f"anchor article {_anchor_number(new_num)} not found", []
    parent = byid.get(anchor.get("parent_id"))
    if parent is None:
        return "skipped", "anchor has no parent chapter", []
    top, everything = _build_subtree(art_cn, parent, base_celex, amending_celex)
    _splice(parent, [top["id"]], anchor["id"])
    provisions.extend(everything)
    for n in everything:
        byid[n["id"]] = n
    return "applied", f"inserted article {top['id']} after {anchor['id']}", [top["id"]]


def _apply_op(provisions, byid, op: Operation, celex: str, amending_celex: str) -> tuple:
    tf = op.target or {}
    if op.op == "delete":
        node = resolve(byid, celex, tf)
        if node is None:
            return "skipped", "delete target not found", []
        _detach_from_parent(byid, node)
        _remove_subtree(provisions, byid, node)
        return "applied", f"deleted {node['id']}", [node["id"]]

    if op.op == "replace":
        node = resolve(byid, celex, tf)
        if node is None:
            return "skipped", "replace target not found", []
        return _apply_replace(provisions, byid, node, op, celex, amending_celex)

    if op.op in ("insert", "add"):
        if op.item_kind == "article":
            return _apply_insert_article(provisions, byid, op, celex, amending_celex)
        if op.item_kind == "annex":
            # A whole new annex (e.g. Annex XIV) is free-form content the article
            # grammar does not parse faithfully — defer rather than inject garbled
            # law into the corpus (same fail-safe as the verbatim display).
            ref = op.content[0].enumerator if op.content else op.target_ref
            return "deferred", f"free-form annex not consolidated: {ref}", []
        parent = resolve(byid, celex, tf)
        if parent is None:
            return "skipped", "insert container not found", []
        return _apply_insert(provisions, byid, parent, op, celex, amending_celex, op.op)

    return "skipped", f"unhandled op {op.op}", []


# ── entry point ──────────────────────────────────────────────────────────────

def consolidate(base_provisions: List[dict], instructions: List[AmendmentInstruction],
                amending_celex: str) -> tuple[List[dict], List[OpResult]]:
    """Apply all instructions to a COPY of the base provisions.

    Returns ``(consolidated_provisions, report)``; the report records what
    happened to every operation (applied/skipped + reason)."""
    provisions = copy.deepcopy(base_provisions)
    byid = _index(provisions)
    report: List[OpResult] = []
    for instr in instructions:
        celex = instr.target_celex
        for op in instr.operations:
            status, detail, ids = _apply_op(provisions, byid, op, celex, amending_celex)
            report.append(OpResult(instr.point_num, op.op, op.target_ref, status, detail, ids))
    return provisions, report
