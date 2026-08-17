"""Verbatim provision display — an authoritative, LLM-free render of a directly
requested provision, with in-force amendments spliced in.

Why this exists: a "show me Article X" request must return the *actual* text from
the corpus, not an LLM reconstruction. When such a lookup was routed through the
generative path, the model paraphrased and *fabricated* base paragraphs (e.g. it
invented AI Act Article 6(3)–(7)) while presenting them as controlling law. This
path bypasses generation entirely: it renders the ordered ``HAS_PART`` subtree
verbatim (``retrieve_by_refs`` already attaches it) and splices the controlling
amendments — inserted paragraphs, replacements, deletions — into place, each
tagged with its amending act. Deterministic; no LLM; nothing to fabricate.

Two public entry points, both re-exported from ``application.agent``:

* :func:`wants_verbatim_display` — conservative intent gate on the *original*
  question (so a display request is honoured even mid-conversation, unaffected by
  the history rewrite that otherwise reframes it).
* :func:`render_provision_display` — the render itself, or ``None`` when the
  provision can't be resolved (the caller then falls back to generation).
"""
from __future__ import annotations

import re

from application._config import (
    _REG_NAME_TO_CELEX,
    _detect_mentioned_regulations,
    _extract_provision_refs,
)
from application._postprocessing import _act_display, _build_amendment_provenance
from domain.legislation_catalog import LEGISLATION

# ── intent detection ────────────────────────────────────────────────────────
# Fire ONLY for a pure display request naming exactly one provision in exactly
# one regulation. Anything analytical (how does it apply, obligations, is it
# high-risk, a scenario) must go to the generative path.
_DISPLAY_VERB = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+)?"
    r"(?:show(?:\s+me)?|display|give\s+me|quote|print|reproduce|read(?:\s+me)?|"
    r"pull\s+up|render|let'?s\s+see|what\s+does|what\s+is|what'?s)\b",
    re.I,
)
_ANALYTICAL = re.compile(
    r"\b(?:how|why|appl(?:y|ies|ication)|oblig\w*|comply|complian\w*|classif\w*|"
    r"high[-\s]?risk|liab\w*|responsib\w*|assess\w*|scenario|integrat\w*|require\w*|"
    r"if\s+a\b|for\s+a\b|my\b|does\s+it\b|would\s+it\b)",
    re.I,
)
_MAX_DISPLAY_WORDS = 16


def wants_verbatim_display(question: str) -> tuple[str, str] | None:
    """Return ``(ref, celex)`` to render verbatim, or ``None``.

    Runs on the ORIGINAL user question (not the history-rewritten one), so a
    literal "show me Article X" is honoured even when asked after other turns.
    Deliberately narrow: a display verb, no analytical language, a short
    question, and exactly one provision reference in exactly one regulation.
    """
    if not question:
        return None
    q = question.strip()
    if not _DISPLAY_VERB.search(q) or _ANALYTICAL.search(q):
        return None
    if len(q.split()) > _MAX_DISPLAY_WORDS:
        return None
    refs = _extract_provision_refs(q)
    if len(refs) != 1:
        return None
    regs = _detect_mentioned_regulations(q)
    if len(regs) != 1:
        return None
    # `_detect_mentioned_regulations` yields the human name ("EU AI Act"); the
    # render needs the CELEX the graph is keyed on.
    name = next(iter(regs))
    celex = _REG_NAME_TO_CELEX.get(name, name if name in LEGISLATION else None)
    if not celex:
        return None
    return refs[0], celex


# ── amendment-instruction parsing (EU grammar is regular) ────────────────────

_QUOTE = "‘’‛`\"'"


def _strip_heading(article_text: str) -> str:
    """Drop the 'Article 1 — Amendments to Regulation … |' container prefix."""
    return re.sub(r"\s+", " ", (article_text or "").rsplit("|", 1)[-1]).strip()


def _extract_quoted_block(instr: str) -> str:
    """The replacement/insertion text — inside the first ‘ … ’ (curly preferred,
    straight only as a fallback so an apostrophe in the body never mis-triggers)."""
    m = re.search(r"‘(.+?)’(?:\s*;?\s*)*$", instr, re.S) or re.search(r"‘(.+)’", instr, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"'(.+)'", instr, re.S)
    return m.group(1).strip() if m else ""


def _operation_of(instr: str) -> str:
    low = instr.lower()
    if re.search(r"\b(?:is|are)\s+inserted\b", low) or "following paragraph" in low and "inserted" in low:
        return "inserted"
    if re.search(r"\b(?:is|are)\s+added\b", low):
        return "added"
    if re.search(r"\bis\s+deleted\b", low) or re.search(r"\bare\s+deleted\b", low):
        return "deleted"
    if re.search(r"\bis\s+replaced\b", low):
        return "replaced"
    if re.search(r"\bis\s+amended\b", low):
        return "amended"
    return "amended"


def _subtarget_number(instr: str) -> str | None:
    """The paragraph/point the instruction acts on: 'paragraph 3 is replaced' → '3',
    'point (14) is amended' → '14'. None when it targets the article as a whole."""
    m = re.search(r"\bparagraph\s+(\d+[a-z]?)\b", instr, re.I) or \
        re.search(r"\bpoint\s+\(?(\d+[a-z]?|[a-z])\)?", instr, re.I)
    return m.group(1) if m else None


def _parse_inserted_paragraphs(block: str, base_ref: str, act: str) -> list[dict]:
    """Split an inserted quoted block ('1a. … 1b. … 1c. …') into synthetic
    paragraph nodes that render like the base paragraphs, each tagged as inserted."""
    block = re.sub(r"\s+", " ", block.replace("\xa0", " ")).strip()
    # Split before each 'Nx.' enumerator (a digit-group extended by a letter, the
    # canonical shape of an inserted paragraph — 1a, 1b, 1c).
    parts = re.split(r"\s+(?=\d+[a-z]+\.\s)", block)
    nodes: list[dict] = []
    for part in parts:
        m = re.match(r"(\d+[a-z]+)\.\s", part)
        if not m:
            continue
        enum = m.group(1)
        nodes.append({
            "depth": 1,
            "kind": "paragraph",
            "number": enum,
            "ref": f"{base_ref}({enum})",
            "text": part.strip(),
            "_inserted_by": act,
        })
    return nodes


def _anchor_digits(enum: str) -> str:
    m = re.match(r"\d+", enum)
    return m.group(0) if m else ""


def _is_flat_insertion(instr: str) -> bool:
    """True only for a single, top-level "the following paragraph(s) is/are
    inserted" whose content is flat paragraphs (Article 6's 1a/1b/1c). A
    multi-part amendment ("… is amended as follows: (a) … (b) …") or one inserting
    letter-points into a subparagraph (Article 5) is beyond the flat splice — the
    upstream parser flattens its nested wording, so it must NOT be reproduced
    inline as if verbatim. FAIL SAFE: only splice what we can render faithfully."""
    low = instr.lower()
    if "amended as follows" in low:
        return False                                   # multi-part amendment
    if re.search(r"points?\s+(?:is|are)\s+inserted", low):
        return False                                   # inserts letter-points, not paragraphs
    return bool(re.search(r"paragraphs?\s+(?:is|are)\s+inserted", low))


def _splice(subtree: list[dict], amendments: list[dict], base_ref: str) -> tuple[list[dict], list[str], list[str]]:
    """Weave the in-force amendments into a copy of the ordered subtree.

    Returns (nodes, act_labels, notes). Simple flat insertions graft new paragraph
    nodes at their position; replacements/deletions/amendments tag the affected
    base node so its (still pre-amendment) text is never shown as unqualified
    current law. A COMPLEX insertion the flat splice can't render faithfully
    (e.g. Article 5's nested points) is NOT reproduced inline — it becomes a
    *note* — so garbled amendment text is never presented as verbatim law. The
    exact wording lives in the AMENDMENTS APPLIED footer + the amending act until
    graph consolidation lands.
    """
    nodes = [dict(n) for n in subtree]
    act_labels: list[str] = []
    notes: list[str] = []
    for a in amendments:
        act = a.get("amending_act") or "a later act"
        label = _act_display(act)
        if label not in act_labels:
            act_labels.append(label)
        instr = _strip_heading(a.get("article_text") or a.get("text") or "")
        if not instr:
            continue
        op = _operation_of(instr)
        if op in ("inserted", "added"):
            if _is_flat_insertion(instr):
                block = _extract_quoted_block(instr)
                new_nodes = _parse_inserted_paragraphs(block, base_ref, act)
                if new_nodes:
                    nodes = _insert_after_paragraph(nodes, _anchor_digits(new_nodes[0]["number"]), new_nodes)
                    continue
            # complex insertion — do NOT inline garbled/flattened text
            notes.append(
                f"⚠ **{act}** inserts further provisions into **{base_ref}** whose nested "
                f"wording is not reproduced inline above; see the AMENDMENTS APPLIED note "
                f"below and the amending act for the exact text."
            )
            continue
        # replaced / amended / deleted:
        sub = _subtarget_number(instr)
        tag = {"deleted": "repealed", "replaced": "replaced", "amended": "amended"}.get(op, "amended")
        _tag_paragraph(nodes, sub, f"⚠ [paragraph {sub} {tag} by {act} — see amendments below]" if sub
                       else f"⚠ [amended by {act} — see amendments below]")
    return nodes, act_labels, notes


def _insert_after_paragraph(nodes: list[dict], anchor_num: str, new_nodes: list[dict]) -> list[dict]:
    """Insert *new_nodes* immediately after the depth-1 paragraph numbered
    *anchor_num* and all of its descendants (so 1a-1c land between 1 and 2)."""
    if not anchor_num:
        return nodes + new_nodes
    out: list[dict] = []
    i, n, done = 0, len(nodes), False
    while i < n:
        node = nodes[i]
        out.append(node)
        if not done and (node.get("depth") == 1) and (node.get("number") or "").strip() == anchor_num:
            i += 1
            while i < n and (nodes[i].get("depth") or 0) > 1:   # carry the paragraph's own children first
                out.append(nodes[i]); i += 1
            out.extend(new_nodes)
            done = True
            continue
        i += 1
    return out if done else nodes + new_nodes


def _tag_paragraph(nodes: list[dict], sub_num: str | None, note: str) -> None:
    """Attach *note* to the paragraph *sub_num*. Its chapeau node may be an empty
    container (text lives in a subparagraph, e.g. Art 6(3)/43(3)), which the
    renderer skips — so tag the first node of the paragraph that actually renders."""
    n = len(nodes)
    for i, node in enumerate(nodes):
        if node.get("depth") == 1 and (sub_num is None or (node.get("number") or "").strip() == sub_num):
            j = i
            while j < n and (j == i or (nodes[j].get("depth") or 0) > 1):
                if (nodes[j].get("text") or "").strip():
                    nodes[j]["_note"] = note
                    break
                j += 1
            if sub_num is not None:
                return


# ── rendering ────────────────────────────────────────────────────────────────

# Paragraph-level kinds render flush and reset the indent origin; everything else
# (points, roman sub-items, dash indents) nests beneath its paragraph.
_PARA_LEVEL = frozenset({"article", "paragraph", "subparagraph"})
_INDENT = "    "                    # one nesting level (nbsp — HTML keeps it, Markdown ignores it)


def _render_nodes(nodes: list[dict]) -> str:
    """Blockquote render: one markdown *paragraph* per unit, with a blank '>' line
    between each so every paragraph/point becomes its own ``<p>``.

    Enumerators are **bolded** ("**1.**", "**(a)**"). That is load-bearing, not
    cosmetic: it (i) stops ``marked`` parsing a leading "1." as an ordered-list
    item, and (ii) stops the front-end legal-list normaliser splitting a point at
    an inline cross-reference (e.g. the "point (a)" inside point (b)) — a bolded
    leading marker is preceded by '*', so it is no longer counted as an
    enumerator, and the lone inline reference then trips the normaliser's
    "needs >=2 markers" guard. No leading-space indentation (4 spaces would become
    a code block); the bold enumerators carry the hierarchy instead."""
    lines: list[str] = []
    first = True
    baseline = 0                                       # depth of the nearest paragraph-level ancestor
    for nd in nodes:
        depth = nd.get("depth") or 0
        kind = nd.get("kind")
        if kind in _PARA_LEVEL:
            baseline = depth                           # paragraphs/subparagraphs reset the indent origin
        if depth == 0:
            continue                                   # heading is rendered above the quote
        text = re.sub(r"\s+", " ", (nd.get("text") or "").replace("\xa0", " ")).strip()
        if not text:
            continue
        number = (nd.get("number") or "").strip()
        if kind in _PARA_LEVEL:
            body = re.sub(r"^(\d+[a-z]?\.)\s*", r"**\1** ", text)   # bold the leading N. / Na.
        elif kind == "indent":
            body = f"— {text}"
        elif number:
            body = f"**({number})** {text}"
        else:
            body = text
        if nd.get("_inserted_by"):
            body += f" — *inserted by {nd['_inserted_by']}*"
        if nd.get("_note"):
            body += f" — *{nd['_note']}*"
        # Points/sub-points nest under their paragraph via NON-ASCII (nbsp) indent
        # — HTML preserves it, but Markdown does not read it as a code block / list,
        # so the hierarchy shows (e.g. Art 7(2)(k)(i)/(ii) under (k)) without the
        # mangling ASCII indentation caused. Depth is measured from the paragraph,
        # so a point under a subparagraph is not double-indented.
        visual = 0 if kind in _PARA_LEVEL else max(depth - baseline, 1)
        indent = _INDENT * visual
        if not first:
            lines.append(">")                          # blank quote line between every unit
        lines.append(f"> {indent}{body}")
        first = False
    return "\n".join(lines)


def _pick_subject(provisions: list[dict], ref: str, celex: str) -> dict | None:
    want = ref.strip().lower()
    for p in provisions:
        pref = (p.get("article_ref") or p.get("display_ref") or "").strip().lower()
        if pref == want and p.get("regulation_id") == celex and p.get("subtree"):
            return p
    return next((p for p in provisions if p.get("subtree")), None)


def render_provision_display(retriever, ref: str, celex: str) -> str | None:
    """Verbatim render of *ref* in *celex*, with in-force amendments spliced in.
    Returns markdown, or ``None`` when the provision can't be resolved (the caller
    falls back to the generative path)."""
    try:
        provisions = retriever.retrieve_by_refs([ref], {celex})
    except Exception:
        return None
    subject = _pick_subject(provisions, ref, celex)
    if not subject or not subject.get("subtree"):
        return None
    subtree = subject["subtree"]
    root_id = subtree[0].get("id")
    base_ref = (subtree[0].get("ref") or ref).strip()
    heading = re.sub(r"\s+", " ", (subtree[0].get("text") or "")).strip()

    amendments: list[dict] = []
    if root_id:
        try:
            amendments = retriever.retrieve_amendments([root_id]) or []
        except Exception:
            amendments = []

    nodes, act_labels, notes = _splice(list(subtree), amendments, base_ref)
    body = _render_nodes(nodes)
    if not body:
        return None
    if notes:
        body += "\n>\n" + "\n>\n".join(f"> {n}" for n in notes)

    reg_name = (LEGISLATION.get(celex, {}) or {}).get("name", celex)
    title = f"### {base_ref} — {heading}" if heading else f"### {base_ref}"
    if act_labels:
        subtitle = (
            f"*{reg_name}, shown as currently in force — consolidating amendments by "
            f"{', '.join(act_labels)}. Verbatim source text with in-force amendments marked; "
            f"not legal advice.*"
        )
    else:
        subtitle = f"*{reg_name} — verbatim source text; not legal advice.*"

    out = f"{title}\n{subtitle}\n\n{body}"
    provenance = _build_amendment_provenance(out, amendments)
    return out + provenance if provenance else out
