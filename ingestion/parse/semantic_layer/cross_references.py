"""
Cross-reference extraction and resolution for EU legislative provisions.

This module provides:

1. **Extraction**  — ``extract_raw_refs(text)`` runs all regex patterns from
   :mod:`domain.ontology.cross_reference_patterns` against a provision's text,
   returning raw match dicts with category and captured groups.

2. **Resolution** — ``CrossReferenceResolver`` is a state-aware resolver that
   walks the flat provisions list, tracks the current Article / Paragraph
   context, and converts raw matches into concrete ``relation`` dicts with
   fully-qualified ``source`` and ``target`` provision IDs ready for the
   graph database.

3. **Enumeration expansion** — ``expand_range_ref(match)`` splits range and
   joint references ("Articles 8, 9, 10 and 11") into individual target
   numbers.

Architecture
------------
The resolver is designed to run as a **post-parse pass** over the provisions
list produced by the structural parsers.  It does *not* touch the DOM — only
the flat ``provisions`` and their ``text`` fields.

The output ``relations`` list uses the same schema as
:data:`domain.schema.graph_schema.json` → ``relation``:
``{"source": "<id>", "type": "<rel_type>", "target": "<id>", "properties": {...}}``.
"""
from __future__ import annotations

import re
from typing import Any

from domain.ontology.cross_reference_patterns import (
    ALL_PATTERNS,
    FOOTNOTE_MARKER,
)


# ---------------------------------------------------------------------------
# Qualifier pattern: "Article X(...), point (y) of" / "Annex Y, Section Z of/to"
# preceding an external reference.  Captured so the crosslinker can resolve
# to a specific provision in the target regulation.
# ---------------------------------------------------------------------------
_QUALIFIER_RE = re.compile(
    r"""
    (?:
        Articles?\s+(?P<article>\d+)
        (?:\((?P<para>\d+)\))?                      # (paragraph)
        (?:,?\s*(?:first|second|third|fourth|fifth)\s+subparagraph)?
        (?:,?\s*point\s+\((?P<point>[a-z0-9]+)\)
           (?:\((?P<subpoint>[a-z0-9]+)\))?
        )?
    |
        Annex(?:es)?\s+(?P<annex>[IVX]+|\d+)
        (?:,?\s*Section\s+(?P<section>[A-Z]|\d+))?
    )
    \s+(?:of|to)\s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Raw extraction (stateless)
# ---------------------------------------------------------------------------

def extract_raw_refs(text: str) -> list[dict[str, Any]]:
    """Run all cross-reference patterns on *text* and return raw matches.

    Each result dict contains:
      - category: pattern category name (e.g. "explicit", "relative")
      - span: (start, end) character offsets in the cleaned text
      - match: matched substring
      - groups: dict of non-None named groups

    Overlapping matches are deduplicated: when two spans overlap, the
    longer (more specific) match wins.  For equal-length overlaps, the
    priority order from ALL_PATTERNS is respected (relative > explicit >
    range > external).
    """
    clean = FOOTNOTE_MARKER.sub("", text)
    results: list[dict[str, Any]] = []
    for category, pattern in ALL_PATTERNS.items():
        for m in pattern.finditer(clean):
            results.append({
                "category": category,
                "span": m.span(),
                "match": m.group(),
                "groups": {k: v for k, v in m.groupdict().items() if v},
            })

    return _deduplicate_overlaps(results)


def _deduplicate_overlaps(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove overlapping matches, keeping the longer (more specific) one.

    When an explicit match like ``Article 13`` is fully contained within a
    range match like ``Articles 13, 14 and 16``, the shorter explicit match
    is dropped to avoid duplicate edges.
    """
    if len(refs) <= 1:
        return refs

    # Sort by start position, then by descending span length
    refs.sort(key=lambda r: (r["span"][0], -(r["span"][1] - r["span"][0])))

    kept: list[dict[str, Any]] = []
    for ref in refs:
        s, e = ref["span"]
        # Check if this ref is fully contained in an already-kept ref
        subsumed = False
        for k in kept:
            ks, ke = k["span"]
            if ks <= s and e <= ke and (s, e) != (ks, ke):
                subsumed = True
                break
        if not subsumed:
            kept.append(ref)

    return kept


# ---------------------------------------------------------------------------
# Range / joint-reference expansion
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"\d+(?:\(\d+\))?")


def expand_range_ref(groups: dict[str, str]) -> list[str]:
    """Expand a RANGE_REF match into a list of individual target numbers.

    For spans ("102 to 109"), returns all integers in [start, end].
    For enumerations ("8, 9, 10 and 11"), returns each listed number.
    """
    start = groups.get("start", "")
    end = groups.get("end")
    middle = groups.get("middle", "") or ""
    last = groups.get("last")

    if end is not None:
        # Span: "102 to 109"
        start_num = _extract_base_num(start)
        end_num = _extract_base_num(end)
        if start_num is not None and end_num is not None:
            return [str(n) for n in range(start_num, end_num + 1)]
        return [start, end]

    if last is not None:
        # Enumeration: "8, 9, 10 and 11"
        nums = [start]
        nums.extend(n.strip() for n in _NUM_RE.findall(middle))
        nums.append(last)
        return nums

    return [start]


def _extract_base_num(s: str) -> int | None:
    """Extract the leading integer from a string like '5(2)' → 5, or '109' → 109."""
    m = re.match(r"^\(?(\d+)", s)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# State-aware resolver
# ---------------------------------------------------------------------------

class CrossReferenceResolver:
    """Walks a provisions list and resolves cross-references into relations.

    The resolver tracks the current structural context (article, paragraph)
    while iterating provisions in document order.  Relative references are
    filled in using this state.

    Usage::

        resolver = CrossReferenceResolver(celex="32024R1689", provisions=provisions)
        relations = resolver.resolve_all()
    """

    # Relationship types emitted
    REL_CITES     = "CITES"
    REL_CITES_EXT = "CITES_EXTERNAL"
    REL_AMENDS    = "AMENDS"

    def __init__(self, celex: str, provisions: list[dict]) -> None:
        self.celex = celex
        self.provisions = provisions
        self._by_id: dict[str, dict] = {p["id"]: p for p in provisions}
        self._index = _build_provision_index(celex, provisions)

    _ORDINAL_MAP: dict[str, str] = {
        "first": "1", "second": "2", "third": "3",
        "fourth": "4", "fifth": "5",
    }

    def resolve_all(self) -> list[dict[str, Any]]:
        """Iterate all provisions, extract and resolve cross-references.

        Returns a list of relation dicts.
        """
        relations: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str]] = set()
        # Context state for relative references
        ctx_article: str | None = None
        ctx_paragraph: str | None = None
        ctx_subparagraph: str | None = None
        ctx_chapter: str | None = None

        for prov in self.provisions:
            kind = prov.get("kind", "")

            # Update structural context as we walk
            if kind == "chapter":
                ctx_chapter = prov.get("number")
            elif kind == "article":
                ctx_article = prov.get("number")
                ctx_paragraph = None
                ctx_subparagraph = None
                if ctx_chapter is None:
                    ctx_chapter = self._ancestor_number(prov, "chapter")
            elif kind == "paragraph":
                ctx_paragraph = prov.get("number")
                ctx_subparagraph = None
                if ctx_article is None:
                    ctx_article = self._article_of(prov)
            elif kind == "subparagraph":
                ctx_subparagraph = prov.get("number")
                if ctx_paragraph is None:
                    ctx_paragraph = self._ancestor_number(prov, "paragraph")
                if ctx_article is None:
                    ctx_article = self._article_of(prov)

            text = prov.get("text", "")
            if not text:
                continue

            clean_text = FOOTNOTE_MARKER.sub("", text)
            raw_refs = extract_raw_refs(text)
            source_id = prov["id"]

            for ref in raw_refs:
                rels = self._resolve_single(
                    ref, source_id, ctx_article, ctx_paragraph,
                    ctx_subparagraph, clean_text, ctx_chapter,
                )
                for rel in rels:
                    if rel["source"] == rel["target"]:
                        continue
                    edge_key = (rel["source"], rel["target"])
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    relations.append(rel)

        return relations

    # ------------------------------------------------------------------
    # Internal resolution dispatch
    # ------------------------------------------------------------------

    def _resolve_single(
        self,
        ref: dict,
        source_id: str,
        ctx_article: str | None,
        ctx_paragraph: str | None,
        ctx_subparagraph: str | None = None,
        source_text: str = "",
        ctx_chapter: str | None = None,
    ) -> list[dict[str, Any]]:
        category = ref["category"]
        groups = ref["groups"]

        if category == "explicit":
            return self._resolve_explicit(groups, source_id)
        if category == "relative":
            return self._resolve_relative(
                groups, source_id, ctx_article, ctx_paragraph,
                ctx_subparagraph, ctx_chapter,
            )
        if category == "range":
            return self._resolve_range(groups, source_id)
        if category == "external":
            return self._resolve_external(
                groups, source_id, ref["match"],
                source_text, ref.get("span", (0, 0))[0],
            )
        if category == "amended_by":
            return self._resolve_amended_by(groups, source_id)
        return []

    # ------------------------------------------------------------------
    # Explicit refs: "Article 5(1), point (g)" / "Annex III"
    # ------------------------------------------------------------------

    def _resolve_explicit(
        self, groups: dict, source_id: str,
    ) -> list[dict[str, Any]]:
        # Decimal annex-point enumeration: "Points 4.3., 4.4., 4.5. ... of Annex VII"
        dec_annex = groups.get("dec_annex")
        if dec_annex:
            nums = self._expand_decimal_enum(groups)
            relations = []
            for num in nums:
                target = self._lookup_annex_point(dec_annex, num)
                if not target:
                    target = self._lookup("annex", dec_annex)
                if target:
                    relations.append(self._make_rel(
                        source_id, target, self.REL_CITES,
                        {"ref_text": f"point {num} of Annex {dec_annex}"},
                    ))
            return relations

        # "Section N of Chapter X" (explicit chapter-section)
        cpt_of_sec = groups.get("cpt_of_sec")
        if cpt_of_sec:
            sec_num = groups.get("sec_of_cpt", "")
            target = self._lookup_section(cpt_of_sec, sec_num)
            if target:
                return [self._make_rel(source_id, target, self.REL_CITES,
                                       {"ref_text": f"Section {sec_num} of Chapter {cpt_of_sec}"})]
            return []

        # "Section X of Annex Y" (reversed order)
        annex_of_sec = groups.get("annex_of_sec")
        if annex_of_sec:
            sec_id = groups.get("sec_of_annex", "")
            target = self._lookup_annex_section(annex_of_sec, sec_id)
            if not target:
                target = self._lookup("annex", annex_of_sec)
            if target:
                ref = f"Section {sec_id} of Annex {annex_of_sec}"
                return [self._make_rel(source_id, target, self.REL_CITES,
                                       {"ref_text": ref})]
            return []

        # "point N of Annex Y" (reversed order)
        annex_of_pt = groups.get("annex_of_pt")
        if annex_of_pt:
            pt_num = groups.get("pt_of_annex", "")
            target = self._lookup_annex_point(annex_of_pt, pt_num)
            if not target:
                target = self._lookup("annex", annex_of_pt)
            if target:
                ref = f"point {pt_num} of Annex {annex_of_pt}"
                return [self._make_rel(source_id, target, self.REL_CITES,
                                       {"ref_text": ref})]
            return []

        # Annex branch (forward order: "Annex III, Section A, point 2")
        annex = groups.get("annex")
        if annex:
            section = groups.get("section")
            annex_pt = groups.get("annex_pt")
            # Try to resolve to the deepest available node
            if annex_pt and section:
                target = self._lookup_annex_point(annex, annex_pt)
            elif section:
                target = self._lookup_annex_section(annex, section)
            elif annex_pt:
                target = self._lookup_annex_point(annex, annex_pt)
            else:
                target = None
            if not target:
                target = self._lookup("annex", annex)
            if target:
                ref_parts = [f"Annex {annex}"]
                if section:
                    ref_parts.append(f"Section {section}")
                if annex_pt:
                    ref_parts.append(f"point {annex_pt}")
                return [self._make_rel(source_id, target, self.REL_CITES,
                                       {"ref_text": ", ".join(ref_parts)})]
            return []

        # Article branch — resolve to the deepest matching node
        article = groups.get("article")
        if not article:
            return []

        target = self._resolve_article_chain(
            article,
            groups.get("para"),
            groups.get("point"),
            groups.get("subpoint"),
        )
        if target:
            ref_parts = [f"Article {article}"]
            if groups.get("para"):
                ref_parts.append(f"({groups['para']})")
            if groups.get("ordinal"):
                ref_parts.append(f"{groups['ordinal']} subparagraph")
            if groups.get("point"):
                ref_parts.append(f"point ({groups['point']})")
            if groups.get("subpoint"):
                ref_parts.append(f"({groups['subpoint']})")
            props: dict[str, Any] = {"ref_text": " ".join(ref_parts)}
            if groups.get("ordinal"):
                props["subparagraph_ordinal"] = groups["ordinal"].lower()
            return [self._make_rel(source_id, target, self.REL_CITES, props)]
        return []

    # ------------------------------------------------------------------
    # Relative refs: "paragraph 3", "the first subparagraph"
    # ------------------------------------------------------------------

    def _resolve_relative(
        self,
        groups: dict,
        source_id: str,
        ctx_article: str | None,
        ctx_paragraph: str | None,
        ctx_subparagraph: str | None = None,
        ctx_chapter: str | None = None,
    ) -> list[dict[str, Any]]:
        # "this paragraph" / "this Article" / "this subparagraph"
        this_target = groups.get("this_target")
        if this_target:
            target = self._resolve_this(
                this_target.lower(), source_id,
                ctx_article, ctx_paragraph, ctx_subparagraph,
            )
            if target:
                return [self._make_rel(source_id, target, self.REL_CITES,
                                       {"ref_text": f"this {this_target}",
                                        "relative": True})]
            return []

        # "of this Article" / "of this paragraph"
        of_this = groups.get("of_this")
        if of_this:
            target = self._resolve_this(
                of_this.lower(), source_id,
                ctx_article, ctx_paragraph, ctx_subparagraph,
            )
            if target:
                return [self._make_rel(source_id, target, self.REL_CITES,
                                       {"ref_text": f"of this {of_this}",
                                        "relative": True})]
            return []

        # "referred to in Article N" / "pursuant to Article N"
        # Optionally followed by ", first subparagraph, point (h)"
        ref_kw = groups.get("ref_kw")
        ref_num = groups.get("ref_num")
        if ref_kw and ref_num:
            ref_pt = groups.get("ref_pt")
            ref_subpt = groups.get("ref_subpt")
            if ref_kw.lower() == "article":
                # "referred to in Article N[, para, point(subpoint)]" — drill down
                if ref_pt:
                    target = self._resolve_article_chain(ref_num, None, ref_pt, ref_subpt)
                else:
                    target = self._lookup("article", ref_num)
            else:
                # "referred to in paragraph N[, subparagraph, point (x)(y)]"
                if ctx_article:
                    if ref_pt and ref_subpt:
                        target = self._index.get(
                            ("roman_item", f"{ctx_article}.{ref_num}.{ref_pt}.{ref_subpt}")
                        )
                        if not target:
                            target = self._lookup_point(ctx_article, ref_num, ref_pt)
                    elif ref_pt:
                        target = self._lookup_point(ctx_article, ref_num, ref_pt)
                    else:
                        target = self._lookup_para(ctx_article, ref_num)
                else:
                    target = None
            if target:
                qualifier = groups.get("qualifier", "")
                ref_text_parts = [qualifier, ref_kw, ref_num]
                if groups.get("ref_sub_ord"):
                    ref_text_parts.append(f"{groups['ref_sub_ord']} subparagraph")
                if ref_pt:
                    ref_text_parts.append(f"point ({ref_pt})")
                if ref_subpt:
                    ref_text_parts.append(f"({ref_subpt})")
                return [self._make_rel(source_id, target, self.REL_CITES,
                                       {"ref_text": " ".join(ref_text_parts).strip(),
                                        "relative": True})]
            return []

        # "paragraph N[, point (x)]"
        para = groups.get("para")
        if para and ctx_article:
            para_pt = groups.get("para_pt")
            if para_pt:
                target = self._lookup_point(ctx_article, para, para_pt)
            else:
                target = self._lookup_para(ctx_article, para)
            if target:
                return [self._make_rel(source_id, target, self.REL_CITES,
                                       {"ref_text": f"paragraph {para}",
                                        "relative": True})]
            return []

        # "point (x)(y) of …" — resolve within current article/para
        pt_letter = groups.get("pt_letter")
        if pt_letter and ctx_article:
            pt_sub = groups.get("pt_sub")
            para_ctx = ctx_paragraph or "1"
            if pt_sub:
                target = self._index.get(
                    ("roman_item", f"{ctx_article}.{para_ctx}.{pt_letter}.{pt_sub}")
                )
                if not target:
                    target = self._lookup_point(ctx_article, para_ctx, pt_letter)
            else:
                target = self._lookup_point(ctx_article, para_ctx, pt_letter)
            if target:
                ref_text = f"point ({pt_letter})"
                if pt_sub:
                    ref_text += f"({pt_sub})"
                return [self._make_rel(source_id, target, self.REL_CITES,
                                       {"ref_text": ref_text,
                                        "relative": True})]

        # "the first subparagraph [of paragraph N]" — ordinal subparagraph
        ordinal = groups.get("ordinal")
        if ordinal and groups.get("the"):
            sp_num = self._ORDINAL_MAP.get(ordinal.lower())
            if sp_num:
                of_para = groups.get("of_para")
                para_ctx = of_para or ctx_paragraph
                if ctx_article and para_ctx:
                    target = self._index.get(
                        ("subparagraph", f"{ctx_article}.{para_ctx}.{sp_num}")
                    )
                    if target:
                        ref_text = f"the {ordinal} subparagraph"
                        if of_para:
                            ref_text += f" of paragraph {of_para}"
                        return [self._make_rel(source_id, target, self.REL_CITES,
                                               {"ref_text": ref_text,
                                                "relative": True})]

        # "Section N of this Chapter"
        sec_num = groups.get("sec_num")
        sec_scope = groups.get("sec_scope")
        if sec_num and sec_scope:
            if sec_scope.lower() == "chapter" and ctx_chapter:
                target = self._lookup_section(ctx_chapter, sec_num)
                if target:
                    return [self._make_rel(source_id, target, self.REL_CITES,
                                           {"ref_text": f"Section {sec_num} of this Chapter",
                                            "relative": True})]

        return []

    # ------------------------------------------------------------------
    # Range refs: "Articles 102 to 109" / "Articles 8, 9, 10 and 11"
    # ------------------------------------------------------------------

    def _resolve_range(
        self, groups: dict, source_id: str,
    ) -> list[dict[str, Any]]:
        kind_raw = groups.get("kind", "").lower().rstrip("s")  # "article"
        nums = expand_range_ref(groups)

        relations = []
        for num in nums:
            target = self._lookup(kind_raw, num)
            if target:
                relations.append(
                    self._make_rel(source_id, target, self.REL_CITES, {
                        "ref_text": f"{groups.get('kind', '')} {num}",
                    })
                )
        return relations

    # ------------------------------------------------------------------
    # External refs: "Regulation (EU) 2016/679"
    # ------------------------------------------------------------------

    def _resolve_external(
        self, groups: dict, source_id: str, match_text: str,
        source_text: str = "", match_start: int = 0,
    ) -> list[dict[str, Any]]:
        doc_type = groups.get("doc_type", "")
        series = groups.get("series", "")
        number = groups.get("number", "")
        suffix = groups.get("suffix", "")

        ext_id = _external_act_id(doc_type, series, number, suffix)

        # Capture "Article X(Y), point (z) of" / "Annex Y of" qualifier
        ref_text = match_text.strip()
        if source_text and match_start > 0:
            preceding = source_text[:match_start]
            qm = _QUALIFIER_RE.search(preceding)
            if qm:
                ref_text = f"{qm.group().strip()} {ref_text}"

        return [self._make_rel(source_id, ext_id, self.REL_CITES_EXT, {
            "ref_text": ref_text,
            "doc_type": doc_type,
            "series": series,
            "number": number,
        })]

    # ------------------------------------------------------------------
    # "as amended by" relationships
    # ------------------------------------------------------------------

    def _resolve_amended_by(
        self, groups: dict, source_id: str,
    ) -> list[dict[str, Any]]:
        base_id = _external_act_id(
            groups.get("base_type", ""),
            groups.get("base_series", ""),
            groups.get("base_number", ""),
            groups.get("base_suffix", ""),
        )
        amending_id = _external_act_id(
            groups.get("amending_type", ""),
            groups.get("amending_series", ""),
            groups.get("amending_number", ""),
            groups.get("amending_suffix", ""),
        )
        return [self._make_rel(amending_id, base_id, self.REL_AMENDS, {
            "source_provision": source_id,
        })]

    # ------------------------------------------------------------------
    # Node lookup helpers
    # ------------------------------------------------------------------

    def _lookup(self, kind: str, number: str) -> str | None:
        """Look up a provision ID by kind and number."""
        return self._index.get((kind, number))

    def _lookup_para(self, article: str, para: str) -> str | None:
        return self._index.get(("paragraph", f"{article}.{para}"))

    def _lookup_point(self, article: str, para: str, point: str) -> str | None:
        return self._index.get(("point", f"{article}.{para}.{point}"))

    def _resolve_article_chain(
        self,
        article: str,
        para: str | None,
        point: str | None,
        subpoint: str | None,
    ) -> str | None:
        """Resolve the deepest matching node in the article hierarchy."""
        if subpoint and point and para:
            t = self._index.get(("roman_item", f"{article}.{para}.{point}.{subpoint}"))
            if t:
                return t
        if point and para:
            t = self._lookup_point(article, para, point)
            if t:
                return t
        if point and not para:
            # Try direct article-level point first (e.g. definitions article)
            if subpoint:
                t = self._index.get(("roman_item", f"{article}.{point}.{subpoint}"))
                if t:
                    return t
            t = self._index.get(("point", f"{article}.{point}"))
            if t:
                return t
            # Fall back: scan paragraph-based points
            for key, pid in self._index.items():
                if key[0] == "point" and key[1].startswith(f"{article}.") and key[1].endswith(f".{point}"):
                    return pid
        if para:
            t = self._lookup_para(article, para)
            if t:
                return t
        return self._lookup("article", article)

    def _article_of(self, prov: dict) -> str | None:
        """Walk parent chain to find the enclosing article number."""
        return self._ancestor_number(prov, "article")

    def _ancestor_number(self, prov: dict, target_kind: str) -> str | None:
        """Walk parent chain to find the nearest ancestor of *target_kind*."""
        pid = prov.get("parent_id")
        while pid:
            parent = self._by_id.get(pid)
            if not parent:
                break
            if parent.get("kind") == target_kind:
                return parent.get("number")
            pid = parent.get("parent_id")
        return None

    # ------------------------------------------------------------------
    # "this" reference helpers
    # ------------------------------------------------------------------

    def _resolve_this(
        self,
        target_kind: str,
        source_id: str,
        ctx_article: str | None,
        ctx_paragraph: str | None,
        ctx_subparagraph: str | None,
    ) -> str | None:
        """Resolve 'this paragraph', 'this Article', 'this subparagraph'."""
        if target_kind == "article" and ctx_article:
            return self._lookup("article", ctx_article)
        if target_kind == "paragraph" and ctx_article and ctx_paragraph:
            return self._lookup_para(ctx_article, ctx_paragraph)
        if target_kind == "subparagraph":
            if ctx_article and ctx_paragraph and ctx_subparagraph:
                return self._index.get(
                    ("subparagraph", f"{ctx_article}.{ctx_paragraph}.{ctx_subparagraph}")
                )
            # Fall back: the source node itself if it is a subparagraph
            src = self._by_id.get(source_id)
            if src and src.get("kind") == "subparagraph":
                return source_id
        return None

    # ------------------------------------------------------------------
    # Annex lookup helpers
    # ------------------------------------------------------------------

    def _lookup_annex_section(self, annex_num: str, section_id: str) -> str | None:
        return self._index.get(("annex_section", f"{annex_num}.{section_id}"))

    def _lookup_annex_point(self, annex_num: str, point_num: str) -> str | None:
        return self._index.get(("annex_point", f"{annex_num}.{point_num}"))

    def _lookup_section(self, chapter_num: str, section_num: str) -> str | None:
        """Look up a chapter-qualified section."""
        return self._index.get(("section", f"{chapter_num}.{section_num}"))

    @staticmethod
    def _expand_decimal_enum(groups: dict) -> list[str]:
        """Expand decimal annex-point enumeration into individual point numbers."""
        nums = [groups.get("dec_start", "")]
        middle = groups.get("dec_middle", "") or ""
        for m in re.finditer(r"\d+\.\d+", middle):
            nums.append(m.group())
        last = groups.get("dec_last")
        if last:
            nums.append(last)
        return [n for n in nums if n]

    # ------------------------------------------------------------------
    # Relation factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_rel(
        source: str, target: str, rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "source": source,
            "type": rel_type,
            "target": target,
            "properties": properties or {},
        }


# ---------------------------------------------------------------------------
# Provision index builder
# ---------------------------------------------------------------------------

def _build_provision_index(
    celex: str, provisions: list[dict],
) -> dict[tuple[str, str], str]:
    """Build a lookup index: (kind, compound_key) → provision ID.

    Compound keys:
      - article:         number                          (e.g. "5")
      - paragraph:       article_num.para_num            (e.g. "5.1")
      - subparagraph:    article_num.para_num.sp_num     (e.g. "5.1.2")
      - point:           article_num.para_num.point_lbl  (e.g. "5.1.g")
      - roman_item:      article.para.point.roman        (e.g. "5.1.g.i")
      - annex:           number                          (e.g. "III")
      - annex_section:   annex_num.section_id            (e.g. "I.A", "VIII.1")
      - annex_point:     annex_num.point_num             (e.g. "III.2")
      - chapter:         number                          (e.g. "III")
      - section:         chapter_num.number                (e.g. "III.1")
      - recital:         number                          (e.g. "170")
    """
    index: dict[tuple[str, str], str] = {}
    # Pre-build parent lookup
    by_id: dict[str, dict] = {p["id"]: p for p in provisions}

    def _ancestor_number(prov: dict, target_kind: str) -> str | None:
        pid = prov.get("parent_id")
        while pid:
            parent = by_id.get(pid)
            if not parent:
                break
            if parent.get("kind") == target_kind:
                return parent.get("number")
            pid = parent.get("parent_id")
        return None

    for prov in provisions:
        kind = prov.get("kind", "")
        number = prov.get("number")
        if not number:
            continue

        if kind == "article":
            index[("article", number)] = prov["id"]
        elif kind == "annex":
            index[("annex", number)] = prov["id"]
        elif kind == "chapter":
            index[("chapter", number)] = prov["id"]
        elif kind == "section":
            chapter_num = _ancestor_number(prov, "chapter")
            if chapter_num:
                index[("section", f"{chapter_num}.{number}")] = prov["id"]
            else:
                index[("section", number)] = prov["id"]
        elif kind == "recital":
            index[("recital", number)] = prov["id"]
        elif kind == "paragraph":
            art_num = _ancestor_number(prov, "article")
            if art_num:
                index[("paragraph", f"{art_num}.{number}")] = prov["id"]
        elif kind == "subparagraph":
            art_num = _ancestor_number(prov, "article")
            para_num = _ancestor_number(prov, "paragraph")
            if art_num and para_num:
                index[("subparagraph", f"{art_num}.{para_num}.{number}")] = prov["id"]
        elif kind == "point":
            art_num = _ancestor_number(prov, "article")
            para_num = _ancestor_number(prov, "paragraph")
            if art_num and para_num:
                index[("point", f"{art_num}.{para_num}.{number}")] = prov["id"]
            elif art_num:
                # Direct article-level points (e.g. definitions article — no paragraph)
                index[("point", f"{art_num}.{number}")] = prov["id"]
        elif kind == "roman_item":
            art_num = _ancestor_number(prov, "article")
            para_num = _ancestor_number(prov, "paragraph")
            pt_num = _ancestor_number(prov, "point")
            if art_num and para_num and pt_num:
                index[("roman_item", f"{art_num}.{para_num}.{pt_num}.{number}")] = prov["id"]
            elif art_num and pt_num:
                # Direct article-level roman items (under a no-paragraph point)
                index[("roman_item", f"{art_num}.{pt_num}.{number}")] = prov["id"]
        elif kind == "annex_section":
            annex_num = _ancestor_number(prov, "annex")
            if annex_num:
                index[("annex_section", f"{annex_num}.{number}")] = prov["id"]
        elif kind == "annex_point":
            annex_num = _ancestor_number(prov, "annex")
            if annex_num:
                index[("annex_point", f"{annex_num}.{number}")] = prov["id"]

    return index


def _external_act_id(
    doc_type: str, series: str, number: str, suffix: str,
) -> str:
    """Build a stable external-act identifier for graph nodes.

    Examples:
      "Regulation (EU) 2016/679"  → "ext_regulation_eu_2016_679"
      "Directive 2002/58/EC"      → "ext_directive_2002_58_ec"
    """
    parts = ["ext", doc_type.lower()]
    if series:
        parts.append(series.lower())
    if number:
        parts.append(number.replace("/", "_"))
    if suffix:
        parts.append(suffix.lower())
    return "_".join(parts)
