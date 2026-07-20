"""Answer faithfulness verification.

Deterministic post-generation check that every verbatim quote in the LLM
answer corresponds to text present in the retrieved provisions.  Closes the
"plausible but fabricated quotation" failure mode (e.g. citing
"Annex III Point 1(a)" for medical diagnosis when the actual Annex III
Point 1 covers biometrics).

Two modes:
    - flag mode (default when enabled): append a warning block listing the
      unverified quotes; the answer is still returned.
    - strict mode (future): re-prompt the LLM once with instructions to
      replace unverified quotes with grounded text.  Not implemented yet —
      currently falls back to flag behaviour at the integration layer.

Controlled by environment variable ``CRSS_FAITHFULNESS_CHECK`` (default "1"):
    "0"           -> disabled (no-op)
    "1" or unset  -> flag mode: redact unverified quotes + prepend a warning
    "2"           -> strict mode (currently equivalent to flag mode)

The check is fully deterministic.  No LLM calls.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from application.contracts import Definition, Provision

# Graduated grounding thresholds.  A quote fragment that is not an exact
# substring may still be "near-verbatim" — i.e. the model dropped an article
# ("from the input" -> "from input"), reordered words, or lightly reworded,
# without fabricating.  Such a fragment counts as grounded (so we never delete
# real law over a dropped "the") but is surfaced for a wording check.  Both
# guards must hold: high overall character recall AND a long single contiguous
# matching span (so scattered re-use of common words, or a flipped negation
# that splits the span, cannot masquerade as near-verbatim).
_NEAR_VERBATIM_RECALL: float = 0.90
_NEAR_VERBATIM_BLOCK: float = 0.60

# Structural guards against the "grounded but misattributed" failure mode.  The
# faithfulness check above verifies that quoted text exists *somewhere* in the
# retrieved corpus — but not that it belongs to the provision the answer cites
# for it.  Once the retriever force-loads an entire obligation cluster, the LLM
# can dump a wall of real provision text concatenated under a single citation
# (observed: a ~2,000-char "quote" attributed to one Article that actually
# strung together a dozen Articles' texts).  Such a quote is individually
# grounded fragment-by-fragment, so it sails through grounding_verdict.  Two
# deterministic guards catch it:
#   A. concatenation — a single quote drawing a long contiguous span from more
#      than ``_CONCAT_MAX_SOURCES`` distinct provisions is a dump, not a quote.
#   B. misattribution — a grounded quote whose text is absent from the specific
#      provision its nearest ``[Article X]`` label cites is displaced text.
# Both verdicts remove the quote with a distinct (non-"fabrication") flag.
_CONCAT_BLOCK_MIN: int = 50      # contiguous chars to credit a quote to a source
_CONCAT_MAX_SOURCES: int = 2     # > this many distinct sources => concatenation
_CONCAT_MIN_QUOTE: int = 200     # only long quotes can be multi-provision dumps
_ATTRIBUTION_WINDOW: int = 220   # chars before a quote scanned for its citation

# ---------------------------------------------------------------------------
# Quote extraction
# ---------------------------------------------------------------------------

# Minimum quoted-text length to bother verifying.  Below this threshold the
# substring check produces too many false positives on common legal phrases
# (e.g. "shall ensure", "where applicable") that appear across many provisions.
_MIN_QUOTE_LEN: int = 40

# Match a quoted span starting with any opening double-quote glyph and ending
# at the next matching closing glyph.  Captures the inner text.  Single quotes
# are intentionally excluded (apostrophes / possessives produce too much noise).
# The body excludes newlines: a verbatim legal quote is a single contiguous run
# of provision text, never a multi-line markdown block. Without this, an
# unbalanced/stray opening quote glyph matches greedily to a closing glyph many
# lines later, swallowing the model's own analysis (bullets, bold, headings)
# into one "quote". That span then fails faithfulness and is excised by offset \u2014
# fusing the surrounding words into nonsense ("medical devicesubstantial
# modification"). Keeping quotes single-line confines the match to a real quote.
_QUOTE_PATTERN = re.compile(
    r"""
    (?:[\*_]+)?               # optional leading markdown emphasis
    (?P<open>["\u201C\u201D\u201E\u201F\u00AB\u00BB])
    (?P<body>[^"\u201C\u201D\u201E\u201F\u00AB\u00BB\n]{40,})
    (?P<close>["\u201C\u201D\u201E\u201F\u00AB\u00BB])
    (?:[\*_]+)?               # optional trailing markdown emphasis
    """,
    re.VERBOSE,
)

# Upper bound on a single verbatim quote. The prompt asks for a "short operative
# fragment"; a single-line span longer than this is a runaway match or a dump,
# not a quotation \u2014 extracting (and possibly removing) it risks corrupting the
# answer, so it is skipped.
_MAX_QUOTE_LEN: int = 600

# Inserted where a fabricated/displaced quote is removed, so the surrounding
# words do not fuse into an unreadable splice.
_REDACTION_MARKER: str = "[\u2026]"


@dataclass(frozen=True)
class Quote:
    """A single verbatim quotation extracted from an answer."""

    text: str           # the inner quoted text, original
    start: int          # char offset in answer
    end: int            # char offset in answer (exclusive)

    @property
    def preview(self) -> str:
        """Short preview suitable for the warning block (<= 120 chars)."""
        body = " ".join(self.text.split())
        if len(body) <= 117:
            return body
        return body[:117].rstrip() + "..."


def extract_quotes(answer: str) -> list[Quote]:
    """Extract verbatim quotations from *answer*.

    Catches straight quotes, smart double quotes, and guillemets, including
    quotes wrapped in markdown emphasis (``*"..."*``).  Single quotes are
    intentionally ignored.  Only quotations of at least ``_MIN_QUOTE_LEN``
    characters are returned.
    """
    quotes: list[Quote] = []
    for match in _QUOTE_PATTERN.finditer(answer):
        body = match.group("body").strip()
        if len(body) < _MIN_QUOTE_LEN or len(body) > _MAX_QUOTE_LEN:
            # Too short → noise; too long → a runaway match or dump, not a
            # quotation. Skipping the latter keeps a stray quote glyph from
            # swallowing real analysis that would then be excised.
            continue
        quotes.append(Quote(text=body, start=match.start(), end=match.end()))
    return quotes


# ---------------------------------------------------------------------------
# Normalization + verification
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_ELLIPSIS_RE = re.compile(r"\[\s*\.\.\.\s*\]|\.\.\.|\u2026")
_MD_EMPHASIS_RE = re.compile(r"([*_]{1,3})(?=\S)(.+?)(?<=\S)\1")
# Hyphen / dash family folded to a single space.  LLMs routinely "tidy" the
# orthography of a verbatim quote \u2014 e.g. writing "machine-readable" where the
# EU text has "machine readable" \u2014 and a single such edit must not turn a
# grounded operational obligation into a false "unverified" flag (and deletion).
_DASH_RE = re.compile(r"[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
# Apostrophe / single-quote variants folded to a straight ASCII apostrophe so
# defined-term quotes like \u2018AI system\u2019 match regardless of the glyph the model
# emits.  (NFKC does NOT fold these, despite the old docstring's claim.)
_APOSTROPHE_RE = re.compile(r"[\u2018\u2019\u201a\u201b`\u00b4]")
_CITATION_REF_RE = re.compile(
    r"\b(Article\s+\d+[a-z]?(?:\(\d+\))?(?:\([a-z]\))?|"
    r"Annex\s+[IVXLC]+|Recital\s+\d+)(?=\W|$)",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Normalize text for substring matching.

    Applies, in order: NFKC normalization (folds ligatures, full-width forms and
    non-breaking spaces); markdown-emphasis stripping; lowercasing; folding of
    the hyphen/dash family and apostrophe/single-quote variants; and whitespace
    collapse.  The dash and apostrophe folds absorb the orthographic micro-edits
    LLMs make when quoting verbatim text (e.g. "machine-readable" vs "machine
    readable", or ‘ vs '), which would otherwise produce false "unverified"
    flags on genuinely grounded quotes.
    """
    text = unicodedata.normalize("NFKC", text)
    # Strip markdown emphasis markers (e.g. **text**, *text*) so legal
    # quotes copied with formatting can still match verbatim source text.
    text = _MD_EMPHASIS_RE.sub(r"\2", text)
    text = text.lower()
    text = _DASH_RE.sub(" ", text)
    text = _APOSTROPHE_RE.sub("'", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _split_on_ellipsis(quote: str) -> list[str]:
    """Split *quote* on ellipsis markers, returning normalized fragments."""
    fragments = [frag.strip() for frag in _ELLIPSIS_RE.split(quote)]
    return [_normalize(frag) for frag in fragments if frag.strip()]


def _fragment_verdict(fragment: str, corpus: str) -> str:
    """Classify a single fragment as ``'exact'``, ``'near'`` or ``'absent'``.

    Exact = verbatim substring.  Near = high character recall against a
    contiguous corpus window with a long single matching span (a trivial
    omission/rewording, not a fabrication).  Absent = neither.
    """
    if fragment in corpus:
        return "exact"
    flen = len(fragment)
    if flen == 0:
        return "absent"
    # Anchor the search on the fragment's longest tokens so difflib only runs
    # on a few bounded windows instead of the whole (large) corpus.
    tokens = sorted(set(fragment.split()), key=len, reverse=True)
    for anchor in tokens[:3]:
        if len(anchor) < 4:
            break
        start = 0
        checked = 0
        while checked < 40:
            idx = corpus.find(anchor, start)
            if idx == -1:
                break
            checked += 1
            lo = max(0, idx - flen - 40)
            hi = min(len(corpus), idx + flen + 40)
            window = corpus[lo:hi]
            matcher = difflib.SequenceMatcher(None, fragment, window, autojunk=False)
            blocks = matcher.get_matching_blocks()
            matched = sum(b.size for b in blocks)
            longest = max((b.size for b in blocks), default=0)
            if (
                matched / flen >= _NEAR_VERBATIM_RECALL
                and longest / flen >= _NEAR_VERBATIM_BLOCK
            ):
                return "near"
            start = idx + 1
    return "absent"


def grounding_verdict(quote_text: str, normalized_corpus: str) -> str:
    """Return ``'exact'``, ``'near'`` or ``'absent'`` for a whole quote.

    Splits on ellipsis markers; the quote's verdict is the *weakest* of its
    significant fragments (all exact -> exact; all grounded but ≥ one near
    -> near; any absent -> absent).  Fragments below the length threshold are
    skipped to avoid penalising short connectors flanking ellipses.
    """
    fragments = _split_on_ellipsis(quote_text)
    if not fragments:
        return "absent"
    min_frag = max(_MIN_QUOTE_LEN // 2, 20)
    significant = [frag for frag in fragments if len(frag) >= min_frag]
    if not significant:
        significant = fragments
    verdicts = [_fragment_verdict(frag, normalized_corpus) for frag in significant]
    if any(v == "absent" for v in verdicts):
        return "absent"
    return "near" if any(v == "near" for v in verdicts) else "exact"


def verify_quote(quote_text: str, normalized_corpus: str) -> bool:
    """Return True iff *quote_text* is grounded (exact or near-verbatim).

    Thin bool wrapper over :func:`grounding_verdict` for callers that only need
    grounded-vs-fabricated.  Near-verbatim quotes count as grounded — a dropped
    article must never be treated as a fabrication.
    """
    return grounding_verdict(quote_text, normalized_corpus) != "absent"


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------


def _provision_text(provision: dict[str, Any]) -> str:
    """Extract the verbatim text payload from a provision dict.

    Delegates to :meth:`application.contracts.Provision.text_payload` — the
    single authoritative definition of a provision's quotable text (body +
    children + interpretive-link lines, exactly what ``_context.py`` renders so
    the corpus contains everything the LLM saw). ``_faithfulness`` is the
    contract's first real consumer; the equivalence tests in
    ``tests/test_contracts.py`` pin the payload so it cannot drift.
    """
    return Provision.from_dict(provision).text_payload()


def _definition_text(definition: dict[str, Any]) -> str:
    """Extract the quotable text from a definition lookup result.

    Delegates to :meth:`application.contracts.Definition.text_payload`. The
    definitions block (formal/scoped defined-term definitions) is part of the
    REGULATORY CONTEXT the LLM sees, so quotes drawn from it — e.g. the AI Act
    Article 3(1) 'AI system' definition — must be verifiable.
    """
    return Definition.from_dict(definition).text_payload()


def _build_corpus(
    provisions: list[dict[str, Any]],
    definitions: list[dict[str, Any]] | None = None,
) -> str:
    """Build a single normalized corpus string from everything the LLM saw.

    The corpus must mirror the full REGULATORY CONTEXT — retrieved provisions
    (with their children and interpretive-link lines) *and* the definitions
    block — otherwise legitimately-grounded quotes from definitions or guidance
    are falsely flagged, which is why the check could not be enabled by default.
    """
    parts: list[str] = [_provision_text(p) for p in provisions if p]
    for d in definitions or []:
        if d:
            def_text = _definition_text(d)
            if def_text:
                parts.append(def_text)
    return _normalize("\n".join(parts))


def _normalize_ref(ref: str) -> str:
    """Normalize provision references for deterministic equality checks."""
    ref = " ".join(ref.split())
    ref = re.sub(r"\bannex\b", "Annex", ref, flags=re.IGNORECASE)
    ref = re.sub(r"\barticle\b", "Article", ref, flags=re.IGNORECASE)
    ref = re.sub(r"\brecital\b", "Recital", ref, flags=re.IGNORECASE)

    # Uppercase roman numerals in Annex refs (Annex xi -> Annex XI)
    m = re.match(r"^(Annex)\s+([ivxlc]+)$", ref, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)} {m.group(2).upper()}"
    return ref


def _base_ref_family(ref: str) -> str:
    """Collapse a citation ref to its base provision family.

    ``Article 5(1)(f)`` → ``Article 5``; ``Annex IX, Chapter I, point 3.5`` →
    ``Annex IX``; ``Recital 44`` → ``Recital 44``. Sources and citations that
    share a family refer to the same base provision at different depths —
    ancestor, descendant or sibling — and must adjudicate as one unit: the
    guard's old ancestor-only walk false-flagged verbatim Article 5(1)(f) text
    whenever the bag's source entry was keyed at a different depth than the
    answer's citation label.
    """
    ref = _normalize_ref(ref)
    m = re.match(r"^(Article\s+\d+[a-z]?)", ref, flags=re.IGNORECASE)
    if m:
        return _normalize_ref(m.group(1))
    m = re.match(r"^(Annex\s+[IVXLC]+)", ref, flags=re.IGNORECASE)
    if m:
        return _normalize_ref(m.group(1))
    m = re.match(r"^(Recital\s+\d+)", ref, flags=re.IGNORECASE)
    if m:
        return _normalize_ref(m.group(1))
    return ref


def _is_recital_family(ref: str) -> bool:
    """True when a source ref is a recital (non-operative preamble text)."""
    return _base_ref_family(ref).lower().startswith("recital")


def _article_ref_parent_chain(ref: str) -> list[str]:
    """Return progressively broader parent refs for an Article citation.

    Example: ``Article 5(1)(a)`` -> ["Article 5(1)", "Article 5"]
    """
    m = re.match(r"^(Article\s+\d+[a-z]?)(.*)$", ref, flags=re.IGNORECASE)
    if not m:
        return []
    base = _normalize_ref(m.group(1))
    tail = m.group(2) or ""
    parents: list[str] = []

    # Walk backwards across parenthetical segments.
    segments = re.findall(r"\([^()]+\)", tail)
    while segments:
        segments = segments[:-1]
        if segments:
            parents.append(base + "".join(segments))
        else:
            parents.append(base)
    if not segments and not parents:
        parents.append(base)
    return parents


def extract_citation_refs(answer: str) -> set[str]:
    """Extract cited Article/Annex/Recital refs from answer text."""
    return {_normalize_ref(m.group(1)) for m in _CITATION_REF_RE.finditer(answer)}


def extract_context_refs(provisions: list[dict[str, Any]]) -> set[str]:
    """Extract Article/Annex/Recital refs present in retrieved context."""
    refs: set[str] = set()
    for p in provisions or []:
        article_ref = p.get("article_ref")
        if isinstance(article_ref, str):
            if _CITATION_REF_RE.search(article_ref):
                refs.add(_normalize_ref(article_ref))
        for child in p.get("children", []) or []:
            child_ref = child.get("ref")
            if isinstance(child_ref, str):
                if _CITATION_REF_RE.search(child_ref):
                    refs.add(_normalize_ref(child_ref))
    return refs


def out_of_scope_citation_refs(answer: str, provisions: list[dict[str, Any]]) -> list[str]:
    """Return sorted list of cited refs not present in retrieved context."""
    cited = extract_citation_refs(answer)
    if not cited:
        return []
    ctx_refs = extract_context_refs(provisions)
    missing: list[str] = []
    for ref in sorted(cited):
        if ref in ctx_refs:
            continue
        # Treat nested article refs as in-scope when a broader parent article
        # is present in context (e.g. Article 5 covers Article 5(1)(a)).
        parents = _article_ref_parent_chain(ref)
        if any(parent in ctx_refs for parent in parents):
            continue
        missing.append(ref)
    return missing


# ---------------------------------------------------------------------------
# Structural guards (concatenation + misattribution)
# ---------------------------------------------------------------------------


def _build_sources(
    provisions: list[dict[str, Any]],
    definitions: list[dict[str, Any]] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Split the corpus into per-source normalized texts.

    Unlike :func:`_build_corpus` (one flat blob), this keeps one entry per
    distinct provision/definition so a quote can be tested against an individual
    source.  A provision's children fold into its own entry (a sub-paragraph
    quote belongs to its article), and same-ref entries are merged (a provision
    and the definition it carries share one source).  Returns ``(sources,
    source_map)`` where ``sources`` is the list of distinct normalized texts and
    ``source_map`` maps a normalized citation ref to its normalized text.
    """
    by_ref: dict[str, str] = {}
    unkeyed: list[str] = []

    def _add(ref: str | None, text: str) -> None:
        if not text:
            return
        if ref is None:
            unkeyed.append(text)
        else:
            by_ref[ref] = f"{by_ref[ref]}\n{text}" if ref in by_ref else text

    def _ref_of(raw: Any) -> str | None:
        if isinstance(raw, str) and _CITATION_REF_RE.search(raw):
            return _normalize_ref(raw)
        return None

    for p in provisions or []:
        if p:
            _add(_ref_of(p.get("article_ref")), _normalize(_provision_text(p)))
    for d in definitions or []:
        if d:
            _add(_ref_of(d.get("article_ref")), _normalize(_definition_text(d)))

    sources = list(by_ref.values()) + unkeyed
    return sources, by_ref, unkeyed


def _longest_match(a: str, b: str) -> int:
    """Length of the longest contiguous substring shared by *a* and *b*."""
    if not a or not b:
        return 0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b),
    ).size


def _distinct_source_count(
    quote_norm: str,
    source_map: dict[str, str],
    unkeyed: list[str],
) -> int:
    """Count distinct provision *families* contributing a >= ``_CONCAT_BLOCK_MIN``
    span to the quote.

    Family-collapsed, not per-source: a concatenation dump draws from several
    *different provisions*, so the count must be over base families
    (``_base_ref_family``), not over source entries. Without this, a verbatim
    quote of one point of an article whose sibling points share an opening
    chapeau (e.g. AI Act Article 5(1)'s "the placing on the market, the putting
    into service …", repeated across points (d)-(h)) matched the ≥50-char span
    in three separately-keyed sibling nodes and was miscounted as a 3-provision
    dump — a false ATTRIBUTION FLAG. Unkeyed sources (containers with no citable
    ref) each count individually, since a real dump across them is still
    possible.
    """
    families: set[str] = set()
    for ref, src in source_map.items():
        if _longest_match(quote_norm, src) >= _CONCAT_BLOCK_MIN:
            families.add(_base_ref_family(ref))
    count = len(families)
    for src in unkeyed:
        if _longest_match(quote_norm, src) >= _CONCAT_BLOCK_MIN:
            count += 1
    return count


# Any label a quote can be attributed to. Adjudicable labels (Article/Annex/
# Recital) are resolvable against retrieved sources; blocker labels (guidance
# section numbers, MDCG document ids, bare point numbers) are citation forms
# the source map cannot key, so a quote attributed to one must NOT be
# adjudicated at all — the old behaviour fell through to an *earlier, unrelated*
# Article match in the window and false-flagged guidance-cited quotes.
_ATTRIBUTION_LABEL_RE = re.compile(
    r"(?P<cite>Article\s+\d+[a-z]?(?:\(\d+\))?(?:\([a-z]\))?"
    r"|Annex\s+[IVXLC]+(?:,\s*(?:Chapter|Part|Section|point)\s+[\w.()]+)*"
    r"|Recital\s+\d+)"
    r"|(?P<blocker>MDCG\s*\d{4}[‑–-]\d+|Section\s+\d[\w.]*|point\s+\d[\w.]*)",
    re.IGNORECASE,
)


# Chars scanned *after* a quote (same line only) for a trailing citation.
# CRSS answers frequently use the cite-after layout — “quote…” (Article 10) —
# and adjudicating such a quote against the label of the *previous* bullet
# produced positive-displacement false flags (v6 eval: HQ_012, 11 cite-after
# quotes, 4 false misattributions).
_ATTRIBUTION_AFTER_WINDOW: int = 100


def _nearest_citation_ref(
    answer: str, quote_start: int, quote_end: int | None = None
) -> str | None:
    """Return the adjudicable citation ref nearest to a quote span.

    Scans the ``_ATTRIBUTION_WINDOW`` characters before the quote for the last
    attribution label and — when ``quote_end`` is given — the same-line
    ``_ATTRIBUTION_AFTER_WINDOW`` characters after it for a leading label
    (cite-after layout: ``“…” (Article 10)``); the label nearest to the quote
    span wins. Returns the normalized ref when that label is an
    Article/Annex/Recital; returns None both when there is no label and when
    the nearest label is a non-adjudicable form (guidance section, MDCG id) —
    in the latter case the quote is attributed to a source the map cannot
    resolve, and adjudicating it against some earlier Article ref in the
    window would be a false flag.
    """
    lo = max(0, quote_start - _ATTRIBUTION_WINDOW)
    window = answer[lo:quote_start]
    best: tuple[int, "re.Match[str]"] | None = None
    matches = list(_ATTRIBUTION_LABEL_RE.finditer(window))
    if matches:
        last = matches[-1]
        best = (len(window) - last.end(), last)
    if quote_end is not None:
        after = answer[quote_end : quote_end + _ATTRIBUTION_AFTER_WINDOW]
        newline = after.find("\n")
        if newline != -1:
            after = after[:newline]
        trailing = _ATTRIBUTION_LABEL_RE.search(after)
        if trailing and (best is None or trailing.start() < best[0]):
            best = (trailing.start(), trailing)
    if best is None:
        return None
    label = best[1]
    if label.group("blocker"):
        return None
    # Compound annex labels ("Annex IX, Chapter I, point 3.5") resolve at the
    # family level; extract the plain head the source map is keyed by.
    head = _CITATION_REF_RE.search(label.group("cite"))
    return _normalize_ref(head.group(1)) if head else None


def _resolve_cited_source(cited_ref: str, source_map: dict[str, str]) -> str | None:
    """Resolve a citation ref to its source text, walking up the parent chain.

    A quote cited as ``Article 43(4)`` is grounded by the ``Article 43``
    provision (whose text includes its sub-paragraphs), so parent refs are
    tried when the exact ref is not itself a retrieved source.
    """
    if cited_ref in source_map:
        return source_map[cited_ref]
    for parent in _article_ref_parent_chain(cited_ref):
        if parent in source_map:
            return source_map[parent]
    return None


def _structural_verdict(
    quote: "Quote",
    answer: str,
    source_map: dict[str, str],
    unkeyed: list[str],
) -> str | None:
    """Return ``'concatenated'`` / ``'misattributed'`` / None for a grounded quote.

    Only meaningful for quotes already grounded somewhere in the corpus — these
    guards distinguish "real text, wrong place" from genuine fabrication (which
    the absent verdict already handles).
    """
    quote_norm = _normalize(quote.text)
    if (
        len(quote_norm) >= _CONCAT_MIN_QUOTE
        and _distinct_source_count(quote_norm, source_map, unkeyed) > _CONCAT_MAX_SOURCES
    ):
        return "concatenated"
    cited = _nearest_citation_ref(answer, quote.start, quote.end)
    if cited:
        # Adjudicate against the cited provision's whole *family* — every
        # retrieved source keyed at any depth of the same base provision
        # (ancestor, descendant or sibling). Sources are keyed by whatever
        # depth the retrieval anchored ("Article 5" vs "Article 5(1)"), and
        # the answer cites at yet another depth; the old ancestor-only walk
        # false-flagged verbatim Article 5(1)(f) text whenever those depths
        # disagreed. Silent when no family source was retrieved (cannot
        # adjudicate — no false flag).
        family = _base_ref_family(cited)
        family_grounds = any(
            grounding_verdict(quote.text, src) != "absent"
            for ref, src in source_map.items()
            if _base_ref_family(ref) == family
        )
        if not family_grounds:
            # Displacement must be *proven*, not inferred from absence: the
            # retrieved copy of the cited provision can be incomplete (large
            # articles' flattened bodies and child expansions are capped), so
            # "grounded somewhere in the pooled corpus but not in my copy of
            # your citation" is not evidence of a wrong cite — observed:
            # verbatim IVDR Article 48(7) text, correctly cited, flagged
            # because Article 48's capped source text stopped before (7).
            # Flag only when a *different operative* family positively grounds
            # the quote — that is the exact claim the ATTRIBUTION FLAG makes.
            #
            # Recital families are excluded from the displacement proof: a
            # recital recites the rule its operative article enacts, so a
            # recital grounding an Article/Annex citation is the same rule
            # seen through its non-operative preamble, not a relocation to a
            # different provision (observed: verbatim Article 5(1)(f) and
            # Article 25(1)(c) text, correctly cited, grounded only in
            # Recital 44 / Recital 84 because the cited article's retrieved
            # copy was truncated). Container/chapter nodes need no exclusion —
            # their display_ref does not match the citation grammar, so they
            # are unkeyed and never enter this per-family adjudication.
            other_family_grounds = any(
                grounding_verdict(quote.text, src) != "absent"
                for ref, src in source_map.items()
                if _base_ref_family(ref) != family
                and not _is_recital_family(ref)
            )
            if other_family_grounds:
                return "misattributed"
    return None


# ---------------------------------------------------------------------------
# Illustrative-quote detection
#
# The model quotes text that was never *claimed* to be law: drafted sample
# wording ("your rejection notice could state: '…'"), scenario echoes it
# embellished beyond the user's own words, template placeholders ("lack of
# experience in X"). Verifying those against the legal corpus is a category
# error — they ground nowhere by construction — and counting them as
# fabrications dominated the v6 eval's regression (15 of 21 "fabricated"
# quotes: 8 notification templates in HQ_037, 5 app-marketing echoes in
# HQ_028, 2 in HQ_008). An ungrounded quote is treated as illustrative only
# when BOTH hold: no provision citation is attached to it (an attributed
# quote is always a legal-quote claim), and either the introducing clause
# carries an example cue or the quote itself reads as addressed prose /
# template text. Illustrative quotes are kept in the answer and reported
# separately — never redacted, never counted as fabricated.
# ---------------------------------------------------------------------------

_ILLUSTRATIVE_CUE_RE = re.compile(
    r"(?:\be\.g\.|\bfor (?:example|instance)\b|\bsuch as\b|\bsample\b|"
    r"\btemplates?\b|\bexamples?\b|\bwording\b|\bphras(?:e|ed|ing)\b|"
    r"\billustrat\w+\b|\bnotification\b|\bdisclaimer\b|\bnotice\b|"
    r"\b(?:could|might|may|would|should)\s+(?:read|say|state|include|look like|be worded)\b|"
    r"\blike\b)"
    r"[^\n]{0,80}$",
    re.IGNORECASE,
)

# Content that marks the quote itself as drafted/addressed prose rather than
# provision text: second-person address, first-person-plural framing, tildes
# and standalone "X" placeholders ("lack of experience in X" — but not
# "X-rays", "Annex X" or "Article X", where X is real).
_ILLUSTRATIVE_CONTENT_RE = re.compile(
    r"\b(?:you|your|we|our)\b|~|(?<!Annex )(?<!Article )\bX\b(?![-\w])",
)


def _is_illustrative(answer: str, q: "Quote") -> bool:
    """True when an ungrounded quote is the model's own illustrative wording.

    A trailing citation (``“…” (Article 12)``) is an unambiguous legal-quote
    claim and vetoes the tier outright. A *leading* label only vetoes when no
    illustrative cue sits closer to the quote: in "your Article 26
    notification could state: '…'" the article reference belongs to the
    illustrative framing, not to a quote attribution.
    """
    after = answer[q.end : q.end + _ATTRIBUTION_AFTER_WINDOW]
    newline = after.find("\n")
    if newline != -1:
        after = after[:newline]
    if _ATTRIBUTION_LABEL_RE.search(after):
        return False  # trailing citation → a legal-quote claim
    lead = answer[max(0, q.start - 100) : q.start]
    cue_matches = list(_ILLUSTRATIVE_CUE_RE.finditer(lead))
    label_matches = list(_ATTRIBUTION_LABEL_RE.finditer(lead))
    cue_end = cue_matches[-1].end() if cue_matches else -1
    label_end = label_matches[-1].end() if label_matches else -1
    if cue_end >= 0 and cue_end >= label_end:
        return True
    if label_end >= 0:
        return False  # leading attribution with no closer cue
    return bool(_ILLUSTRATIVE_CONTENT_RE.search(q.text))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaithfulnessReport:
    """Result of verifying every quote in an answer against the corpus."""

    total_quotes: int
    verified: list[Quote] = field(default_factory=list)
    unverified: list[Quote] = field(default_factory=list)
    near_verbatim: list[Quote] = field(default_factory=list)
    misattributed: list[Quote] = field(default_factory=list)
    # The model's own drafted wording (templates, scenario echoes) — kept in
    # the answer, never counted as fabrication (see _is_illustrative).
    illustrative: list[Quote] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        # Near-verbatim quotes are grounded (kept, not redacted); only genuinely
        # divergent quotes (fabricated or displaced/concatenated) make a report
        # not-ok, since both are removed from the answer.
        return not self.unverified and not self.misattributed

    @property
    def removed(self) -> list[Quote]:
        """Every quote stripped from the answer (fabricated + displaced)."""
        return self.unverified + self.misattributed

    @property
    def verified_count(self) -> int:
        return len(self.verified)

    @property
    def unverified_count(self) -> int:
        return len(self.unverified)

    @property
    def near_verbatim_count(self) -> int:
        return len(self.near_verbatim)

    @property
    def misattributed_count(self) -> int:
        return len(self.misattributed)

    @property
    def illustrative_count(self) -> int:
        return len(self.illustrative)


def check_faithfulness(
    answer: str,
    provisions: list[dict[str, Any]],
    definitions: list[dict[str, Any]] | None = None,
    question: str | None = None,
) -> FaithfulnessReport:
    """Verify every long verbatim quote in *answer* against the full context.

    The corpus spans both retrieved *provisions* and the *definitions* block so
    that quotes drawn from formal definitions or interpretive guidance verify
    correctly.  Returns a :class:`FaithfulnessReport`.  Quotes below the length
    threshold are silently dropped (not counted as verified or unverified).

    When *question* is provided, quotes grounded in the question itself are
    exempt: the model restating the user's scenario in quote marks ("assists
    radiologists in detecting lung nodules") is not a claim about legal text,
    and flagging it as a fabricated regulatory quote was a false positive.
    """
    quotes = extract_quotes(answer)
    if not quotes:
        return FaithfulnessReport(total_quotes=0)
    corpus = _build_corpus(provisions, definitions)
    _sources, source_map, unkeyed = _build_sources(provisions, definitions)
    question_norm = _normalize(question) if question else ""
    verified: list[Quote] = []
    near_verbatim: list[Quote] = []
    unverified: list[Quote] = []
    misattributed: list[Quote] = []
    illustrative: list[Quote] = []
    for q in quotes:
        if question_norm and grounding_verdict(q.text, question_norm) != "absent":
            verified.append(q)   # scenario echo, not a regulatory quote
            continue
        verdict = grounding_verdict(q.text, corpus)
        if verdict == "absent":
            # Uncited drafted wording (templates, embellished scenario echoes)
            # grounds nowhere by construction — it is not a legal-quote claim
            # and must not be redacted as fabrication (see _is_illustrative).
            if _is_illustrative(answer, q):
                illustrative.append(q)
                continue
            # Genuine fabrication: not grounded anywhere.  Structural guards
            # only apply to text that *is* real but displaced, so skip them.
            unverified.append(q)
            continue
        # Grounded somewhere — but is it grounded where the answer claims, and
        # is it a single quotation rather than a concatenated dump?
        if _structural_verdict(q, answer, source_map, unkeyed):
            misattributed.append(q)
        elif verdict == "exact":
            verified.append(q)
        else:
            near_verbatim.append(q)
    return FaithfulnessReport(
        total_quotes=len(quotes),
        verified=verified,
        unverified=unverified,
        near_verbatim=near_verbatim,
        misattributed=misattributed,
        illustrative=illustrative,
    )


def build_warning_block(report: FaithfulnessReport) -> str | None:
    """Return a markdown warning block, or None when nothing needs surfacing.

    Two tiers, so a near-verbatim quote (kept in the answer) is not lumped in
    with a fabrication (removed from the answer):

    - **Unverified** quotes could not be grounded and have been *removed* \u2014 a
      loud flag listing each.
    - **Near-verbatim** quotes are grounded but differ in exact wording from the
      source \u2014 a light note prompting a wording check.  They remain in the text.
    """
    if report.ok and not report.near_verbatim:
        return None
    lines: list[str] = []
    if report.unverified:
        lines.append(
            "> \u26a0 **FAITHFULNESS FLAG** \u2014 "
            f"{report.unverified_count} of {report.total_quotes} verbatim quote(s) "
            "could not be matched against the retrieved regulatory context and "
            "have been removed from the answer. Treat the underlying point as "
            "unverified until checked against the source provisions."
        )
        for q in report.unverified:
            lines.append(f"> - \u201C{q.preview}\u201D")
    if report.misattributed:
        if lines:
            lines.append(">")
        lines.append(
            "> \u26a0 **ATTRIBUTION FLAG** \u2014 "
            f"{report.misattributed_count} of {report.total_quotes} quote(s) "
            "contain real regulatory text that does not belong to the provision "
            "they cite (or concatenate text from several provisions under one "
            "citation) and have been removed. Re-check which provision each "
            "obligation actually comes from."
        )
        for q in report.misattributed:
            lines.append(f"> - \u201C{q.preview}\u201D")
    if report.near_verbatim:
        if lines:
            lines.append(">")
        lines.append(
            "> \u2139\uFE0F **Wording check** \u2014 "
            f"{report.near_verbatim_count} quote(s) are near-verbatim (grounded "
            "in the source but with minor wording differences). Verify exact "
            "wording before quoting them externally."
        )
        for q in report.near_verbatim:
            lines.append(f"> - \u201C{q.preview}\u201D")
    return "\n".join(lines)


def remove_unverified_quotes(answer: str, report: FaithfulnessReport) -> str:
    """Remove fabricated and misattributed quote spans from *answer*.

    This enforces that neither fabricated verbatim quotations nor real-but-
    displaced/concatenated quotes survive in the user-facing output. Redaction
    is done by character offsets captured at extraction time.
    """
    removed = report.removed
    if not removed:
        return answer

    redacted = answer
    for q in sorted(removed, key=lambda x: x.start, reverse=True):
        if 0 <= q.start < q.end <= len(redacted):
            # Replace with a marker rather than deleting: an empty splice fuses
            # the words on either side of the removed quote ("...AI Act" +
            # "Actor role" -> "AI ActActor role"). The marker preserves the
            # sentence boundary; the warning block lists what was removed.
            redacted = redacted[:q.start] + _REDACTION_MARKER + redacted[q.end:]

    # Basic cleanup for common artifacts left by quote removal.
    redacted = re.sub(r"[ \t]{2,}", " ", redacted)
    redacted = re.sub(r"\n{3,}", "\n\n", redacted)
    redacted = re.sub(r"\*\*\s*\*\*", "", redacted)
    return redacted.strip()


# ---------------------------------------------------------------------------
# Quote repair
#
# Redaction throws away information the verification already computed: for a
# *misattributed* quote the adjudication found which provision the text really
# lives in, and for a *near-verbatim* (and many "absent") quotes the matcher
# located the true source span. Repair uses that knowledge deterministically —
# substitute the source's exact words, re-point the citation to the true
# provision — so a paraphrase-as-quotation becomes a correct answer instead of
# a hole plus a warning banner. No LLM call; unrepairable quotes still redact.
# ---------------------------------------------------------------------------

# A fabricated (absent-verdict) quote is repaired only from the provision it
# *cites*, and only when the model's text is recognisably a paraphrase of a
# specific passage there (SequenceMatcher ratio over the best 1-3 sentence
# run). Below this, substituting "the real text" risks planting a passage that
# does not support the surrounding claim.
_REPAIR_FABRICATED_MIN_RATIO: float = 0.60
# Near-verbatim quotes are already ≥90%-recall grounded; the sentence-run match
# must be strong before we overwrite the model's wording with the source's.
_REPAIR_NEAR_MIN_RATIO: float = 0.80
_REPAIR_MAX_SENTENCE_RUN: int = 3

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+|\n+")


def _best_sentence_run(quote_text: str, raw_source: str) -> tuple[float, str | None]:
    """Best-matching run of 1-3 consecutive source sentences for *quote_text*.

    Returns ``(ratio, raw_run)`` where *raw_run* preserves the source's exact
    casing/wording (single-line, inner double quotes folded) — the string that
    can replace the model's paraphrase inside the answer's quote marks.
    """
    quote_norm = _normalize(quote_text)
    if not quote_norm:
        return 0.0, None
    raw_sents = [s.strip() for s in _SENTENCE_SPLIT_RE.split(raw_source) if s.strip()]
    norm_sents = [_normalize(s) for s in raw_sents]
    best_ratio, best_run = 0.0, None
    for i in range(len(raw_sents)):
        acc = ""
        for j in range(i, min(i + _REPAIR_MAX_SENTENCE_RUN, len(raw_sents))):
            acc = norm_sents[j] if j == i else f"{acc} {norm_sents[j]}"
            if len(acc) > 3 * len(quote_norm) + 40:
                break
            m = difflib.SequenceMatcher(None, quote_norm, acc, autojunk=False)
            if m.real_quick_ratio() <= best_ratio:
                continue
            ratio = m.ratio()
            if ratio > best_ratio:
                raw_run = " ".join(raw_sents[i : j + 1])
                best_ratio, best_run = ratio, raw_run
    if best_run is not None:
        # Quote bodies are single-line and must not contain double-quote glyphs
        # (they would terminate the enclosing quotation).
        best_run = _WHITESPACE_RE.sub(" ", best_run).replace('"', "'").strip()
    return best_ratio, best_run


def _build_raw_sources(
    provisions: list[dict[str, Any]],
    definitions: list[dict[str, Any]] | None,
) -> dict[str, tuple[str, str]]:
    """Return ``{normalized_ref: (pretty_ref, raw_text)}`` for repair lookups."""
    out: dict[str, tuple[str, str]] = {}
    for item, text_of in [
        *((p, _provision_text) for p in provisions or []),
        *((d, _definition_text) for d in definitions or []),
    ]:
        if not item:
            continue
        raw_ref = item.get("article_ref")
        if not (isinstance(raw_ref, str) and _CITATION_REF_RE.search(raw_ref)):
            continue
        key = _normalize_ref(raw_ref)
        text = text_of(item)
        if not text:
            continue
        if key in out:
            out[key] = (out[key][0], out[key][1] + "\n" + text)
        else:
            out[key] = (raw_ref, text)
    return out


def _unique_grounding_ref(
    quote_text: str,
    source_map: dict[str, str],
) -> str | None:
    """The single normalized ref whose source grounds *quote_text*, if unique.

    Candidates collapse by base provision family: "Article 43", "Article
    43(4)" and "Article 43(4), subparagraph 2" all grounding a quote is one
    provision seen at three depths, not an ambiguity. Within the surviving
    family the most specific (longest) ref wins, so re-pointing lands on the
    precise provision. More than one distinct family → None (a genuinely
    ambiguous quote must not be re-attributed by guess).
    """
    candidates = [
        ref for ref, src in source_map.items()
        if grounding_verdict(quote_text, src) != "absent"
    ]
    if not candidates:
        return None
    families = {_base_ref_family(ref) for ref in candidates}
    if len(families) != 1:
        return None
    return max(candidates, key=len)


def _citation_ref_span(
    answer: str, quote_start: int, quote_end: int | None = None
) -> tuple[int, int] | None:
    """Span of the citation label a repoint must edit for a quote.

    Mirrors :func:`_nearest_citation_ref`'s selection exactly — last label in
    the before-window vs the same-line cite-after label, nearest to the quote
    wins — so the repair re-points the very label the checker adjudicated
    against. Before this mirrored the checker, a cite-after misattribution
    (``“…” (Article 10)``) was flaggable but structurally unfixable: both
    repair tiers searched only the before-window, so the offender was redacted
    on every run (HQ_006's chronic pattern).
    """
    lo = max(0, quote_start - _ATTRIBUTION_WINDOW)
    window = answer[lo:quote_start]
    # (distance-to-quote, label match, absolute offset of the searched string)
    best: tuple[int, "re.Match[str]", int] | None = None
    matches = list(_ATTRIBUTION_LABEL_RE.finditer(window))
    if matches:
        last = matches[-1]
        best = (len(window) - last.end(), last, lo)
    if quote_end is not None:
        after = answer[quote_end : quote_end + _ATTRIBUTION_AFTER_WINDOW]
        newline = after.find("\n")
        if newline != -1:
            after = after[:newline]
        trailing = _ATTRIBUTION_LABEL_RE.search(after)
        if trailing and (best is None or trailing.start() < best[0]):
            best = (trailing.start(), trailing, quote_end)
    if best is None:
        return None
    _dist, label, base = best
    if label.group("blocker"):
        return None   # non-adjudicable label — the checker never flagged via it
    head = _CITATION_REF_RE.search(label.group("cite"))
    if not head:
        return None
    return (
        base + label.start("cite") + head.start(1),
        base + label.start("cite") + head.end(1),
    )


def repair_and_redact(
    answer: str,
    report: FaithfulnessReport,
    provisions: list[dict[str, Any]],
    definitions: list[dict[str, Any]] | None = None,
    *,
    redact_residuals: bool = True,
) -> tuple[str, FaithfulnessReport, list[str]]:
    """Repair what verification already solved; redact only the remainder.

    With ``redact_residuals=False`` the deterministic repairs are applied but
    unrepairable offenders are left *in place* (their offsets in the residual
    report may be stale after edits — re-run :func:`check_faithfulness` on the
    returned text for fresh spans). This is the strict-mode (mode 2) entry: the
    LLM repair tier gets one shot at the residuals before final redaction.

    Per offending quote, in one descending-offset pass (so earlier edits never
    invalidate later offsets):

    - **misattributed** (single true source, not a concatenated dump): re-point
      the nearest citation ref to the true provision; if the wording is only
      near-verbatim there, also substitute the source's exact sentence run.
    - **fabricated**: substitute the cited provision's best sentence run when
      the model's text is recognisably a paraphrase of it; otherwise redact.
    - **near-verbatim**: substitute the exact source wording when the match is
      strong; otherwise keep as-is (grounded either way).

    Returns ``(answer, residual_report, repair_notes)``; the residual report
    holds only the quotes still removed/noted, for the warning block. Callers
    keep feeding the *original* report to confidence — a repaired fabrication
    still reflects generation behaviour.
    """
    _sources, source_map, unkeyed = _build_sources(provisions, definitions)
    raw_sources = _build_raw_sources(provisions, definitions)

    # action: (quote, kind, payload)
    actions: list[tuple[Quote, str, dict[str, Any]]] = []
    residual_unverified: list[Quote] = []
    residual_misattributed: list[Quote] = []
    residual_near: list[Quote] = []
    notes: list[str] = []

    def _run_for(quote: Quote, ref: str | None, min_ratio: float) -> str | None:
        if ref is None or ref not in raw_sources:
            # Walk up the parent chain like attribution resolution does.
            for parent in _article_ref_parent_chain(ref or ""):
                if parent in raw_sources:
                    ref = parent
                    break
            else:
                return None
        ratio, run = _best_sentence_run(quote.text, raw_sources[ref][1])
        if run and ratio >= min_ratio and len(run) >= _MIN_QUOTE_LEN:
            return run
        return None

    for q in report.unverified:
        if _ELLIPSIS_RE.search(q.text):
            residual_unverified.append(q)   # multi-fragment: not repairable
            continue
        cited = _nearest_citation_ref(answer, q.start, q.end)
        run = _run_for(q, cited, _REPAIR_FABRICATED_MIN_RATIO)
        if run:
            actions.append((q, "substitute", {"text": run}))
            notes.append(f"quote corrected to the exact text of {raw_sources.get(cited, (cited,))[0] if cited else 'its source'}")
        else:
            residual_unverified.append(q)

    for q in report.misattributed:
        quote_norm = _normalize(q.text)
        if (
            len(quote_norm) >= _CONCAT_MIN_QUOTE
            and _distinct_source_count(quote_norm, source_map, unkeyed) > _CONCAT_MAX_SOURCES
        ):
            residual_misattributed.append(q)   # dump, not a quotation
            continue
        true_ref = _unique_grounding_ref(q.text, source_map)
        ref_span = _citation_ref_span(answer, q.start, q.end)
        if true_ref and true_ref in raw_sources and ref_span:
            payload: dict[str, Any] = {
                "ref_span": ref_span,
                "ref_text": raw_sources[true_ref][0],
            }
            if grounding_verdict(q.text, source_map[true_ref]) == "near":
                run = _run_for(q, true_ref, _REPAIR_NEAR_MIN_RATIO)
                if run:
                    payload["text"] = run
            actions.append((q, "repoint", payload))
            notes.append(
                f"citation corrected: the quoted text is from {raw_sources[true_ref][0]}"
            )
        else:
            residual_misattributed.append(q)

    for q in report.near_verbatim:
        cited = _nearest_citation_ref(answer, q.start, q.end)
        run = _run_for(q, cited, _REPAIR_NEAR_MIN_RATIO)
        if run:
            actions.append((q, "substitute", {"text": run}))
            notes.append(
                f"wording aligned to the exact text of {raw_sources.get(cited, (cited,))[0] if cited else 'its source'}"
            )
        else:
            residual_near.append(q)

    # Apply repairs + residual redactions in one descending-offset pass.
    edits: list[tuple[int, int, str]] = []   # (start, end, replacement)
    for q, kind, payload in actions:
        if "text" in payload:
            edits.append((q.start, q.end, "“" + payload["text"] + "”"))
        if kind == "repoint":
            lo, hi = payload["ref_span"]
            edits.append((lo, hi, payload["ref_text"]))
    if redact_residuals:
        for q in residual_unverified + residual_misattributed:
            edits.append((q.start, q.end, _REDACTION_MARKER))

    repaired = answer
    for start, end, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        if 0 <= start < end <= len(repaired):
            repaired = repaired[:start] + replacement + repaired[end:]

    repaired = re.sub(r"[ \t]{2,}", " ", repaired)
    repaired = re.sub(r"\n{3,}", "\n\n", repaired)
    repaired = re.sub(r"\*\*\s*\*\*", "", repaired).strip()

    residual = FaithfulnessReport(
        total_quotes=report.total_quotes,
        verified=list(report.verified) + [q for q, _, _ in actions],
        unverified=residual_unverified,
        near_verbatim=residual_near,
        misattributed=residual_misattributed,
        illustrative=list(report.illustrative),
    )
    return repaired, residual, notes


def build_repair_note(notes: list[str]) -> str | None:
    """Small info block listing deterministic quote repairs (or None)."""
    if not notes:
        return None
    lines = [
        "> \U0001F527 **Auto-verified corrections** — "
        f"{len(notes)} quote(s)/citation(s) were corrected against the "
        "retrieved source text:"
    ]
    lines.extend(f"> - {n}" for n in notes)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mode helpers
# ---------------------------------------------------------------------------


def faithfulness_mode(value: str | None) -> int:
    """Parse the ``CRSS_FAITHFULNESS_CHECK`` env value to an integer mode.

    Returns 0 (off), 1 (flag), or 2 (strict).  Unknown values fall back to 0.
    Mode 2 (strict) adds the LLM-assisted repair tier for offenders the
    deterministic repair cannot fix (see ``application/_faithfulness_repair``);
    the integration lives in ``application/verify.py``.
    """
    if value is None:
        return 0
    value = value.strip().lower()
    if value in {"", "0", "off", "false", "no"}:
        return 0
    if value in {"1", "flag", "true", "yes", "on"}:
        return 1
    if value in {"2", "strict"}:
        return 2
    return 0
