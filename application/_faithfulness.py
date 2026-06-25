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
    return sources, by_ref


def _longest_match(a: str, b: str) -> int:
    """Length of the longest contiguous substring shared by *a* and *b*."""
    if not a or not b:
        return 0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b),
    ).size


def _distinct_source_count(quote_norm: str, sources: list[str]) -> int:
    """Count sources contributing a >= ``_CONCAT_BLOCK_MIN`` span to the quote."""
    count = 0
    for src in sources:
        if _longest_match(quote_norm, src) >= _CONCAT_BLOCK_MIN:
            count += 1
    return count


def _nearest_citation_ref(answer: str, quote_start: int) -> str | None:
    """Return the citation ref immediately preceding a quote, if any.

    Scans the ``_ATTRIBUTION_WINDOW`` characters before the quote for the last
    ``Article X`` / ``Annex Y`` reference — the label the quote is attributed to.
    """
    lo = max(0, quote_start - _ATTRIBUTION_WINDOW)
    window = answer[lo:quote_start]
    matches = list(_CITATION_REF_RE.finditer(window))
    if not matches:
        return None
    return _normalize_ref(matches[-1].group(1))


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
    sources: list[str],
    source_map: dict[str, str],
) -> str | None:
    """Return ``'concatenated'`` / ``'misattributed'`` / None for a grounded quote.

    Only meaningful for quotes already grounded somewhere in the corpus — these
    guards distinguish "real text, wrong place" from genuine fabrication (which
    the absent verdict already handles).
    """
    quote_norm = _normalize(quote.text)
    if (
        len(quote_norm) >= _CONCAT_MIN_QUOTE
        and _distinct_source_count(quote_norm, sources) > _CONCAT_MAX_SOURCES
    ):
        return "concatenated"
    cited = _nearest_citation_ref(answer, quote.start)
    if cited:
        src = _resolve_cited_source(cited, source_map)
        # Flag only when we actually hold the cited source and the quote is
        # absent from it — if the cited provision was never retrieved we cannot
        # adjudicate attribution, so we stay silent (no false flag).
        if src is not None and grounding_verdict(quote.text, src) == "absent":
            return "misattributed"
    return None


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


def check_faithfulness(
    answer: str,
    provisions: list[dict[str, Any]],
    definitions: list[dict[str, Any]] | None = None,
) -> FaithfulnessReport:
    """Verify every long verbatim quote in *answer* against the full context.

    The corpus spans both retrieved *provisions* and the *definitions* block so
    that quotes drawn from formal definitions or interpretive guidance verify
    correctly.  Returns a :class:`FaithfulnessReport`.  Quotes below the length
    threshold are silently dropped (not counted as verified or unverified).
    """
    quotes = extract_quotes(answer)
    if not quotes:
        return FaithfulnessReport(total_quotes=0)
    corpus = _build_corpus(provisions, definitions)
    sources, source_map = _build_sources(provisions, definitions)
    verified: list[Quote] = []
    near_verbatim: list[Quote] = []
    unverified: list[Quote] = []
    misattributed: list[Quote] = []
    for q in quotes:
        verdict = grounding_verdict(q.text, corpus)
        if verdict == "absent":
            # Genuine fabrication: not grounded anywhere.  Structural guards
            # only apply to text that *is* real but displaced, so skip them.
            unverified.append(q)
            continue
        # Grounded somewhere — but is it grounded where the answer claims, and
        # is it a single quotation rather than a concatenated dump?
        if _structural_verdict(q, answer, sources, source_map):
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
# Mode helpers
# ---------------------------------------------------------------------------


def faithfulness_mode(value: str | None) -> int:
    """Parse the ``CRSS_FAITHFULNESS_CHECK`` env value to an integer mode.

    Returns 0 (off), 1 (flag), or 2 (strict).  Unknown values fall back to 0.
    Mode 2 currently behaves as mode 1 at the integration layer.
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
