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

Enable via environment variable ``CRSS_FAITHFULNESS_CHECK``:
    "0" or unset  -> disabled (no-op)
    "1"           -> flag mode
    "2"           -> strict mode (currently equivalent to flag mode)

The check is fully deterministic.  No LLM calls.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

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
_QUOTE_PATTERN = re.compile(
    r"""
    (?:[\*_]+)?               # optional leading markdown emphasis
    (?P<open>["\u201C\u201D\u201E\u201F\u00AB\u00BB])
    (?P<body>[^"\u201C\u201D\u201E\u201F\u00AB\u00BB]{40,})
    (?P<close>["\u201C\u201D\u201E\u201F\u00AB\u00BB])
    (?:[\*_]+)?               # optional trailing markdown emphasis
    """,
    re.VERBOSE,
)


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
        if len(body) < _MIN_QUOTE_LEN:
            continue
        quotes.append(Quote(text=body, start=match.start(), end=match.end()))
    return quotes


# ---------------------------------------------------------------------------
# Normalization + verification
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_ELLIPSIS_RE = re.compile(r"\[\s*\.\.\.\s*\]|\.\.\.|\u2026")
_MD_EMPHASIS_RE = re.compile(r"([*_]{1,3})(?=\S)(.+?)(?<=\S)\1")
_CITATION_REF_RE = re.compile(
    r"\b(Article\s+\d+[a-z]?(?:\(\d+\))?(?:\([a-z]\))?|"
    r"Annex\s+[IVXLC]+|Recital\s+\d+)(?=\W|$)",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Normalize text for substring matching.

    Unicode NFKC normalization, lowercase, whitespace collapsed to single
    spaces.  NFKC collapses smart quotes / typographic dashes to their ASCII
    equivalents so quote-style differences between the model output and the
    retrieved corpus do not produce false negatives.
    """
    text = unicodedata.normalize("NFKC", text)
    # Strip markdown emphasis markers (e.g. **text**, *text*) so legal
    # quotes copied with formatting can still match verbatim source text.
    text = _MD_EMPHASIS_RE.sub(r"\2", text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _split_on_ellipsis(quote: str) -> list[str]:
    """Split *quote* on ellipsis markers, returning normalized fragments."""
    fragments = [frag.strip() for frag in _ELLIPSIS_RE.split(quote)]
    return [_normalize(frag) for frag in fragments if frag.strip()]


def verify_quote(quote_text: str, normalized_corpus: str) -> bool:
    """Return True iff *quote_text* is grounded in *normalized_corpus*.

    Splits the quote on ellipsis markers and requires every fragment of at
    least ``_MIN_QUOTE_LEN // 2`` characters to appear as a substring of the
    corpus.  Fragments below the threshold are skipped to avoid penalising
    short connector words flanking ellipses.
    """
    fragments = _split_on_ellipsis(quote_text)
    if not fragments:
        return False
    min_frag = max(_MIN_QUOTE_LEN // 2, 20)
    significant = [frag for frag in fragments if len(frag) >= min_frag]
    if not significant:
        significant = fragments
    return all(frag in normalized_corpus for frag in significant)


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------


def _provision_text(provision: dict[str, Any]) -> str:
    """Extract the verbatim text payload from a provision dict.

    Matches the fields used by ``application/_context.py`` so the corpus
    contains exactly what the LLM saw.
    """
    parts: list[str] = []
    body = provision.get("article_text") or provision.get("text") or ""
    if body:
        parts.append(str(body))
    for child in provision.get("children", []) or []:
        child_text = child.get("raw_text") or child.get("text") or ""
        if child_text:
            parts.append(str(child_text))
    return "\n".join(parts)


def _build_corpus(provisions: list[dict[str, Any]]) -> str:
    """Build a single normalized corpus string from all retrieved provisions."""
    raw = "\n".join(_provision_text(p) for p in provisions if p)
    return _normalize(raw)


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
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaithfulnessReport:
    """Result of verifying every quote in an answer against the corpus."""

    total_quotes: int
    verified: list[Quote] = field(default_factory=list)
    unverified: list[Quote] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unverified

    @property
    def verified_count(self) -> int:
        return len(self.verified)

    @property
    def unverified_count(self) -> int:
        return len(self.unverified)


def check_faithfulness(
    answer: str,
    provisions: list[dict[str, Any]],
) -> FaithfulnessReport:
    """Verify every long verbatim quote in *answer* against *provisions*.

    Returns a :class:`FaithfulnessReport`.  Quotes below the length threshold
    are silently dropped (not counted as verified or unverified).
    """
    quotes = extract_quotes(answer)
    if not quotes:
        return FaithfulnessReport(total_quotes=0)
    corpus = _build_corpus(provisions)
    verified: list[Quote] = []
    unverified: list[Quote] = []
    for q in quotes:
        if verify_quote(q.text, corpus):
            verified.append(q)
        else:
            unverified.append(q)
    return FaithfulnessReport(
        total_quotes=len(quotes),
        verified=verified,
        unverified=unverified,
    )


def build_warning_block(report: FaithfulnessReport) -> str | None:
    """Return a markdown warning block listing unverified quotes, or None.

    Intended to be prepended to the final answer so a senior reviewer sees
    the audit flag before reading the analysis.
    """
    if report.ok:
        return None
    lines = [
        "> \u26a0 **FAITHFULNESS FLAG** \u2014 "
        f"{report.unverified_count} of {report.total_quotes} verbatim quote(s) "
        "in this answer could not be matched against the retrieved regulatory "
        "context. Verify each flagged quote against the source provisions "
        "before relying on it.",
    ]
    for q in report.unverified:
        lines.append(f"> - \u201C{q.preview}\u201D")
    return "\n".join(lines)


def remove_unverified_quotes(answer: str, report: FaithfulnessReport) -> str:
    """Remove unverified quote spans from *answer*.

    This enforces that fabricated verbatim quotations do not survive in the
    user-facing output. Redaction is done by character offsets captured at
    extraction time.
    """
    if report.ok or not report.unverified:
        return answer

    redacted = answer
    for q in sorted(report.unverified, key=lambda x: x.start, reverse=True):
        if 0 <= q.start < q.end <= len(redacted):
            redacted = redacted[:q.start] + redacted[q.end:]

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
