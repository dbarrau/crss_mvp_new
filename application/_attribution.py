"""Guidance-attribution guard — deterministic correction of guidance-document
references in a generated answer.

Why this exists (verified, not assumed): the model reliably fabricates a guidance
*document* title by blending the document-title prefix with the *section* it
quoted. The "Exclusions from the scope of the AI Act" section of the
**AI Office Guidelines on Transparency (Art. 50 AI Act)** is rendered as
"AI Office Guidelines on Exclusions from the Scope of the AI Act". Prompt
instructions and an explicit "Source document:" line in the context do NOT hold
against this — reproduced across a server restart with the correct title present
three separate ways in context. Attribution is a core value proposition, so it is
guarded deterministically here (same philosophy as the faithfulness/phantom
guards), not nudged.

The guard is conservative: it only rewrites an AI Office title when (a) exactly
one AI Office guidance document was retrieved (so the correction is unambiguous)
and (b) the fabricated tail actually matches a retrieved *section* name (so it is
provably a section→document conflation, not a legitimate phrase). It also strips
machinery leaks — "(referenced in the context)" and the broader family of short
parentheticals ending in "in the context)" (e.g. "(interpretive links in the
context)").
"""
from __future__ import annotations

import re

from application._context import _celex_is_guidance

# Machinery leak: "(referenced in the context)" / "referenced in the context".
# Deliberately tight on "referenced in ... context" so it never touches the
# legitimate legal phrase "in the context of Article 50 AI Act".
_LEAK_RE = re.compile(
    r"\s*\(?\b(?:as\s+)?referenced\s+in\s+(?:the\s+)?context\b\)?[.,]?",
    re.IGNORECASE,
)

# Broader machinery leak: any short parenthetical that ENDS in "in the context)"
# — e.g. "(interpretive links in the context)", "(as provided in context)". The
# closing paren must come immediately after "context", which is what separates
# the leak from the legitimate phrase "(… in the context of Article 50)" (there
# "context" is followed by " of", so the paren never closes on it). Bounded to a
# single parenthetical (no nested parens / newlines) so it stays local.
_LEAK_PAREN_RE = re.compile(
    r"\s*\([^()\n]*?\bin\s+(?:the\s+)?context\)",
    re.IGNORECASE,
)

# An "AI Office Guidelines on <title>" mention, ending at an "AI Act" boundary
# (either a "(… AI Act)" parenthetical, as in the real title, or a trailing
# "… the AI Act", as in the fabrication). Tail length is capped to bound the
# match; a section cross-check (below) is what actually authorises a rewrite.
_AI_OFFICE_TITLE_RE = re.compile(
    r"(?:[Tt]he\s+)?AI Office Guidelines?\s+on\s+.{1,55}?"
    r"(?:\([^)\n]*AI Act\)|(?:the\s+)?AI Act)"
)


def _norm(s: str) -> str:
    """Lowercase, fold punctuation to spaces, collapse whitespace."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _ai_office_docs(provisions: list[dict]) -> dict[str, str]:
    """{celex: document-title} for retrieved AI Office guidance documents."""
    out: dict[str, str] = {}
    for p in provisions:
        celex = p.get("celex", "") or ""
        if celex.startswith("AI_OFFICE_") and _celex_is_guidance(celex):
            title = (p.get("regulation") or "").strip()
            if title:
                out[celex] = title
    return out


def _ai_office_section_to_docs(provisions: list[dict]) -> dict[str, set[str]]:
    """Map normalised section name → set of owning document titles.

    Sourced from each AI Office provision's ``article_ref`` and the segments of
    its ``article_path`` (excluding the doc-title segment). These are exactly the
    strings the model conflates into a fabricated document title, so resolving a
    fabricated tail *through its owning section* pins the correct document even
    when several AI Office documents were retrieved. A section owned by more than
    one document is ambiguous and is not used for correction.
    """
    mapping: dict[str, set[str]] = {}
    for p in provisions:
        celex = p.get("celex", "") or ""
        if not (celex.startswith("AI_OFFICE_") and _celex_is_guidance(celex)):
            continue
        title = (p.get("regulation") or "").strip()
        if not title:
            continue
        names: list[str] = []
        ref = (p.get("article_ref") or "").strip()
        if ref:
            names.append(ref)
        for seg in (p.get("article_path") or "").split(" / "):
            seg = seg.strip()
            if seg and seg != title:
                names.append(seg)
        for nm in names:
            n = _norm(nm)
            if len(n) >= 8:
                mapping.setdefault(n, set()).add(title)
    return mapping


def normalize_guidance_attribution(
    answer: str, provisions: list[dict]
) -> tuple[str, list[str]]:
    """Return ``(corrected_answer, changes)``.

    ``changes`` is a list of human-readable descriptions (empty when the answer
    was untouched) for logging.
    """
    changes: list[str] = []
    if not answer:
        return answer, changes

    # 1. Strip machinery-leak phrases.
    answer, n = _LEAK_RE.subn("", answer)
    answer, n2 = _LEAK_PAREN_RE.subn("", answer)
    if n + n2:
        changes.append(f"stripped {n + n2} machinery-leak phrase(s)")

    # 2. Correct fabricated AI Office document titles by resolving the fabricated
    #    tail through its owning section — precise even with several AI Office
    #    docs retrieved.
    sec_to_docs = _ai_office_section_to_docs(provisions)
    if sec_to_docs:
        real_title_norms = {_norm(t) for t in _ai_office_docs(provisions).values()}

        def _fix(m: re.Match) -> str:
            found = m.group(0)
            the_m = re.match(r"(?i)^the\s+", found)
            lead = found[: the_m.end()] if the_m else ""
            core = found[len(lead):].strip()
            if _norm(core) in real_title_norms:
                return found  # already a real document title — leave as-is
            tail = re.sub(r"(?i)^AI Office Guidelines?\s+on\s+", "", core).strip()
            tail_n = _norm(tail)
            # Resolve the owning document(s) for this tail-as-section. Authorise
            # a rewrite ONLY when exactly one document owns the matched section —
            # never for an arbitrary clause that happens to end in "AI Act".
            owners: set[str] = set()
            for sec_n, titles in sec_to_docs.items():
                if tail_n == sec_n or (len(tail_n) >= 8 and (tail_n in sec_n or sec_n in tail_n)):
                    owners |= titles
            if len(owners) != 1:
                return found
            correct = next(iter(owners))
            changes.append(f"corrected fabricated guidance title {core!r} -> {correct!r}")
            return f"{lead}{correct}"

        answer = _AI_OFFICE_TITLE_RE.sub(_fix, answer)

    return answer, changes
