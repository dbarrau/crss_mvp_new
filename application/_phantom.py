"""Phantom-provision guard — strip citations to provisions that do not exist.

**The failure class** (observed 14 Jul 2026, CRSS self-classification answer):
the model cites provisions in *prose* — not inside verbatim quotes — that do
not exist in the cited regulation.  The dominant subclass is **draft-numbering
leakage**: the AI Act spent three years as a draft ("Title IA", Articles
4a–4c, the draft Article 28 value-chain clause), and that corpus dominates the
model's training data, so when retrieval leaves a gap the model fills it with
the *pre-final* numbering.  Same failure family as the jurisdiction guard
(law typed from training memory), and the same lesson applies: a prompt rule
does not hold; a deterministic strip does.

**Why quote guards cannot catch this**: the faithfulness/attribution checks
adjudicate verbatim *quotes*; a quote-free, memory-driven paragraph sails
through with ``fab=0 mis=0``.  This guard gives every answer at least
citation-existence verification.

**Mechanism** (closed-world, deterministic, no LLM): every ``Article N`` /
``Annex N`` / ``Recital N`` mention in the answer is resolved against the
whole-graph reference index (``GraphRetriever.reference_index()``), which
covers every provision of every ingested regulation.  A mention whose base
family exists in **no** in-corpus regulation is a phantom; its line is removed
and a distinct **PHANTOM CITATION FLAG** is prepended.

**Existence is checked against the graph, not a numbering heuristic**: the
consolidated MDR/IVDR genuinely contain a lettered ``Article 10a`` (EUDAMED
amendment), so "lettered articles are fake" would false-flag real law.

**Attribution scoping** (the can't-adjudicate-stay-silent discipline of the
misattribution guard, validated against 108 stored eval answers):

- the primary check is against the **corpus union**: a family that exists in
  no ingested regulation is phantom — unless the line also cites an
  **out-of-corpus act** ("Article 30 of Regulation (EU) 2019/1020", the TFEU,
  the Charter), in which case we hold no text to adjudicate against → silent;
- a family that exists somewhere in the corpus is additionally checked
  against an explicitly named act only when the act alias is **immediately
  adjacent** to the mention ("MDR Article 110a", "Recital 43 GDPR") — a
  loose nearest-name heuristic mis-scoped mentions to another act named
  later in the same line (observed: AI Act's Annex V scoped to GDPR because
  the line quoted a GDPR compliance statement);
- per-kind adjudication: an act is only adjudicated for a mention kind
  (article / annex / recital) it actually has families for in the graph.
  GDPR/MDR/IVDR recitals are not ingested, so "Recital 71 GDPR" — real law,
  absent from the graph — must stay silent, not flag (observed false
  positive).

**Boundary (stated, not hidden)**: this guard checks *existence*, not
*semantics*.  "Article 28(2)" cited with draft-era meaning passes, because
Article 28 exists in the final act — that class needs content-level checking,
which no existence test can provide.
"""
from __future__ import annotations

import logging
import re

from application._faithfulness import _base_ref_family

logger = logging.getLogger(__name__)

# Node ids of regulation provisions start with their CELEX number
# (e.g. "32017R0745_article_2"); guidance ids ("MDCG_2019_11_...") do not.
# Guidance display_refs ("Section 5.1") never match the citation grammar, so
# excluding them from the family index costs nothing and keeps attribution
# clean.
_CELEX_ID_RE = re.compile(r"^(3\d{4}[A-Z]\d{4})_")

# ---------------------------------------------------------------------------
# Mention grammar.  Handles singular/plural, lists and ranges:
#   "Article 6(3)", "Articles 9–15", "Articles 4a–4c", "Articles 111/113",
#   "Annexes I and III", "Recitals 44 to 46", "Art. 10a".
# Ranges are checked by their *endpoints* only (interior enumeration of
# lettered ranges is not well-defined; a phantom range reliably has a phantom
# endpoint).
# ---------------------------------------------------------------------------

_SEP = r"\s*(?:[,/]|[–—\-−]|\band\b|\bor\b|\bto\b|\bthrough\b)\s*"

_ARTICLE_MENTION_RE = re.compile(
    rf"\bArt(?:icle)?s?\.?\s+("
    rf"\d+[a-z]?(?:\(\d+[a-z]?\))?(?:\([a-z]+\))?"
    rf"(?:{_SEP}\d+[a-z]?(?:\(\d+[a-z]?\))?(?:\([a-z]+\))?)*"
    rf")",
    re.IGNORECASE,
)
_ANNEX_MENTION_RE = re.compile(
    rf"\bAnnex(?:es)?\s+([IVXLC]+\b(?:{_SEP}[IVXLC]+\b)*)",
    re.IGNORECASE,
)
_RECITAL_MENTION_RE = re.compile(
    rf"\bRecitals?\s+(\d+\b(?:{_SEP}\d+\b)*)",
    re.IGNORECASE,
)

# Tokenisers for the captured list (parenthetical depth is stripped first:
# paragraph/point existence is NOT checked — retrieval parses to varying
# depths and a missing depth node must never flag a real article).
_PAREN_RE = re.compile(r"\([^)]*\)")
_ART_TOKEN_RE = re.compile(r"\b\d+[a-z]?\b", re.IGNORECASE)
_ROMAN_TOKEN_RE = re.compile(r"\b[IVXLC]+\b", re.IGNORECASE)
_NUM_TOKEN_RE = re.compile(r"\b\d+\b")

# Markdown emphasis breaks mention spans ("**Articles 4a**–4c"); scan a
# stripped copy of each line, remove the original line.
_MD_MARKS_RE = re.compile(r"[*_`]")

# Out-of-corpus act citations near a mention → silent (cannot adjudicate).
# Numeric EU-act forms are resolved against the corpus first, so
# "Regulation (EU) 2016/679" attributes to GDPR rather than silencing.
_OTHER_ACT_RE = re.compile(
    r"(?:Regulation|Directive|Decision)\s*\((?:EU|EC|EEC|Euratom)\)"
    r"\s*(?:No\.?\s*)?(\d{2,4}/\d{2,4})"
    r"|(?:Regulation|Directive|Decision)\s+(?:No\.?\s*)?(\d{2,4}/\d{2,4})"
    r"|\bTFEU\b|\bTEU\b|\bCharter\s+of\s+Fundamental\s+Rights\b|\bthe\s+Charter\b",
    re.IGNORECASE,
)

# Curated short-name aliases for explicit attribution.  Numeric forms
# ("2024/1689") are derived from the CELEX automatically for *every* catalog
# regulation, so a newly ingested act is attributable by number without
# touching this table (see the reg-detection-pattern-gap lesson); the table
# only adds the human short names.
_SHORT_ALIASES: dict[str, tuple[str, ...]] = {
    "32024R1689": ("ai act", "artificial intelligence act"),
    "32016R0679": ("gdpr", "general data protection regulation"),
    "32017R0745": ("mdr", "medical device regulation", "medical devices regulation"),
    "32017R0746": ("ivdr", "in vitro diagnostic regulation"),
}

_CELEX_NUM_RE = re.compile(r"^3(\d{4})[A-Z](\d{4})$")


def _celex_numeric_form(celex: str) -> str | None:
    """``32024R1689`` → ``2024/1689`` (leading zeros of the number stripped)."""
    m = _CELEX_NUM_RE.match(celex)
    if not m:
        return None
    return f"{m.group(1)}/{int(m.group(2))}"


def build_provision_families(
    reference_index: dict[str, tuple[str, str]],
) -> dict[str, set[str]]:
    """``{celex: {lowercased base families}}`` from the whole-graph ref index.

    Families are ``_base_ref_family`` outputs ("article 10a", "annex iii",
    "recital 44"), so depth-qualified display_refs ("Article 10a(1)",
    "Annex VIII, Chapter III, point 6.1") collapse to their citable base.
    """
    families: dict[str, set[str]] = {}
    for node_id, (display_ref, _reg) in reference_index.items():
        m = _CELEX_ID_RE.match(node_id)
        if not m or not display_ref:
            continue
        fam = _base_ref_family(display_ref).lower()
        # Only citation-grammar families matter (skip "Chapter III" etc.)
        if fam.startswith(("article ", "annex ", "recital ")):
            families.setdefault(m.group(1), set()).add(fam)
    return families


# Maximum character gap between a mention span and an act alias span for the
# alias to *explicitly scope* the mention ("MDR Article 110a", "Recital 43
# GDPR").  Beyond this, an act named elsewhere in the line must not capture
# the mention — observed mis-scoping: a line about the AI Act's Annex V that
# quoted a GDPR compliance statement bound "Annex V" to GDPR and false-flagged.
_ADJACENCY_GAP = 25


def _attribution_candidates(
    scan_line: str,
    families_by_celex: dict[str, set[str]],
) -> list[tuple[int, int, str | None]]:
    """Return ``(start, end, celex_or_None)`` act references found in the line.

    ``celex`` set → an in-corpus act reference (scopes an adjacent mention).
    ``None`` → an out-of-corpus act (cannot adjudicate → silences the line's
    union misses).
    """
    low = scan_line.lower()
    candidates: list[tuple[int, int, str | None]] = []
    numeric_to_celex = {
        _celex_numeric_form(c): c
        for c in families_by_celex
        if _celex_numeric_form(c)
    }
    # Short-name + numeric aliases of in-corpus acts
    for celex in families_by_celex:
        aliases = list(_SHORT_ALIASES.get(celex, ()))
        num = _celex_numeric_form(celex)
        if num:
            aliases.append(num)
        for alias in aliases:
            for m in re.finditer(rf"\b{re.escape(alias)}\b", low):
                candidates.append((m.start(), m.end(), celex))
    # Generic EU-act citations; resolve their number against the corpus first
    for m in _OTHER_ACT_RE.finditer(scan_line):
        number = m.group(1) or m.group(2)
        celex = numeric_to_celex.get(number) if number else None
        candidates.append((m.start(), m.end(), celex))
    return candidates


def _span_gap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Character gap between two spans (0 when adjacent or overlapping)."""
    if a_end <= b_start:
        return b_start - a_end
    if b_end <= a_start:
        return a_start - b_end
    return 0


def _mention_tokens(scan_line: str) -> list[tuple[int, int, str]]:
    """Extract ``(start, end, base_family)`` for every provision mention."""
    out: list[tuple[int, int, str]] = []
    for m in _ARTICLE_MENTION_RE.finditer(scan_line):
        listing = _PAREN_RE.sub(" ", m.group(1))
        for tok in _ART_TOKEN_RE.findall(listing):
            out.append((m.start(), m.end(), f"article {tok.lower()}"))
    for m in _ANNEX_MENTION_RE.finditer(scan_line):
        for tok in _ROMAN_TOKEN_RE.findall(m.group(1)):
            out.append((m.start(), m.end(), f"annex {tok.lower()}"))
    for m in _RECITAL_MENTION_RE.finditer(scan_line):
        for tok in _NUM_TOKEN_RE.findall(m.group(1)):
            out.append((m.start(), m.end(), f"recital {tok}"))
    return out


def _kind(family: str) -> str:
    return family.split(" ", 1)[0]


def _line_phantoms(
    scan_line: str,
    families_by_celex: dict[str, set[str]],
    corpus_union: set[str],
) -> list[str]:
    """Return the phantom families cited in this line (empty when clean).

    Two-tier adjudication (see module docstring):

    1. Family absent from the whole corpus → phantom, unless the line also
       references an out-of-corpus act (cannot adjudicate → silent).
    2. Family present in the corpus → checked against a specific act only
       when that act's alias is immediately adjacent to the mention, and only
       for mention kinds the act has families for in the graph (per-kind
       adjudication: un-ingested recitals must not flag).
    """
    mentions = _mention_tokens(scan_line)
    if not mentions:
        return []
    candidates = _attribution_candidates(scan_line, families_by_celex)
    line_has_foreign_act = any(c[2] is None for c in candidates)
    # Kinds each act can adjudicate: only those it has families for.
    kinds_by_celex = {
        celex: {_kind(f) for f in fams}
        for celex, fams in families_by_celex.items()
    }

    phantoms: list[str] = []
    for m_start, m_end, family in mentions:
        if family not in corpus_union:
            # Exists nowhere in the corpus. Silent only when the line cites
            # an act we do not hold (the mention may belong to it).
            if not line_has_foreign_act:
                phantoms.append(family)
            continue
        # Exists somewhere: flag only an explicit, adjacent mis-scoping
        # ("MDR Article 110a" where MDR has no 110a).
        adjacent = [
            (c_start, c_end, celex)
            for c_start, c_end, celex in candidates
            if _span_gap(m_start, m_end, c_start, c_end) <= _ADJACENCY_GAP
        ]
        if not adjacent:
            continue
        _, _, celex = min(
            adjacent,
            key=lambda c: _span_gap(m_start, m_end, c[0], c[1]),
        )
        if celex is None:
            continue  # adjacent out-of-corpus act — cannot adjudicate
        if _kind(family) not in kinds_by_celex.get(celex, set()):
            continue  # act's families of this kind not ingested — silent
        if family not in families_by_celex[celex]:
            phantoms.append(family)
    return phantoms


_PHANTOM_WARNING = (
    "> ⚠ **PHANTOM CITATION FLAG** — {n} statement(s) citing provisions that "
    "do not exist in the corresponding regulation were removed ({refs}). Such "
    "references typically originate from pre-final drafts of the legislation "
    "in the model's training data (e.g. draft AI Act numbering) and cannot be "
    "verified against the official text."
)


def strip_phantom_citations(
    answer: str,
    reference_index: dict[str, tuple[str, str]],
) -> tuple[str, list[str]]:
    """Remove lines citing nonexistent provisions; return (answer, phantoms).

    Removal is line-grained, mirroring the jurisdiction guard: CRSS answers
    are line-structured markdown, and a line leaning on a provision that does
    not exist is memory-driven in its entirety — the citation is only the
    checkable symptom.  When lines are removed, a loud warning block naming
    the phantom references is prepended.

    Degrades to a no-op when the reference index is empty (e.g. retriever
    doubles in tests), so the guard can never fire without a graph to check
    against.
    """
    families_by_celex = build_provision_families(reference_index)
    if not families_by_celex:
        return answer, []
    corpus_union: set[str] = set().union(*families_by_celex.values())

    kept: list[str] = []
    removed_refs: list[str] = []
    for line in answer.splitlines():
        scan_line = _MD_MARKS_RE.sub("", line)
        phantoms = _line_phantoms(scan_line, families_by_celex, corpus_union)
        if phantoms:
            removed_refs.extend(phantoms)
            continue
        kept.append(line)

    if not removed_refs:
        return answer, []

    unique_refs = list(dict.fromkeys(removed_refs))

    def _display(fam: str) -> str:
        kind, _, num = fam.partition(" ")
        return f"{kind.capitalize()} {num.upper() if kind == 'annex' else num}"

    display = ", ".join(_display(r) for r in unique_refs[:8])
    warning = _PHANTOM_WARNING.format(n=len(removed_refs), refs=display)
    return f"{warning}\n\n" + "\n".join(kept), unique_refs
