"""Superseded-provision guard — flag citations to a DELETED provision.

**The failure class**: an amending act deletes a provision (the Digital Omnibus
deletes AI Act Article 10(5)); the consolidated graph loses the node, so current
law is correct — but the model, trained on the pre-amendment text, cites the
deleted number as if still in force ("providers may process special categories
under Article 10(5)"). No other guard catches it: the phantom guard is
article-grained (Article 10 exists → 10(5) passes) and the faithfulness check only
inspects verbatim quotes. Same family as the date-currency guard (a superseded
*date* the model states from memory), but for a superseded *provision*.

**Mechanism** (deterministic, no LLM): the deletions are recorded once, at
consolidation time, in ``domain/ontology/superseded_provisions.SUPERSEDED`` (see
``consolidation/superseded``). Every cited ``Article N(p)`` is resolved to the act
it belongs to — via the EUR-Lex link's CELEX or the nearest regulation name — and,
only when it resolves to the act that deleted it, flagged. The scoping is the
safety rail: MDR Article 10(5) is real law, so an "Article 10(5)" attributed to
the MDR (or unresolved) is never touched. The offending line is removed and a
precise flag names the deletion and, when known, the relocation
("… deleted by the Digital Omnibus; see Article 4a(1)").

**Not inferred from node absence**: a paragraph missing from the graph could be a
parser gap rather than a deletion (MDR/IVDR arrive pre-consolidated from EUR-Lex),
so only provisions an amendment is *recorded* as deleting are flagged — never a
guess. This is why the record is sourced from the consolidation, not the graph.
"""
from __future__ import annotations

import re

from domain.legislation_catalog import LEGISLATION
from domain.ontology.superseded_provisions import SUPERSEDED

# Detection is driven by the RECORD, not a hardcoded shape: every deleted ref is
# reduced to a citable "family" (an article-paragraph "article 10(5)", or a
# deleted annex point "annex i section a point 1") and matched against the same
# families parsed out of the answer. So any provision the consolidation records as
# deleted is guarded, whatever its shape — the guard stays coherent with the
# Omnibus as the record grows.
_ART_PARA = r"Article\s+(\d+[a-z]?)\((\d+[a-z]?)\)"
_ANNEX_POINT = r"Annex\s+([IVXLC]+)\s*,?\s*Section\s+([A-Za-z])\s*,?\s*point\s+(\d+[a-z]?)"
_ART_PARA_RE = re.compile(_ART_PARA, re.IGNORECASE)
_ANNEX_POINT_RE = re.compile(_ANNEX_POINT, re.IGNORECASE)


def _ref_family(ref: str) -> str | None:
    """Reduce a citation (recorded ref OR answer mention) to a comparable family."""
    m = _ART_PARA_RE.match(ref) or _ART_PARA_RE.search(ref)
    if m and ref.strip().lower().startswith("article"):
        return f"article {m.group(1).lower()}({m.group(2).lower()})"
    m = _ANNEX_POINT_RE.search(ref)
    if m and ref.strip().lower().startswith("annex"):
        return f"annex {m.group(1).lower()} section {m.group(2).lower()} point {m.group(3).lower()}"
    return None


_SUPERSEDED_BY_ACT: dict[str, dict[str, dict]] = {}
for _rec in SUPERSEDED:
    _fam = _ref_family(_rec.get("deleted_ref", ""))
    if _fam:
        _SUPERSEDED_BY_ACT.setdefault(_rec["celex"], {})[_fam] = _rec


def _mentions(line: str):
    """Yield ``(start, family)`` for every article-paragraph and annex-point
    citation in *line* — the two citable shapes a deletion can take."""
    for m in _ART_PARA_RE.finditer(line):
        yield m.start(), f"article {m.group(1).lower()}({m.group(2).lower()})"
    for m in _ANNEX_POINT_RE.finditer(line):
        yield (m.start(),
               f"annex {m.group(1).lower()} section {m.group(2).lower()} "
               f"point {m.group(3).lower()}")

# Act identifiers for scoping a bare "Article N(p)" to its regulation — acronym /
# number, never subject-matter concepts (see the link-disambiguation lesson). ALL
# four regs are recognised, not just the superseding one: a mention attributed to
# the MDR must be SEEN as MDR so it neither flags (MDR 10(5) is real) nor gets
# answer-promoted to the AI Act.
_ACT_ALIASES: dict[str, tuple[str, ...]] = {
    "32024R1689": ("ai act", "artificial intelligence act", "2024/1689"),
    "32017R0745": ("mdr", "medical device regulation", "medical devices regulation", "2017/745"),
    "32017R0746": ("ivdr", "in vitro diagnostic regulation", "2017/746"),
    "32016R0679": ("gdpr", "general data protection regulation", "2016/679"),
}
# Every CELEX an EUR-Lex link might carry (base OR consolidated source_celex) →
# the base CELEX the aliases/records are keyed by.
_CELEX_TO_BASE: dict[str, str] = {}
for _base in _ACT_ALIASES:
    _CELEX_TO_BASE[_base] = _base
    _src = LEGISLATION.get(_base, {}).get("source_celex")
    if _src:
        _CELEX_TO_BASE[_src] = _base

_LINK_CELEX_RE = re.compile(r"CELEX:([0-9A-Z]{5}[A-Z][0-9]{4}(?:-\d+)?)")


def _act_references(line: str) -> list[tuple[int, str]]:
    """``[(position, base_celex)]`` for every act reference in *line* — an EUR-Lex
    link CELEX (base or consolidated) or a regulation alias, across all four regs
    so a competing attribution is visible."""
    refs: list[tuple[int, str]] = []
    for m in _LINK_CELEX_RE.finditer(line):
        base = _CELEX_TO_BASE.get(m.group(1)) or _CELEX_TO_BASE.get(m.group(1).split("-")[0])
        if base:
            refs.append((m.start(), base))
    low = line.lower()
    for celex, aliases in _ACT_ALIASES.items():
        for alias in aliases:
            for m in re.finditer(rf"\b{re.escape(alias)}\b", low):
                refs.append((m.start(), celex))
    return refs


def _resolve_act(line: str, m_start: int, refs: list[tuple[int, str]]) -> str | None:
    """The act nearest the mention — the one that labels it. None when there is
    no act reference on the line (cannot attribute → must not flag)."""
    if not refs:
        return None
    return min(refs, key=lambda r: abs(r[0] - m_start))[1]


def _format(rec: dict) -> str:
    base = (f"**{rec['deleted_ref']}** was deleted by the {rec['amender_name']} "
            f"(Regulation (EU) {_num(rec['amender_celex'])}, Article 1 point "
            f"({rec['point_num']}))")
    if rec.get("see_ref"):
        # name 4a(1)'s role explicitly: it is the RELOCATED content, not where the
        # deletion is recorded (that is the point cited above) — a reader must not
        # read "see 4a(1)" as "the deletion lives there".
        return base + f"; its content now lives in **{rec['see_ref']}**."
    return base + " with no direct replacement."


def _num(celex: str) -> str:
    m = re.match(r"^3(\d{4})[A-Z](\d{4})$", celex)
    return f"{m.group(1)}/{int(m.group(2))}" if m else celex


_WARNING = (
    "> ⚠ **SUPERSEDED PROVISION FLAG** — {n} statement(s) cited a provision that "
    "current law no longer contains (deleted by amendment) and were removed:"
)


def strip_superseded_citations(answer: str) -> tuple[str, list[str]]:
    """Remove lines citing a recorded-deleted provision; return (answer, notes).

    A line whose ``Article N(p)`` resolves — by its EUR-Lex link or the nearest
    regulation name — to the act that *deleted* that paragraph is memory-driven in
    its entirety (it asserts superseded law), so the whole line is removed, exactly
    as the phantom guard does. A precise note per distinct provision is surfaced in
    the prepended flag. No-op when nothing matches.
    """
    if not _SUPERSEDED_BY_ACT:
        return answer, []
    lines = answer.splitlines()

    def _scannable(ln: str) -> bool:
        # skip headings and blockquotes (the amendment-provenance footer lives in
        # a blockquote and legitimately quotes "paragraph 5 is deleted")
        s = ln.lstrip()
        return not (s.startswith("#") or s.startswith(">"))

    # Pass 1 — line-level attribution of every mention, and the set of acts each
    # deleted family resolves to across the whole answer.
    per_line: list[list[tuple[str, str | None]]] = []
    fam_acts: dict[str, set[str]] = {}
    for ln in lines:
        fams: list[tuple[str, str | None]] = []
        if _scannable(ln):
            refs = _act_references(ln)
            for start, fam in _mentions(ln):
                act = _resolve_act(ln, start, refs)
                fams.append((fam, act))
                if act:
                    fam_acts.setdefault(fam, set()).add(act)
        per_line.append(fams)

    # A deleted family is answer-scoped to its superseding act only when it
    # resolves to that act somewhere and to NO other act — so an unattributed
    # straggler is promoted, but a family also cited for the MDR is not.
    answer_scoped: dict[str, dict] = {}
    for celex, fam_map in _SUPERSEDED_BY_ACT.items():
        for fam, rec in fam_map.items():
            if fam_acts.get(fam) == {celex}:
                answer_scoped[fam] = rec

    # Pass 2 — strip a line when any mention resolves (line-level or answer-level)
    # to a superseding act.
    kept: list[str] = []
    hit_recs: dict[str, dict] = {}
    for ln, fams in zip(lines, per_line):
        line_hit = False
        for fam, act in fams:
            rec = None
            if act and fam in _SUPERSEDED_BY_ACT.get(act, {}):
                rec = _SUPERSEDED_BY_ACT[act][fam]
            elif fam in answer_scoped:
                rec = answer_scoped[fam]
            if rec:
                hit_recs[rec["deleted_id"]] = rec
                line_hit = True
        if not line_hit:
            kept.append(ln)
    if not hit_recs:
        return answer, []
    notes = [_format(r) for r in hit_recs.values()]
    flag = _WARNING.format(n=len(hit_recs)) + "\n" + "\n".join(f"> - {n}" for n in notes)
    body = "\n".join(kept)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return f"{flag}\n\n{body}", notes
