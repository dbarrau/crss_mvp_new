"""Curated superseded → current AI Act application dates (Digital Omnibus).

Regulation (EU) 2026/1744 (the "Digital Omnibus on AI") **delayed** the AI Act's
high-risk application dates. The consolidated graph already holds the *current*
dates (Article 113(3)(c), tagged ``amended_by``), but a model trained before the
Omnibus confidently states the *original* dates from parametric memory — the
pre-Omnibus "August 2026/2027" dates were reported everywhere — and overrides the
correct dates sitting in its context (verified: it even misattributes the stale
date to Article 113, which says the opposite).

This is the bounded, authoritative record of what moved, consumed by the
date-currency guard (``application/_date_currency.py``). It is deliberately narrow:
ONLY the high-risk application dates the Omnibus changed. A superseded date is
corrected **only** when the answer asserts it in the matching high-risk scope —
crucially, ``2 August 2026`` is *also* the (unchanged, correct) general
application date of the AI Act, so it must never be touched outside a high-risk
(Annex III / Article 6(2)) context.

To extend when a future amendment moves another date: add an entry with its
superseded string, current string, the scope terms that identify the obligation
it governs, and the controlling provision. Keep it sourced from the amending act.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DateChange:
    """One superseded → current application-date change.

    ``scope_res`` are regex fragments (matched case-insensitively) that must
    appear in the same line as the superseded date for a correction to fire —
    they pin the date to the specific high-risk obligation it governs, so a
    legitimate use of the same date string elsewhere is never rewritten.
    ``scope_exclude_res`` are fragments that, if present, VETO the match (used to
    stop the Annex I rule firing on "Annex III").
    """

    superseded: str
    current: str
    provision: str
    scope_res: tuple[str, ...]
    scope_exclude_res: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


# The two high-risk dates the Digital Omnibus delayed. Sourced from AI Act
# Article 113(3)(c)(i)/(ii) as amended by Reg (EU) 2026/1744.
OMNIBUS_DATE_CHANGES: tuple[DateChange, ...] = (
    # Annex III (Article 6(2)) stand-alone high-risk: 2 Aug 2026 → 2 Dec 2027.
    DateChange(
        superseded="2 August 2026",
        current="2 December 2027",
        provision="Article 113(3)(c)(i)",
        scope_res=(r"annex\s+iii", r"6\s*\(\s*2\s*\)"),
        note="Annex III / Article 6(2) high-risk AI systems apply from "
             "2 December 2027 (Article 113(3)(c)(i), as amended by the Digital "
             "Omnibus, Regulation (EU) 2026/1744) — not the pre-Omnibus "
             "2 August 2026.",
    ),
    # Annex I (Article 6(1)) product-safety high-risk: 2 Aug 2027 → 2 Aug 2028.
    DateChange(
        superseded="2 August 2027",
        current="2 August 2028",
        provision="Article 113(3)(c)(ii)",
        # "safety component" is the defining marker of Annex I / Art 6(1) high-risk
        # (AI that is a safety component of a product), which the model often uses
        # in place of the literal "Annex I". Excluded scopes below keep it from
        # firing on the (unchanged) sandbox deadline or an Annex III line.
        scope_res=(r"annex\s+i\b", r"6\s*\(\s*1\s*\)", r"safety\s+component"),
        scope_exclude_res=(r"annex\s+iii", r"sandbox"),
        note="Annex I / Article 6(1) high-risk AI systems (safety components of "
             "products) apply from 2 August 2028 (Article 113(3)(c)(ii), as "
             "amended by the Digital Omnibus, Regulation (EU) 2026/1744) — not "
             "the pre-Omnibus 2 August 2027.",
    ),
)
