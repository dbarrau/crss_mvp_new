"""Date-currency guard — correct superseded AI Act application dates.

The consolidated graph holds the *current* (Omnibus-delayed) high-risk dates, and
they reach the model's context — but a model trained before the Digital Omnibus
states the *original* dates from parametric memory and overrides the context
(verified: it asserts "Annex III high-risk applies from 2 August 2026" and even
attributes it to Article 113, which actually says 2 December 2027). A prompt rule
does not hold against a confident parametric date, so — like the jurisdiction and
phantom guards — this is corrected deterministically after generation.

It is deliberately conservative. A superseded date is rewritten only when, on the
same line: (a) the superseded string appears, (b) a scope term pins it to the
high-risk obligation it governs (Annex III / Article 6(2), or Annex I /
Article 6(1)), and (c) the current date is NOT already present (so "originally X,
now Y" phrasing is left intact). This matters because ``2 August 2026`` is *also*
the correct, unchanged general-application date of the AI Act — it must never be
touched outside a high-risk context. The change map lives in
``domain/ontology/omnibus_date_changes.py`` (single source of truth).

Corrections are surfaced, never silent: the caller appends a transparency note so
the reader sees which date was corrected and why.
"""
from __future__ import annotations

import re

from domain.ontology.omnibus_date_changes import OMNIBUS_DATE_CHANGES

_NBSP = "\u00a0"


def _date_pattern(date: str) -> re.Pattern[str]:
    """Match *date* allowing either a regular space or a non-breaking space
    between its tokens (EUR-Lex stores some dates with nbsp)."""
    return re.compile(re.escape(date).replace(r"\ ", "[ \u00a0]"), re.IGNORECASE)


# Precompile the rules once.
_RULES = [
    (
        ch,
        _date_pattern(ch.superseded),
        _date_pattern(ch.current),
        [re.compile(s, re.IGNORECASE) for s in ch.scope_res],
        [re.compile(s, re.IGNORECASE) for s in ch.scope_exclude_res],
    )
    for ch in OMNIBUS_DATE_CHANGES
]


def correct_superseded_dates(answer: str) -> tuple[str, list[str]]:
    """Return ``(corrected_answer, notes)``.

    ``notes`` is empty when nothing was corrected; otherwise one human-readable
    note per distinct provision corrected (for a transparency block + logging).
    """
    if not answer:
        return answer, []

    notes: list[str] = []
    fired: set[str] = set()
    out_lines: list[str] = []

    for line in answer.split("\n"):
        for ch, sup_re, cur_re, scope_res, exclude_res in _RULES:
            norm = line.replace(_NBSP, " ")
            if not sup_re.search(norm):
                continue
            if cur_re.search(norm):
                continue  # current date already stated -> "originally X, now Y"
            if not any(r.search(norm) for r in scope_res):
                continue  # not pinned to this high-risk obligation
            if any(r.search(norm) for r in exclude_res):
                continue  # excluded scope (e.g. Annex III for the Annex I rule)
            # Fire: rewrite the superseded date to the current one on this line
            # (nbsp folded to a regular space in passing).
            line = sup_re.sub(ch.current, norm)
            if ch.provision not in fired:
                fired.add(ch.provision)
                notes.append(ch.note)
        out_lines.append(line)

    return "\n".join(out_lines), notes
