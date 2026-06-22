"""Temporal applicability of the regulations (when obligations actually bite).

The graph models *what* each regulation requires but not *when* a requirement
becomes applicable. For compliance **readiness** that temporal dimension is
half the question: as of mid-2026 the EU AI Act is in force but only partly
*applicable* (prohibitions and GPAI rules apply; general high-risk obligations
do not yet), so an answer that treats every article as equally in force is
materially wrong.

This module is a small **curated** knowledge base of the key applicability
milestones, each carrying a citation to the governing transitional article, and
helpers to render an "as-of" applicability note for the agent context. It is
deliberately annotation-only: it informs the answer, it does not filter the
graph.

Sources: AI Act (2024/1689) Art 113; MDR (2017/745) Art 123 + Art 120 as amended
by Reg (EU) 2023/607; IVDR (2017/746) Art 113 + Art 110 as amended by Reg (EU)
2022/112; GDPR (2016/679) Art 99.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class Milestone:
    """A date from which a defined scope of a regulation becomes applicable."""
    date: date
    scope: str       # what becomes applicable on this date
    citation: str    # governing provision


@dataclass(frozen=True)
class RegulationApplicability:
    celex: str
    short_name: str
    entry_into_force: date
    general_application: date | None
    milestones: tuple[Milestone, ...]
    transitional_note: str = ""


# ---------------------------------------------------------------------------
# Curated milestones
# ---------------------------------------------------------------------------

APPLICABILITY: dict[str, RegulationApplicability] = {
    # EU AI Act — staggered application under Article 113.
    "32024R1689": RegulationApplicability(
        celex="32024R1689",
        short_name="EU AI Act",
        entry_into_force=date(2024, 8, 1),
        general_application=date(2026, 8, 2),
        milestones=(
            Milestone(date(2025, 2, 2),
                      "Chapters I–II: general provisions and prohibited AI practices (Art 5)",
                      "Art 113(a)"),
            Milestone(date(2025, 8, 2),
                      "Ch III Sec 4 (notified bodies), Ch V (general-purpose AI models), "
                      "Ch VII (governance), Ch XII (penalties), Art 78",
                      "Art 113(b)"),
            Milestone(date(2026, 8, 2),
                      "General application, incl. Annex III high-risk systems (Art 6(2))",
                      "Art 113"),
            Milestone(date(2027, 8, 2),
                      "Art 6(1) high-risk classification for Annex I products and their obligations",
                      "Art 113(c)"),
        ),
    ),
    # MDR.
    "32017R0745": RegulationApplicability(
        celex="32017R0745",
        short_name="MDR",
        entry_into_force=date(2017, 5, 25),
        general_application=date(2021, 5, 26),
        milestones=(
            Milestone(date(2021, 5, 26), "Date of application", "Art 123(2)"),
            Milestone(date(2027, 12, 31),
                      "Legacy MDD/AIMDD Class III and implantable Class IIb transition deadline",
                      "Art 120 (as amended by Reg (EU) 2023/607)"),
            Milestone(date(2028, 12, 31),
                      "Legacy other Class IIb, Class IIa, Im/Is and up-classified Class I deadline",
                      "Art 120 (as amended by Reg (EU) 2023/607)"),
        ),
        transitional_note=(
            "Legacy device transition extended by Reg (EU) 2023/607; deadlines run by "
            "risk class to 2027-12-31 / 2028-12-31, conditions apply (Art 120)."
        ),
    ),
    # IVDR.
    "32017R0746": RegulationApplicability(
        celex="32017R0746",
        short_name="IVDR",
        entry_into_force=date(2017, 5, 26),
        general_application=date(2022, 5, 26),
        milestones=(
            Milestone(date(2022, 5, 26), "Date of application", "Art 113(1)"),
            Milestone(date(2025, 5, 26), "Legacy Class D devices transition deadline",
                      "Art 110 (as amended by Reg (EU) 2022/112)"),
            Milestone(date(2026, 5, 26), "Legacy Class C devices transition deadline",
                      "Art 110 (as amended by Reg (EU) 2022/112)"),
            Milestone(date(2027, 5, 26), "Legacy Class B and Class A sterile transition deadline",
                      "Art 110 (as amended by Reg (EU) 2022/112)"),
        ),
        transitional_note=(
            "IVDR transitional periods for legacy devices are staggered by risk class "
            "through 2027 under Art 110 as amended by Reg (EU) 2022/112."
        ),
    ),
    # GDPR.
    "32016R0679": RegulationApplicability(
        celex="32016R0679",
        short_name="GDPR",
        entry_into_force=date(2016, 5, 24),
        general_application=date(2018, 5, 25),
        milestones=(
            Milestone(date(2018, 5, 25), "Date of application", "Art 99(2)"),
        ),
    ),
}


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def status_as_of(celex: str, as_of: date) -> dict | None:
    """Return the applicability status of *celex* as of *as_of*.

    ``None`` if the regulation is not in the curated set. Otherwise a dict with
    ``in_force`` (bool), ``generally_applicable`` (bool), and ``applied`` /
    ``pending`` lists of milestones split at *as_of*.
    """
    reg = APPLICABILITY.get(celex)
    if reg is None:
        return None
    applied = [m for m in reg.milestones if m.date <= as_of]
    pending = [m for m in reg.milestones if m.date > as_of]
    return {
        "celex": celex,
        "short_name": reg.short_name,
        "in_force": reg.entry_into_force <= as_of,
        "generally_applicable": bool(
            reg.general_application and reg.general_application <= as_of
        ),
        "entry_into_force": reg.entry_into_force,
        "applied": applied,
        "pending": pending,
    }


def applicability_note(celexes: set[str], as_of: date) -> str:
    """Render a compact, citation-bearing applicability note for the agent.

    Returns ``""`` when none of *celexes* has staged/pending applicability worth
    flagging (e.g. a long-applicable regulation with nothing pending), to avoid
    noise. Regulations with pending milestones as of *as_of* are always shown.
    """
    lines: list[str] = []
    for celex in sorted(celexes):
        st = status_as_of(celex, as_of)
        if st is None:
            continue
        reg = APPLICABILITY[celex]
        has_pending = bool(st["pending"])
        is_staged = len(reg.milestones) > 1
        # Only surface regulations that are temporally interesting as of as_of:
        # something pending, or a multi-stage regulation (e.g. the AI Act).
        if not has_pending and not is_staged:
            continue
        head = f"{reg.short_name} ({celex}) — in force since {reg.entry_into_force.isoformat()}"
        if not st["generally_applicable"] and reg.general_application:
            head += f"; general application {reg.general_application.isoformat()}"
        lines.append(head + ":")
        for m in reg.milestones:
            mark = "applies" if m.date <= as_of else "NOT YET applicable"
            lines.append(f"    - {m.date.isoformat()} [{mark}] {m.scope} ({m.citation})")
        if reg.transitional_note:
            lines.append(f"    - transitional: {reg.transitional_note}")
    if not lines:
        return ""
    header = (
        f"APPLICABILITY (as of {as_of.isoformat()}) — obligations below may not "
        f"all be in effect yet; cite the applicable date when it is material:"
    )
    return header + "\n" + "\n".join(lines)
