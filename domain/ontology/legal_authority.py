"""Interpretive-authority taxonomy for INTERPRETS sources.

A source that *interprets* a binding provision does not itself always carry
binding force, and the difference is legally material:

- **MDCG guidance** is persuasive soft law — useful, but not legally binding.
- A **harmonised standard** (cited in the OJ) confers a *presumption of
  conformity* with the requirement it covers.
- A **Commission implementing/delegated act** or **common specification** is
  binding law.

Flattening every INTERPRETS edge to a single ``"persuasive"`` string (the prior
behaviour) erases this. Deriving the authority from the source type lets the
agent weight an interpretive source correctly. Only MDCG guidance is currently
ingested, so all edges resolve to PERSUASIVE today — but the mapping is the
single place to extend when other source types are added.
"""
from __future__ import annotations

PERSUASIVE = "persuasive"                                 # non-binding guidance (MDCG, FAQs)
PRESUMPTION_OF_CONFORMITY = "presumption_of_conformity"   # harmonised standards
BINDING = "binding"                                       # implementing/delegated acts, common specs

AUTHORITY_LEVELS: tuple[str, ...] = (
    PERSUASIVE,
    PRESUMPTION_OF_CONFORMITY,
    BINDING,
)

# Source family / document type -> interpretive authority.
_AUTHORITY_BY_SOURCE: dict[str, str] = {
    "guidance": PERSUASIVE,
    "harmonised_standard": PRESUMPTION_OF_CONFORMITY,
    "implementing_act": BINDING,
    "delegated_act": BINDING,
    "common_specification": BINDING,
}

# Conservative default: never over-state an unknown source's legal weight.
_DEFAULT_AUTHORITY = PERSUASIVE


def authority_for_source(source_type: str | None) -> str:
    """Return the interpretive authority level for an INTERPRETS source type.

    Unknown or missing source types default to PERSUASIVE — the safe assumption
    is that an unrecognized interpretive source is non-binding.
    """
    if not source_type:
        return _DEFAULT_AUTHORITY
    return _AUTHORITY_BY_SOURCE.get(source_type.strip().lower(), _DEFAULT_AUTHORITY)
