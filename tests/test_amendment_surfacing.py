"""Amendment surfacing: linker extraction + retrieval-context + prompt directive.

Covers the (b)+(c) path that makes a superseding amendment visible when the base
provision is retrieved — the systemic replacement for the earlier question-gated
date bridge and the inline date annotation.
"""
from __future__ import annotations

from application._context import _format_one_provision
from application._retrieval import _surface_amendments
from application.agent import _build_user_message, _select_question_route
from canonicalization.amendment_linker import _amended_celex_of, _target_id_of
from domain.legislation_catalog import AI_ACT_CELEX

OMNIBUS = "32026R1744"


# ---------------------------------------------------------------------------
# amendment_linker: container + target extraction
# ---------------------------------------------------------------------------

def test_container_resolves_only_loaded_other_regulations():
    # Names a loaded reg (AI Act) → resolved to its CELEX.
    assert _amended_celex_of(
        "Regulation (EU) 2024/1689 is amended as follows:", OMNIBUS
    ) == AI_ACT_CELEX
    # Self-reference → skipped.
    assert _amended_celex_of(
        "Regulation (EU) 2026/1744 is amended as follows:", OMNIBUS
    ) is None
    # Not held in the catalog → skipped (no phantom target).
    assert _amended_celex_of(
        "Regulation (EU) 2002/58 is amended as follows:", OMNIBUS
    ) is None
    # No amendment verb → not a container.
    assert _amended_celex_of(
        "as referred to in Regulation (EU) 2024/1689", OMNIBUS
    ) is None


def test_target_extraction_covers_the_amendment_grammars():
    # "in Article N, … is amended" (Article 113 date case, with EUR-Lex nbsp).
    assert _target_id_of(
        "in Article 113, the third paragraph is amended as follows:", AI_ACT_CELEX
    ) == (f"{AI_ACT_CELEX}_art_113", "Article 113", "amended")
    # "Article N is replaced by the following".
    assert _target_id_of(
        "Article 6 is replaced by the following: '...'", AI_ACT_CELEX
    ) == (f"{AI_ACT_CELEX}_art_6", "Article 6", "replaced")
    # "the following Article Na is inserted".
    assert _target_id_of(
        "the following Article 6a is inserted:", AI_ACT_CELEX
    ) == (f"{AI_ACT_CELEX}_art_6a", "Article 6a", "inserted")
    # Annex target.
    assert _target_id_of(
        "in Annex III, point 2 is replaced by the following:", AI_ACT_CELEX
    ) == (f"{AI_ACT_CELEX}_anx_III", "Annex III", "replaced")


def test_target_extraction_rejects_non_amendment_text():
    # A replacement sub-point that does not itself name the target article: the
    # link belongs on its parent point, not here.
    assert _target_id_of(
        "point (c) is replaced by the following: '(c) ... 2 December 2027 ...'",
        AI_ACT_CELEX,
    ) is None
    # An incidental cross-reference (no amendment verb) is never a target.
    assert _target_id_of(
        "in accordance with Article 6 the provider shall keep records",
        AI_ACT_CELEX,
    ) is None


# ---------------------------------------------------------------------------
# _context: controlling-amendment marker
# ---------------------------------------------------------------------------

def _amending_block(**extra) -> dict:
    base = {
        "article_ref": "Article 1(40)",
        "regulation": "Digital Omnibus on AI",
        "celex": OMNIBUS,
        "article_id": f"{OMNIBUS}_art_1_pt_40",
        "article_text": (
            "in Article 113, the third paragraph is amended as follows: point (c) "
            "is replaced by the following: '(c) ... 2 December 2027 ... 2 August 2028 ...'"
        ),
        "_amends_expansion": True,
        "_amends_target_ref": "Article 113",
        "amending_act": "Regulation (EU) 2026/1744",
    }
    base.update(extra)
    return base


def test_amending_provision_renders_controlling_marker():
    block = _format_one_provision(3, _amending_block(), "OBLIGATION")
    assert "AMENDING PROVISION" in block
    assert "CONTROLLING" in block
    assert "Article 113" in block          # names what it supersedes
    assert "Regulation (EU) 2026/1744" in block


def test_ordinary_provision_has_no_amendment_marker():
    block = _format_one_provision(
        3, _amending_block(_amends_expansion=False), "OBLIGATION"
    )
    assert "AMENDING PROVISION" not in block


# ---------------------------------------------------------------------------
# _prompts: directive fires only on the marker
# ---------------------------------------------------------------------------

def _route():
    return _select_question_route(
        "What does Article 26 of the AI Act require?",
        explicit_refs=["Article 26"],
        mentioned_regs={"EU AI Act"},
        role_specs=[],
        is_definition_question=False,
    )


def test_directive_injected_when_amendment_marker_present():
    msg = _build_user_message(
        question="Q",
        context="… ⚠ AMENDING PROVISION — CONTROLLING (supersedes …) …",
        route=_route(),
        sufficiency={"ok": True},
    )
    assert "AMENDED PROVISIONS — READ BEFORE ANSWERING" in msg
    assert "CONTROLLING" in msg


def test_directive_absent_without_marker():
    msg = _build_user_message(
        question="Q",
        context="ordinary regulatory context with no amendment",
        route=_route(),
        sufficiency={"ok": True},
    )
    assert "AMENDED PROVISIONS — READ BEFORE ANSWERING" not in msg


# ---------------------------------------------------------------------------
# _retrieval: orchestration surfacing pass prepends amendments over the bag
# ---------------------------------------------------------------------------

class _FakeRetriever:
    """Returns an amender only when the amended article's id is queried —
    mirrors GraphRetriever.retrieve_amendments climbing from any child to the
    article that carries the AMENDS edge."""
    def retrieve_amendments(self, ids):
        if f"{AI_ACT_CELEX}_art_113" in ids:
            return [{
                "article_id": f"{OMNIBUS}_art_1_pt_40",
                "_amends_expansion": True,
                "_amends_target_ref": "Article 113",
                "amending_act": "Regulation (EU) 2026/1744",
            }]
        return []


def test_surface_amendments_prepends_controlling_amendment():
    # Article 113 arrived via a (later-positioned) anchor; the amendment must be
    # hoisted to the front so the context-budget trim cannot drop it.
    provisions = [
        {"article_id": f"{AI_ACT_CELEX}_art_6"},
        {"article_id": f"{AI_ACT_CELEX}_art_113",
         "matched_leaf_id": f"{AI_ACT_CELEX}_art_113_sp_3_pt_c"},
    ]
    _surface_amendments(_FakeRetriever(), provisions)
    assert provisions[0]["article_id"] == f"{OMNIBUS}_art_1_pt_40"
    assert provisions[0]["_amends_expansion"] is True


def test_surface_amendments_noop_when_nothing_amended():
    provisions = [{"article_id": f"{AI_ACT_CELEX}_art_26"}]
    _surface_amendments(_FakeRetriever(), provisions)
    assert [p["article_id"] for p in provisions] == [f"{AI_ACT_CELEX}_art_26"]
