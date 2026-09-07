"""Single source of truth for EUR-Lex provision URLs (domain/eurlex.py).

Pins the URL construction the application citation layer depends on: the anchor
grammar, which CELEX a base act points at (consolidated when one exists), and the
full URL shape (viewer form + the qid the viewer needs to honour the anchor).
"""
from __future__ import annotations

from domain import eurlex
from domain.legislation_catalog import AI_ACT_CELEX, MDR_CELEX, LEGISLATION

_MDR_CONS = LEGISLATION[MDR_CELEX]["source_celex"]


# ── anchor grammar ───────────────────────────────────────────────────────────

def test_anchor_for_ref_shapes():
    assert eurlex.anchor_for_ref("Article 25") == "art_25"
    assert eurlex.anchor_for_ref("Article 25(2)") == "art_25"     # article is finest grain
    assert eurlex.anchor_for_ref("Article 4a") == "art_4a"        # inserted-article suffix
    assert eurlex.anchor_for_ref("Annex III") == "anx_III"
    assert eurlex.anchor_for_ref("Recital 81") == "rct_81"
    assert eurlex.anchor_for_ref("Chapter II") is None            # unrecognised → no anchor


def test_display_from_anchor_roundtrips_the_label():
    assert eurlex.display_from_anchor("art_50") == "Article 50"
    assert eurlex.display_from_anchor("anx_III") == "Annex III"
    assert eurlex.display_from_anchor("rct_81") == "Recital 81"


# ── which CELEX a base act links to ──────────────────────────────────────────

def test_display_celex_prefers_consolidation_when_one_exists():
    assert eurlex.display_celex(MDR_CELEX) == _MDR_CONS           # MDR → its consolidation
    assert eurlex.display_celex(AI_ACT_CELEX) == AI_ACT_CELEX     # AI Act has none → base


def test_is_known_celex():
    assert eurlex.is_known_celex(AI_ACT_CELEX)
    assert not eurlex.is_known_celex("99999R9999")


# ── full URL ─────────────────────────────────────────────────────────────────

def test_provision_url_carries_the_qid_before_the_fragment():
    url = eurlex.provision_url(AI_ACT_CELEX, "art_50")
    # the qid must sit between the CELEX and the fragment or EUR-Lex drops the anchor
    assert f"uri=CELEX:{AI_ACT_CELEX}&qid=" in url
    assert url.rsplit("#", 1)[1] == "art_50"
    assert "&qid=" in url


def test_provision_url_uses_consolidated_celex_for_mdr():
    url = eurlex.provision_url(MDR_CELEX, "art_73")
    assert f"uri=CELEX:{_MDR_CONS}&qid=" in url
    assert f"CELEX:{MDR_CELEX}&" not in url            # base CELEX must not be the URL


def test_provision_url_accepts_a_shared_qid():
    url = eurlex.provision_url(AI_ACT_CELEX, "art_6", qid=123)
    assert "&qid=123#art_6" in url
