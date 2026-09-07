"""EUR-Lex provision hyperlinks (application/_eurlex_links.py).

Pins the two facts a link rests on — the anchor derived from the reference shape
and the CELEX recovered from the adjacent regulation name — plus the safety rule
(ambiguous -> bold-only, never a wrong-regulation link) and the amendment-aware
link target (consolidated source_celex for MDR/IVDR/GDPR; inserted articles left
bold-only). No Neo4j / LLM.
"""
from __future__ import annotations

from application._eurlex_links import (
    _anchor_from_ref,
    build_link_scope,
    link_references,
)
from domain.legislation_catalog import (
    AI_ACT_CELEX,
    MDR_CELEX,
    GDPR_CELEX,
    LEGISLATION,
)

_AI = AI_ACT_CELEX
_MDR = MDR_CELEX
_GDPR = GDPR_CELEX
_MDR_CONS = LEGISLATION[_MDR]["source_celex"]      # 02017R0745-...
_GDPR_CONS = LEGISLATION[_GDPR]["source_celex"]


# ── anchor derivation ────────────────────────────────────────────────────────

def test_anchor_from_ref_shapes():
    assert _anchor_from_ref("Article 25") == "art_25"
    assert _anchor_from_ref("Article 25(2)") == "art_25"        # article is finest grain
    assert _anchor_from_ref("Article 4a") == "art_4a"           # inserted-article suffix
    assert _anchor_from_ref("Annex III") == "anx_III"
    assert _anchor_from_ref("Recital 81") == "rct_81"
    assert _anchor_from_ref("Chapter II") is None               # unrecognised → no anchor


# ── CELEX resolution + link target ───────────────────────────────────────────

def _links_to(out: str, celex: str, anchor: str) -> bool:
    # EUR-Lex needs a qid between the CELEX and the fragment or the anchor is
    # dropped (the /TXT/ viewer's client-side redirect); assert both parts.
    return f"uri=CELEX:{celex}&qid=" in out and f"#{anchor}" in out


def test_adjacent_name_resolves_celex_and_links_base_act():
    out = link_references("under Article 6(1) AI Act", in_scope_celexes=frozenset({_AI}))
    assert _links_to(out, _AI, "art_6")
    assert "&qid=" in out                            # the load-bearing qid is present
    assert out.startswith("under [**Article 6(1)**](")


def test_mdr_links_to_consolidated_source_celex():
    out = link_references("the manufacturer under Article 2(30) MDR",
                          in_scope_celexes=frozenset({_MDR}))
    assert _links_to(out, _MDR_CONS, "art_2")
    assert f"CELEX:{_MDR}&" not in out              # base CELEX must not be the URL


def test_gdpr_links_to_consolidated_source_celex():
    out = link_references("Article 35 GDPR", in_scope_celexes=frozenset({_GDPR}))
    assert _links_to(out, _GDPR_CONS, "art_35")


def test_nearest_name_wins_for_two_refs_in_one_sentence():
    s = "Article 6(1) AI Act and the manufacturer under Article 2(30) MDR"
    out = link_references(s, in_scope_celexes=frozenset({_AI, _MDR}))
    assert _links_to(out, _AI, "art_6")
    assert _links_to(out, _MDR_CONS, "art_2")


def test_annex_and_recital_anchors():
    out = link_references("Annex III AI Act; Recital 81 AI Act",
                          in_scope_celexes=frozenset({_AI}))
    assert _links_to(out, _AI, "anx_III")
    assert _links_to(out, _AI, "rct_81")


# ── the safety rule: never a wrong-regulation link ───────────────────────────

def test_ambiguous_reference_stays_bold_only():
    # no adjacent reg name, two regulations in scope → cannot disambiguate
    out = link_references("This turns on Article 6.",
                          in_scope_celexes=frozenset({_AI, _MDR}))
    assert out == "This turns on **Article 6**."
    assert "http" not in out


def test_scope_uniqueness_resolves_when_single_reg():
    out = link_references("This turns on Article 6.", in_scope_celexes=frozenset({_AI}))
    assert _links_to(out, _AI, "art_6")


def test_preceding_name_also_resolves():
    out = link_references("the AI Act's Article 6 test", in_scope_celexes=frozenset({_AI, _MDR}))
    assert _links_to(out, _AI, "art_6")


# ── amendment awareness ──────────────────────────────────────────────────────

def test_inserted_article_left_bold_only():
    # base act has no #art_4a anchor → suppress the link in Phase 1
    out = link_references("Article 4a AI Act applies",
                          in_scope_celexes=frozenset({_AI}),
                          inserted_articles=frozenset({(_AI, "4a")}))
    assert out == "**Article 4a** AI Act applies"


def test_amended_article_still_links_to_base():
    # Article 25 is partly Omnibus-amended but exists in the base act → still links
    out = link_references("Article 25 AI Act", in_scope_celexes=frozenset({_AI}))
    assert _links_to(out, _AI, "art_25")


# ── heading / quote lines are never altered ──────────────────────────────────

def test_headings_and_quote_lines_skipped():
    text = "## Article 6 AI Act\n> Article 6 AI Act (quoted law)"
    assert link_references(text, in_scope_celexes=frozenset({_AI})) == text


# ── build_link_scope ─────────────────────────────────────────────────────────

def test_build_link_scope_collects_celexes_and_inserted_articles():
    provisions = [
        {"celex": _AI, "subtree": [
            {"id": f"{_AI}_art_4a", "kind": "article", "amended_by": "32026R1744"},
            {"id": f"{_AI}_004a.001", "kind": "paragraph", "amended_by": "32026R1744"},
        ]},
        {"celex": _MDR, "subtree": [
            {"id": f"{_MDR}_art_2", "kind": "article", "amended_by": None},
        ]},
    ]
    in_scope, inserted = build_link_scope(provisions, target_celexes={_AI})
    assert in_scope == frozenset({_AI, _MDR})
    assert inserted == frozenset({(_AI, "4a")})       # only the inserted root


def test_build_link_scope_ignores_root_amend_on_normal_article():
    # a merely-amended article carries amended_by on sub-paragraphs, not the root
    provisions = [{"celex": _AI, "subtree": [
        {"id": f"{_AI}_art_25", "kind": "article", "amended_by": None},
        {"id": f"{_AI}_025.002", "kind": "paragraph", "amended_by": "32026R1744"},
    ]}]
    _, inserted = build_link_scope(provisions, target_celexes=None)
    assert inserted == frozenset()
