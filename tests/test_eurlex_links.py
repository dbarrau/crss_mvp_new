"""EUR-Lex provision hyperlinks (application/_eurlex_links.py).

Pins the two facts a link rests on — the anchor derived from the reference shape
and the CELEX recovered from the adjacent regulation name — plus the safety rule
(ambiguous -> bold-only, never a wrong-regulation link) and the amendment-aware
link target (consolidated source_celex for MDR/IVDR/GDPR; inserted articles left
bold-only). No Neo4j / LLM.
"""
from __future__ import annotations

from application._eurlex_links import (
    build_link_scope,
    build_provisions_footer,
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


# ── single-target default: the dropped-qualifier gap ─────────────────────────

def test_single_target_default_links_unqualified_reference():
    # all-AI-Act answer where the model dropped the "AI Act" qualifier; GDPR is
    # also retrieved so scope-uniqueness cannot fire — the default must catch it
    out = link_references(
        "Providers must report under Article 55(1)(c) to the AI Office.",
        in_scope_celexes=frozenset({_AI, _GDPR}),
        default_celex=_AI,
    )
    assert _links_to(out, _AI, "art_55")


def test_named_cross_reference_overrides_the_default():
    # adjacency wins over the default: a named GDPR ref inside an AI-Act answer
    out = link_references(
        "deployers must comply with Article 9 GDPR on special categories",
        in_scope_celexes=frozenset({_AI, _GDPR}),
        default_celex=_AI,
    )
    assert _links_to(out, _GDPR_CONS, "art_9")
    assert f"CELEX:{_AI}" not in out


def test_no_default_leaves_unqualified_bold_only():
    # without a single target, an unqualified ref across two regs stays bold-only
    out = link_references(
        "Providers must report under Article 55.",
        in_scope_celexes=frozenset({_AI, _GDPR}),
        default_celex=None,
    )
    assert out == "Providers must report under **Article 55**."


def test_preceding_name_also_resolves():
    out = link_references("the AI Act's Article 6 test", in_scope_celexes=frozenset({_AI, _MDR}))
    assert _links_to(out, _AI, "art_6")


def test_bound_name_after_ref_with_of_the_resolves():
    # "Article 6(1) of the AI Act" — the name is bound to the reference (only
    # citation scaffolding between them), so it links to the AI Act even though
    # the MDR is also in scope.
    out = link_references(
        "high-risk under Article 6(1) of the AI Act",
        in_scope_celexes=frozenset({_AI, _MDR, _GDPR}),
    )
    assert _links_to(out, _AI, "art_6")


# ── never a WRONG-regulation link (unbound / concept names must not hijack) ───

def test_ambient_other_regulation_name_does_not_hijack_the_link():
    # An AI-Act Article 6 reference in a sentence that merely MENTIONS the MDR
    # ("under the MDR, satisfying Article 6(1)(b)") must never link to MDR
    # Article 6 (which is 'Distance sales'). A content word sits between the name
    # and the reference, so the name is not bound to it → bold-only, never wrong.
    out = link_references(
        "conformity assessment under the MDR, satisfying Article 6(1)(b)",
        in_scope_celexes=frozenset({_AI, _MDR, _GDPR}),
    )
    assert f"CELEX:{_MDR_CONS}" not in out          # NOT linked to the MDR
    assert "](http" not in out                       # bold-only, no link emitted
    assert "**Article 6(1)**" in out


def test_subject_matter_concept_never_disambiguates_a_reference():
    # "Class IIb" is a subject-matter concept, not an act identifier; it must not
    # pull an Article into the MDR (the old retrieval-scope patterns did exactly
    # this). No bound act name → bold-only.
    out = link_references(
        "A Class IIb device must comply with Article 6.",
        in_scope_celexes=frozenset({_AI, _MDR}),
    )
    assert f"CELEX:{_MDR_CONS}" not in out
    assert "](http" not in out


def test_bound_mdr_name_still_resolves():
    # the fix must not over-reject: a genuinely bound MDR reference still links
    out = link_references("see Article 10 of the MDR", in_scope_celexes=frozenset({_AI, _MDR}))
    assert _links_to(out, _MDR_CONS, "art_10")


# ── amendment awareness ──────────────────────────────────────────────────────

_OMNIBUS = "32026R1744"


def test_inserted_article_links_to_amending_act():
    # base act has no #art_4a anchor → link to the amending act's Article 1
    out = link_references("Article 4a AI Act applies",
                          in_scope_celexes=frozenset({_AI}),
                          inserted_articles={(_AI, "4a"): _OMNIBUS})
    assert _links_to(out, _OMNIBUS, "art_1")
    assert "[**Article 4a**](" in out


def test_inserted_article_dedupes_to_one_amender_link_per_section():
    s = "Article 4a AI Act and Article 75a AI Act both apply."
    out = link_references(s, in_scope_celexes=frozenset({_AI}),
                          inserted_articles={(_AI, "4a"): _OMNIBUS, (_AI, "75a"): _OMNIBUS})
    assert out.count("](https://eur-lex") == 1        # both point at Omnibus art_1 → dedupe


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
    assert inserted == {(_AI, "4a"): "32026R1744"}    # inserted root → its amender


def test_build_link_scope_ignores_root_amend_on_normal_article():
    # a merely-amended article carries amended_by on sub-paragraphs, not the root
    provisions = [{"celex": _AI, "subtree": [
        {"id": f"{_AI}_art_25", "kind": "article", "amended_by": None},
        {"id": f"{_AI}_025.002", "kind": "paragraph", "amended_by": "32026R1744"},
    ]}]
    _, inserted = build_link_scope(provisions, target_celexes=None)
    assert inserted == {}


def test_build_link_scope_amended_numeric_article_is_not_inserted():
    # Article 75 is amended in place (root amended_by) but KEEPS its number, so it
    # still exists in the base act — it must not be treated as inserted, or its
    # citation would be wrongly routed to the amending act.
    provisions = [{"celex": _AI, "subtree": [
        {"id": f"{_AI}_art_75", "kind": "article", "amended_by": "32026R1744"},
    ]}]
    _, inserted = build_link_scope(provisions, target_celexes=None)
    assert inserted == {}                             # numeric → not inserted


def test_amended_numeric_article_links_to_base_not_amender():
    # even flagged amended, Article 75 exists in the base act → base link
    out = link_references("Article 75 AI Act", in_scope_celexes=frozenset({_AI}),
                          inserted_articles={})       # 75 not in inserted set
    assert _links_to(out, _AI, "art_75")


def test_footer_amending_act_line_links_to_omnibus():
    cited = {(_AI, "art_50"): "Article 50"}          # answer cites the AI Act → Omnibus is relevant
    out = build_provisions_footer(cited, {_AI: "EU AI Act"}, amender_celexes={_OMNIBUS})
    assert "Amending act" in out
    assert f"uri=CELEX:{_OMNIBUS}&qid=" in out and "#art_1" in out


def test_footer_drops_amender_that_amends_an_uncited_regulation():
    # an MDR answer must NOT show "Amending act — Digital Omnibus on AI": the
    # Omnibus amends the AI Act, which this answer does not cite.
    cited = {(_MDR, "art_33"): "Article 33"}
    out = build_provisions_footer(cited, {_MDR: "MDR 2017/745"}, amender_celexes={_OMNIBUS})
    assert "Amending act" not in out
    assert _OMNIBUS not in out


def test_footer_skips_amender_already_listed_as_cited_regulation():
    # if the amender itself is a cited regulation, no duplicate "Amending act" line
    cited = {(_OMNIBUS, "art_1"): "Article 1"}
    out = build_provisions_footer(cited, {}, amender_celexes={_OMNIBUS})
    assert "Amending act" not in out


# ── dedupe: article-grained anchors linked once per section ───────────────────

def test_repeated_article_linked_once_per_section():
    s = "Under Article 50(1) AI Act; then Article 50(2) AI Act and Article 50(3) AI Act."
    out = link_references(s, in_scope_celexes=frozenset({_AI}))
    assert out.count("](https://eur-lex") == 1        # only the first Article 50 is a link
    assert "**Article 50(2)**" in out and "](https" not in out.split("**Article 50(2)**")[0][-20:]
    assert "[**Article 50(1)**](" in out              # first mention is the link


def test_dedupe_resets_at_section_heading():
    s = "### A\nArticle 50 AI Act here.\n### B\nArticle 50 AI Act again."
    out = link_references(s, in_scope_celexes=frozenset({_AI}))
    assert out.count("](https://eur-lex") == 2        # one link per section


def test_distinct_articles_each_link_in_same_section():
    s = "Article 6 AI Act and Article 50 AI Act and Article 6 AI Act again."
    out = link_references(s, in_scope_celexes=frozenset({_AI}))
    # Article 6 linked once, Article 50 linked once → 2 links (second Art 6 deduped)
    assert out.count("](https://eur-lex") == 2


# ── "Provisions cited" footer ────────────────────────────────────────────────

def test_cited_collects_every_distinct_provision_even_when_deduped():
    s = "Article 50(1) AI Act, Article 50(2) AI Act, Annex III AI Act."
    cited: dict = {}
    link_references(s, in_scope_celexes=frozenset({_AI}), cited=cited)
    assert cited == {(_AI, "art_50"): "Article 50", (_AI, "anx_III"): "Annex III"}


def test_footer_grouped_sorted_and_linked():
    cited = {
        (_AI, "art_50"): "Article 50",
        (_AI, "art_6"): "Article 6",
        (_AI, "anx_III"): "Annex III",
        (_MDR, "art_2"): "Article 2",
    }
    out = build_provisions_footer(cited, {_AI: "EU AI Act", _MDR: "MDR 2017/745"})
    assert "**Provisions cited**" in out
    # AI Act group ordered Article 6 → Article 50 → Annex III (document order)
    assert out.index("Article 6](") < out.index("Article 50](") < out.index("Annex III](")
    # MDR links to its consolidated source_celex
    assert f"uri=CELEX:{_MDR_CONS}&qid=" in out
    assert "**EU AI Act**" in out and "**MDR 2017/745**" in out


def test_footer_empty_when_nothing_cited():
    assert build_provisions_footer({}, {}) == ""


def test_footer_falls_back_to_catalog_name_without_reg_name():
    out = build_provisions_footer({(_AI, "art_6"): "Article 6"}, reg_names={})
    assert LEGISLATION[_AI]["name"] in out             # e.g. "EU AI Act"
