import json

from domain.legislation_catalog import AI_ACT_CELEX, MDR_CELEX, IVDR_CELEX
from canonicalization.crosslinker import (
    build_target_id,
    discover_resolvable_refs,
    narrow_definition_quotes,
    narrow_document_ref,
)


def test_discover_resolvable_refs_tags_guidance_sources(monkeypatch, tmp_path):
    legislation_root = tmp_path / "legislation"
    guidance_root = tmp_path / "guidance"

    leg_doc = legislation_root / AI_ACT_CELEX / "EN"
    leg_doc.mkdir(parents=True)
    leg_doc.joinpath("parsed.json").write_text(
        json.dumps({
            "celex_id": AI_ACT_CELEX,
            "relations": [
                {
                    "source": f"{AI_ACT_CELEX}_art_1",
                    "type": "CITES_EXTERNAL",
                    "target": "ext_regulation_eu_2017_745",
                    "properties": {"number": "2017/745", "ref_text": "Article 1 MDR"},
                }
            ],
        }),
        encoding="utf-8",
    )

    guidance_doc = guidance_root / "MDCG_2025_6" / "EN"
    guidance_doc.mkdir(parents=True)
    guidance_doc.joinpath("parsed.json").write_text(
        json.dumps({
            "celex_id": "MDCG_2025_6",
            "relations": [
                {
                    "source": "MDCG_2025_6_sec_4_1",
                    "type": "CITES_EXTERNAL",
                    "target": "ext_regulation_eu_2024_1689",
                    "properties": {"number": "2024/1689", "ref_text": "Article 6(1) AIA"},
                }
            ],
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr("canonicalization.crosslinker._DATA_ROOT", legislation_root)
    monkeypatch.setattr("canonicalization.crosslinker._GUIDANCE_ROOT", guidance_root)

    refs = discover_resolvable_refs()

    assert len(refs) == 2
    families = {ref["source"]: ref["_source_family"] for ref in refs}
    assert families[f"{AI_ACT_CELEX}_art_1"] == "legislation"
    assert families["MDCG_2025_6_sec_4_1"] == "guidance"


# ---------------------------------------------------------------------------
# narrow_document_ref — recover provision-level targets from source text when
# the extracted ref_text carried only the bare regulation name.
# ---------------------------------------------------------------------------


def test_narrow_binds_annex_to_the_named_regulation():
    # The MDCG 2019-11 Section 4 case: ref_text was just the regulation, but
    # the text explicitly names Annex VIII *of* that regulation.
    text = (
        "All implementing rules in Annex VIII of Regulation (EU) 2017/745 "
        "shall be considered."
    )
    parts = narrow_document_ref(text, MDR_CELEX)
    assert [build_target_id(MDR_CELEX, p) for p in parts] == [f"{MDR_CELEX}_anx_VIII"]


def test_narrow_supports_short_regulation_names():
    text = "The provider must also observe MDR Article 120 and AI Act Annex III."
    assert [build_target_id(MDR_CELEX, p) for p in narrow_document_ref(text, MDR_CELEX)] == [
        f"{MDR_CELEX}_art_120"
    ]
    assert [build_target_id(AI_ACT_CELEX, p) for p in narrow_document_ref(text, AI_ACT_CELEX)] == [
        f"{AI_ACT_CELEX}_anx_III"
    ]


def test_narrow_never_attributes_unbound_mentions():
    # A bare "Annex VIII" in a dual MDR/IVDR guidance is ambiguous — it must
    # not be attributed to either regulation by guess.
    text = "All implementing rules in Annex VIII shall be considered."
    assert narrow_document_ref(text, MDR_CELEX) == []
    assert narrow_document_ref(text, IVDR_CELEX) == []


def test_narrow_keeps_dual_regulation_bindings_separate():
    text = (
        "Annex VIII of Regulation (EU) 2017/745 applies to devices, while "
        "Annex VIII of Regulation (EU) 2017/746 applies to IVDs."
    )
    assert [build_target_id(MDR_CELEX, p) for p in narrow_document_ref(text, MDR_CELEX)] == [
        f"{MDR_CELEX}_anx_VIII"
    ]
    assert [build_target_id(IVDR_CELEX, p) for p in narrow_document_ref(text, IVDR_CELEX)] == [
        f"{IVDR_CELEX}_anx_VIII"
    ]


def test_narrow_deduplicates_repeated_mentions():
    text = (
        "Article 6 of Regulation (EU) 2024/1689 sets the classification rules; "
        "see also AI Act Article 6 for the derogation."
    )
    parts = narrow_document_ref(text, AI_ACT_CELEX)
    assert [build_target_id(AI_ACT_CELEX, p) for p in parts] == [f"{AI_ACT_CELEX}_art_6"]


def test_narrow_returns_empty_for_empty_text():
    assert narrow_document_ref("", MDR_CELEX) == []


# ---------------------------------------------------------------------------
# narrow_definition_quotes — a quoted "'term' means…" attributed to a
# regulation narrows to that regulation's defining provision.
# ---------------------------------------------------------------------------


def _doc_edge(celex, text, source="MDCG_X_sec_2"):
    return {
        "source": source,
        "target": f"{celex}_document",
        "fallback": None,
        "ref_text": f"Regulation {celex}",
        "rel_type": "INTERPRETS",
        "source_family": "guidance",
        "_source_text": text,
        "_target_celex": celex,
    }


def test_definition_quote_narrows_to_defining_provision():
    term_map = {(MDR_CELEX, "intended purpose"): f"{MDR_CELEX}_art_2_pt_12"}
    text = (
        "According to Regulation (EU) 2017/745 – MDR, “Intended purpose” "
        "means the use for which a device is intended."
    )
    edges, added, count = narrow_definition_quotes([_doc_edge(MDR_CELEX, text)], term_map)
    assert count == 1
    assert added == {f"{MDR_CELEX}_art_2_pt_12"}
    assert edges[0]["target"] == f"{MDR_CELEX}_art_2_pt_12"
    # Husk stays as fallback so a stale term id degrades to old behaviour
    assert edges[0]["fallback"] == f"{MDR_CELEX}_document"
    assert edges[0]["narrowed_from_document"] is True


def test_definition_quote_scopes_lookup_to_the_edge_regulation():
    # The term is only defined in the MDR; the IVDR edge must stay document-level.
    term_map = {(MDR_CELEX, "intended purpose"): f"{MDR_CELEX}_art_2_pt_12"}
    text = "“Intended purpose” means the use for which a device is intended."
    edges, added, count = narrow_definition_quotes([_doc_edge(IVDR_CELEX, text)], term_map)
    assert count == 0 and added == set()
    assert edges[0]["target"] == f"{IVDR_CELEX}_document"


def test_definition_quote_strips_bookkeeping_keys_from_all_edges():
    edges, _, _ = narrow_definition_quotes(
        [_doc_edge(MDR_CELEX, "no quotes here")], {},
    )
    assert "_source_text" not in edges[0] and "_target_celex" not in edges[0]
