"""Tests for the consolidated-act preamble supplement (dispatcher graft).

Consolidated EUR-Lex texts (CONSLEG) omit the preamble, so GDPR/MDR/IVDR
parsed with zero recitals while the AI Act (original CELEX) carried all 180.
``_supplement_preamble`` grafts the original act's preamble subtree into the
consolidated parse; ``_stamp_regulation_provenance`` keeps recitals
interpretive so a rebuild cannot revert the 13 Jul 2026 migration.
"""
from pathlib import Path

import pytest

from ingestion.parse.dispatcher import (
    _stamp_regulation_provenance,
    _supplement_preamble,
)

CELEX = "32016R0679"


def _consolidated_provisions() -> list[dict]:
    root = {
        "id": f"{CELEX}_document", "kind": "document", "text": "",
        "hierarchy_depth": 0, "path": [], "parent_id": None,
        "children": [f"{CELEX}_enc_1"], "lang": "EN",
    }
    enacting = {
        "id": f"{CELEX}_enc_1", "kind": "enacting_terms", "text": "",
        "hierarchy_depth": 1, "path": [root["id"]], "parent_id": root["id"],
        "children": [f"{CELEX}_art_7"], "lang": "EN",
    }
    art = {
        "id": f"{CELEX}_art_7", "kind": "article", "text": "Conditions for consent",
        "hierarchy_depth": 2, "path": [root["id"], enacting["id"]],
        "parent_id": enacting["id"], "children": [], "lang": "EN",
    }
    return [root, enacting, art]


def _fake_original_parser(html, celex, regulation_id, lang="EN"):
    """Mimics parse_eurlex_html on the ORIGINAL act: same id scheme, full doc."""
    root_id = f"{celex}_document"
    return {
        "provisions": [
            {"id": root_id, "kind": "document", "children": [f"{celex}_preamble", f"{celex}_enc_1"]},
            {"id": f"{celex}_preamble", "kind": "preamble", "text": "",
             "hierarchy_depth": 1, "path": [root_id], "parent_id": root_id,
             "children": [f"{celex}_cit_1", f"{celex}_rct_42", f"{celex}_rct_43"], "lang": "EN"},
            {"id": f"{celex}_cit_1", "kind": "citation", "text": "Having regard to the Treaty",
             "hierarchy_depth": 2, "path": [root_id, f"{celex}_preamble"],
             "parent_id": f"{celex}_preamble", "children": [], "lang": "EN", "number": "1"},
            {"id": f"{celex}_rct_42", "kind": "recital", "text": "(42) Consent should be freely given.",
             "hierarchy_depth": 2, "path": [root_id, f"{celex}_preamble"],
             "parent_id": f"{celex}_preamble", "children": [], "lang": "EN", "number": "42"},
            {"id": f"{celex}_rct_43", "kind": "recital", "text": "(43) Clear imbalance presumption.",
             "hierarchy_depth": 2, "path": [root_id, f"{celex}_preamble"],
             "parent_id": f"{celex}_preamble", "children": [], "lang": "EN", "number": "43"},
            # The original's enacting terms must NOT be grafted (they would
            # duplicate the consolidated body).
            {"id": f"{celex}_art_7", "kind": "article", "text": "Original Art 7",
             "hierarchy_depth": 2, "path": [], "parent_id": None, "children": [], "lang": "EN"},
        ],
        "relations": [
            # recital-sourced cross-ref: must be carried over
            {"source": f"{celex}_rct_43", "type": "CITES", "target": f"{celex}_art_7"},
            # document/enacting-sourced: must NOT be carried (duplicates)
            {"source": root_id, "type": "CITES_EXTERNAL", "target": "ext_x"},
            {"source": f"{celex}_art_7", "type": "CITES", "target": f"{celex}_art_9"},
        ],
    }


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    (tmp_path / "raw_preamble.html").write_text("<html>original act</html>")
    (tmp_path / "raw.html").write_text("<html>consolidated</html>")
    return tmp_path


def test_graft_inserts_preamble_subtree_before_enacting_terms(raw_dir):
    provisions = _consolidated_provisions()
    relations: list[dict] = []
    n = _supplement_preamble(
        provisions, relations, raw_dir / "raw.html",
        _fake_original_parser, CELEX, "GDPR", "EN",
    )
    assert n == 2
    root = provisions[0]
    assert root["children"][0] == f"{CELEX}_preamble"  # legal document order
    kinds = [p["kind"] for p in provisions]
    assert kinds.count("recital") == 2
    assert kinds.count("preamble") == 1
    # the original's enacting terms are not duplicated
    assert sum(1 for p in provisions if p["id"] == f"{CELEX}_art_7") == 1
    assert provisions[3]["text"] != "Original Art 7"  # consolidated copy kept


def test_graft_carries_only_subtree_sourced_relations(raw_dir):
    provisions = _consolidated_provisions()
    relations: list[dict] = []
    _supplement_preamble(
        provisions, relations, raw_dir / "raw.html",
        _fake_original_parser, CELEX, "GDPR", "EN",
    )
    sources = {r["source"] for r in relations}
    assert f"{CELEX}_rct_43" in sources          # recital cross-ref carried
    assert f"{CELEX}_document" not in sources    # duplicate CITES_EXTERNAL dropped
    assert f"{CELEX}_art_7" not in sources       # enacting relations dropped


def test_noop_when_preamble_already_present(raw_dir):
    provisions = _consolidated_provisions()
    provisions.append({"id": f"{CELEX}_preamble", "kind": "preamble", "children": []})
    n = _supplement_preamble(
        provisions, [], raw_dir / "raw.html",
        _fake_original_parser, CELEX, "GDPR", "EN",
    )
    assert n == 0


def test_noop_without_supplement_file(tmp_path):
    (tmp_path / "raw.html").write_text("<html>consolidated</html>")
    provisions = _consolidated_provisions()
    n = _supplement_preamble(
        provisions, [], tmp_path / "raw.html",
        _fake_original_parser, CELEX, "GDPR", "EN",
    )
    assert n == 0
    assert len(provisions) == 3


def test_stamp_keeps_recitals_interpretive():
    """The blanket 'binding' stamp defeated the loader's kind-default on every
    rebuild; the stamp must be kind-aware."""
    provisions = [
        {"kind": "article"},
        {"kind": "recital"},
        {"kind": "citation"},
        {"kind": "preamble"},
        {"kind": "paragraph"},
    ]
    _stamp_regulation_provenance(provisions)
    by_kind = {p["kind"]: p["binding_force"] for p in provisions}
    assert by_kind["article"] == "binding"
    assert by_kind["paragraph"] == "binding"
    assert by_kind["recital"] == "interpretive"
    assert by_kind["citation"] == "interpretive"
    assert by_kind["preamble"] == "interpretive"
    assert all(p["source_type"] == "regulation" for p in provisions)
