from canonicalization.crosslinker import discover_resolvable_refs


def test_discover_resolvable_refs_tags_guidance_sources(monkeypatch, tmp_path):
    legislation_root = tmp_path / "legislation"
    guidance_root = tmp_path / "guidance"

    leg_doc = legislation_root / "32024R1689" / "EN"
    leg_doc.mkdir(parents=True)
    leg_doc.joinpath("parsed.json").write_text(
        """
        {
          "celex_id": "32024R1689",
          "relations": [
            {
              "source": "32024R1689_art_1",
              "type": "CITES_EXTERNAL",
              "target": "ext_regulation_eu_2017_745",
              "properties": {"number": "2017/745", "ref_text": "Article 1 MDR"}
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    guidance_doc = guidance_root / "MDCG_2025_6" / "EN"
    guidance_doc.mkdir(parents=True)
    guidance_doc.joinpath("parsed.json").write_text(
        """
        {
          "celex_id": "MDCG_2025_6",
          "relations": [
            {
              "source": "MDCG_2025_6_sec_4_1",
              "type": "CITES_EXTERNAL",
              "target": "ext_regulation_eu_2024_1689",
              "properties": {"number": "2024/1689", "ref_text": "Article 6(1) AIA"}
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr("canonicalization.crosslinker._DATA_ROOT", legislation_root)
    monkeypatch.setattr("canonicalization.crosslinker._GUIDANCE_ROOT", guidance_root)

    refs = discover_resolvable_refs()

    assert len(refs) == 2
    families = {ref["source"]: ref["_source_family"] for ref in refs}
    assert families["32024R1689_art_1"] == "legislation"
    assert families["MDCG_2025_6_sec_4_1"] == "guidance"
