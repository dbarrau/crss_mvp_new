from pathlib import Path

from ingestion.parse import dispatcher
from ingestion.parse.guidance import mdcg_parser


def test_validate_mdcg_markdown_accepts_complete_structure():
    markdown = """# MDCG 2020-3 Rev.1

## 1 Introduction

This section introduces the guidance and ends with a full sentence.

## 2 Scope

This section explains scope and also ends correctly.

## 3 Annex

This section closes properly.
"""

    assert mdcg_parser.validate_mdcg_markdown(markdown) == []


def test_validate_mdcg_markdown_rejects_truncated_heading_transition():
    markdown = """# MDCG 2020-3 Rev.1

## 1 Introduction

This introductory section is complete and ends with a proper sentence.

## 2 Scope

This section contains a long line that is clearly cut off immediately before the

## 3 Annex

This section would never be trusted because the previous section is truncated.
"""

    issues = mdcg_parser.validate_mdcg_markdown(markdown)

    assert len(issues) == 1
    assert "Suspicious heading transition" in issues[0]


def test_parse_guidance_document_regenerates_invalid_cached_markdown(
    monkeypatch,
    tmp_path: Path,
):
    pdf_file = tmp_path / "mdcg_2020_3_rev1.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    clean_md = tmp_path / "mdcg_2020_3_rev1_clean.md"
    clean_md.write_text(
        """# MDCG 2020-3 Rev.1

## 1 Introduction

This section contains a long line that is clearly cut off immediately before the

## 2 Scope

This section should force the cache to be regenerated.
""",
        encoding="utf-8",
    )

    reparsed_markdown = """# MDCG 2020-3 Rev.1

## 1 Introduction

This regenerated section ends cleanly.

## 2 Scope

This regenerated scope section also ends cleanly.

## 3 Annex

This regenerated annex section ends cleanly.
"""

    parse_calls: list[Path] = []
    structured_inputs: list[Path] = []

    def fake_parse_mdcg_pdf(pdf_path: Path, output_dir: Path):
        parse_calls.append(pdf_path)
        clean_md.write_text(reparsed_markdown, encoding="utf-8")
        return {
            "output_files": {
                "clean_markdown": str(clean_md),
            }
        }

    def fake_write_parsed_json(
        md_path,
        doc_id,
        doc_name,
        lang="EN",
        output_path=None,
    ):
        structured_inputs.append(Path(md_path))
        output = Path(output_path)
        output.write_text("{}", encoding="utf-8")
        return output

    monkeypatch.setattr(mdcg_parser, "parse_mdcg_pdf", fake_parse_mdcg_pdf)
    monkeypatch.setattr("ingestion.parse.guidance.mdcg_structurer.write_parsed_json", fake_write_parsed_json)

    parsed_json = dispatcher._parse_guidance_document(
        pdf_file=pdf_file,
        lang="EN",
        doc_id="MDCG_2020_3",
        out_dir=tmp_path,
    )

    assert parse_calls == [pdf_file]
    assert structured_inputs == [clean_md]
    assert parsed_json == tmp_path / "parsed.json"
