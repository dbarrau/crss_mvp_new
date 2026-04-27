from ingestion.parse.base.registry import (
    DOCUMENT_REGISTRY,
    GUIDANCE_IDS,
    LEGISLATION_IDS,
    PARSER_REGISTRY,
    resolve_document,
)


def test_document_registry_covers_both_catalogs():
    assert LEGISLATION_IDS
    assert GUIDANCE_IDS
    assert LEGISLATION_IDS | GUIDANCE_IDS == set(DOCUMENT_REGISTRY)


def test_registry_marks_legislation_and_guidance_families():
    legislation_id = sorted(LEGISLATION_IDS)[0]
    guidance_id = sorted(GUIDANCE_IDS)[0]

    legislation_entry = resolve_document(legislation_id)
    guidance_entry = resolve_document(guidance_id)

    assert legislation_entry is not None
    assert legislation_entry.family == "legislation"
    assert legislation_entry.source_kind == "html"
    assert callable(legislation_entry.parser)

    assert guidance_entry is not None
    assert guidance_entry.family == "guidance"
    assert guidance_entry.source_kind == "pdf"
    assert guidance_entry.parser is None


def test_parser_registry_remains_legislation_only():
    assert set(PARSER_REGISTRY) == set(LEGISLATION_IDS)
    for legislation_id in LEGISLATION_IDS:
        assert callable(PARSER_REGISTRY[legislation_id])
