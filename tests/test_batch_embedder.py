from infrastructure.embeddings.batch_embedder import _build_fetch_query


def test_build_fetch_query_without_filters():
    query, params = _build_fetch_query()

    assert "n.celex IN $doc_ids" not in query
    assert "n.embedding IS NULL" not in query
    assert "doc_ids" not in params
    assert "kinds" in params


def test_build_fetch_query_with_doc_scope_and_missing_filter():
    query, params = _build_fetch_query(
        doc_ids=["MDCG_2020_3", "32024R1689", "MDCG_2020_3"],
        only_missing=True,
    )

    assert query.count("n.celex IN $doc_ids") == 2
    assert query.count("n.embedding IS NULL") == 2
    assert params["doc_ids"] == ["32024R1689", "MDCG_2020_3"]
