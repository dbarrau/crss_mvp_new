import canonicalization.__main__ as canonicalization_main


def test_run_pipeline_orders_stages_and_passes_flags(monkeypatch):
    calls: list[tuple[str, bool, bool | None]] = []

    def fake_crosslink(*, dry_run, cleanup):
        calls.append(("crosslink", dry_run, cleanup))
        return {"cites": 1, "interprets": 2}

    def fake_delegations(*, dry_run):
        calls.append(("delegations", dry_run, None))
        return {"edges_written": 3}

    def fake_terms(*, dry_run):
        calls.append(("terms", dry_run, None))
        return {"edges": 4}

    monkeypatch.setattr(canonicalization_main, "crosslink", fake_crosslink)
    monkeypatch.setattr(canonicalization_main, "link_delegations", fake_delegations)
    monkeypatch.setattr(canonicalization_main, "link_terms", fake_terms)

    summary = canonicalization_main.run_pipeline(dry_run=True, cleanup=True)

    assert calls == [
        ("crosslink", True, True),
        ("delegations", True, None),
        ("terms", True, None),
    ]
    assert summary == {
        "crosslinker": {"cites": 1, "interprets": 2},
        "delegation_linker": {"edges_written": 3},
        "term_linker": {"edges": 4},
    }
