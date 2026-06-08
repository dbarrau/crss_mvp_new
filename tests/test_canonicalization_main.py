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

    def fake_roles(*, dry_run):
        calls.append(("roles", dry_run, None))
        return {"actor_roles": 5, "obligation_of": 9}

    def fake_communities(*, dry_run, seed=42, celex_filter=None):
        calls.append(("communities", dry_run, None))
        return {"nodes": 6, "edges": 10, "communities": 3}

    monkeypatch.setattr(canonicalization_main, "crosslink", fake_crosslink)
    monkeypatch.setattr(canonicalization_main, "link_delegations", fake_delegations)
    monkeypatch.setattr(canonicalization_main, "link_terms", fake_terms)
    monkeypatch.setattr(canonicalization_main, "link_roles", fake_roles)
    monkeypatch.setattr(canonicalization_main, "link_communities", fake_communities)

    summary = canonicalization_main.run_pipeline(dry_run=True, cleanup=True)

    assert calls == [
        ("crosslink", True, True),
        ("delegations", True, None),
        ("terms", True, None),
        ("roles", True, None),
        ("communities", True, None),
    ]
    assert summary == {
        "crosslinker": {"cites": 1, "interprets": 2},
        "delegation_linker": {"edges_written": 3},
        "term_linker": {"edges": 4},
        "role_linker": {"actor_roles": 5, "obligation_of": 9},
        "community_linker": {"nodes": 6, "edges": 10, "communities": 3},
    }


def test_run_pipeline_skips_communities_when_flag_set(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(canonicalization_main, "crosslink", lambda **_: calls.append("crosslink") or {})
    monkeypatch.setattr(canonicalization_main, "link_delegations", lambda **_: calls.append("delegations") or {})
    monkeypatch.setattr(canonicalization_main, "link_terms", lambda **_: calls.append("terms") or {})
    monkeypatch.setattr(canonicalization_main, "link_roles", lambda **_: calls.append("roles") or {})
    monkeypatch.setattr(
        canonicalization_main,
        "link_communities",
        lambda **_: (_ for _ in ()).throw(AssertionError("community_linker should not be called")),
    )

    summary = canonicalization_main.run_pipeline(skip_communities=True)

    assert "communities" not in calls
    assert summary["community_linker"] == {"nodes": 0, "edges": 0, "communities": 0}
