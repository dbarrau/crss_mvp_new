"""Unit tests for provision-role bucketing in `_format_context`.

These tests pin the contract between the retriever output (which now carries
``provision_role``) and the formatter (which groups provisions by role into
labelled sections in the LLM context). No Neo4j or LLM is required.
"""
from application._context import _format_context


def _make_provision(**overrides) -> dict:
    base = {
        "article_id": "test_id",
        "celex": "32024R1689",
        "regulation": "AI Act",
        "article_ref": "Article 1",
        "article_path": "",
        "article_text": "Provision body text.",
        "children": [],
        "cited_provisions": [],
        "cross_reg_cited": [],
        "provision_role": None,
        "score": 1.0,
        "matched_leaf_id": None,
    }
    base.update(overrides)
    return base


# ── single-bucket short-circuit ───────────────────────────────────────────


def test_single_provision_no_section_header():
    """A single provision must NOT trigger bucket headers (legacy parity)."""
    prov = _make_provision(article_ref="Article 5(5)", provision_role="EXEMPTS")
    out = _format_context([prov])
    assert "### EXEMPTIONS" not in out
    assert "[1] Article 5(5)" in out


def test_all_same_role_no_section_header():
    """When every provision shares one bucket, suppress headers entirely."""
    provs = [
        _make_provision(article_ref="Article 16", provision_role="OBLIGATION"),
        _make_provision(article_ref="Article 17", provision_role="OBLIGATION"),
    ]
    out = _format_context(provs)
    assert "### OBLIGATIONS" not in out
    assert "[1] Article 16" in out
    assert "[2] Article 17" in out


def test_all_unknown_role_no_section_header():
    """Provisions without a recognised role share the OTHER bucket only."""
    provs = [
        _make_provision(article_ref="Article 1", provision_role=None),
        _make_provision(article_ref="Article 2", provision_role="UNCLASSIFIED"),
    ]
    out = _format_context(provs)
    assert "### OTHER" not in out
    assert "[1] Article 1" in out


# ── role badge in header ──────────────────────────────────────────────────


def test_known_role_renders_badge():
    prov = _make_provision(article_ref="Article 25(1)", provision_role="EXTENDS_STATUS")
    # Single-bucket case still shows the per-provision badge in the header.
    out = _format_context([prov])
    assert "[role: EXTENDS_STATUS]" in out


def test_unknown_role_omits_badge():
    prov = _make_provision(article_ref="Article 99", provision_role="UNCLASSIFIED")
    out = _format_context([prov])
    assert "[role:" not in out


def test_missing_role_omits_badge():
    prov = _make_provision(article_ref="Article 99", provision_role=None)
    out = _format_context([prov])
    assert "[role:" not in out


# ── bucket grouping and ordering ──────────────────────────────────────────


def test_buckets_emitted_when_multiple_roles_present():
    """Provisions with distinct roles get grouped under labelled sections."""
    provs = [
        _make_provision(article_ref="Article 16", provision_role="OBLIGATION"),
        _make_provision(article_ref="Article 3", provision_role="DEFINES"),
        _make_provision(article_ref="Article 5(5)", provision_role="EXEMPTS",
                        celex="32017R0745", regulation="MDR"),
    ]
    out = _format_context(provs)
    assert "### DEFINITIONS ###" in out
    assert "### EXEMPTIONS / MODIFIERS ###" in out
    assert "### OBLIGATIONS ###" in out


def test_bucket_order_is_semantic_not_input_order():
    """DEFINITIONS must precede EXEMPTIONS must precede OBLIGATIONS, regardless
    of the order provisions were supplied in."""
    provs = [
        _make_provision(article_ref="Article 16", provision_role="OBLIGATION"),
        _make_provision(article_ref="Article 5(5)", provision_role="EXEMPTS"),
        _make_provision(article_ref="Article 3", provision_role="DEFINES"),
    ]
    out = _format_context(provs)
    def_idx = out.index("### DEFINITIONS ###")
    exempt_idx = out.index("### EXEMPTIONS / MODIFIERS ###")
    oblig_idx = out.index("### OBLIGATIONS ###")
    assert def_idx < exempt_idx < oblig_idx


def test_other_bucket_appears_last():
    """Provisions without a recognised role go into OTHER at the end."""
    provs = [
        _make_provision(article_ref="Article 1", provision_role=None),
        _make_provision(article_ref="Article 3", provision_role="DEFINES"),
    ]
    out = _format_context(provs)
    def_idx = out.index("### DEFINITIONS ###")
    other_idx = out.index("### OTHER ###")
    assert def_idx < other_idx


def test_numbering_is_contiguous_across_buckets():
    """Citation indices must reflect the input order, not the bucket order.
    The LLM cites with [N], so renumbering would break answer-side citations.
    """
    provs = [
        _make_provision(article_ref="Article 16", provision_role="OBLIGATION"),  # [1]
        _make_provision(article_ref="Article 3", provision_role="DEFINES"),       # [2]
    ]
    out = _format_context(provs)
    # The OBLIGATION provision was retrieved first → it must keep index [1]
    # even though DEFINITIONS bucket is rendered above OBLIGATIONS.
    assert "[1] Article 16" in out
    assert "[2] Article 3" in out


def test_within_bucket_order_preserves_retrieval_order():
    """Two provisions in the same bucket keep their original ranking order."""
    provs = [
        _make_provision(article_ref="Article 16", provision_role="OBLIGATION"),
        _make_provision(article_ref="Article 26", provision_role="OBLIGATION"),
        _make_provision(article_ref="Article 3", provision_role="DEFINES"),
    ]
    out = _format_context(provs)
    a16_idx = out.index("[1] Article 16")
    a26_idx = out.index("[2] Article 26")
    assert a16_idx < a26_idx
