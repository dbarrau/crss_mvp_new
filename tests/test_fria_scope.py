"""FRIA-scope guard (application/_fria_scope.py).

Article 27(1) scopes the fundamental-rights impact assessment to three deployer
classes. The guard appends a loud SCOPE caveat when the answer asserts the FRIA
duty unconditionally without naming that scope, and stays silent (no false flag)
whenever the scope is stated or the assertion is hedged.
"""
from application._fria_scope import flag_fria_overapplication

_CAVEAT = "FRIA SCOPE"


# ── fires: over-broad, unconditional assertion with no scope qualifier ────────

def test_plain_deployer_overapplication_is_flagged():
    ans = ("As the deployer of the high-risk system, you must perform a "
           "fundamental rights impact assessment under **Article 27**.")
    out, notes = flag_fria_overapplication(ans)
    assert notes and _CAVEAT in out
    assert out.startswith(ans)  # body untouched, caveat appended after


def test_acronym_form_triggers():
    ans = "The deployer shall carry out a FRIA before putting the system into use."
    out, notes = flag_fria_overapplication(ans)
    assert notes and _CAVEAT in out


def test_applies_verb_triggers():
    ans = "Article 27 applies: a fundamental-rights impact assessment is mandatory for the deployer."
    _, notes = flag_fria_overapplication(ans)
    assert notes


# ── silent: the scope caveat is already present (correctly scoped) ────────────

def test_public_body_scope_suppresses():
    ans = ("Because the deployer is a body governed by public law, it must "
           "perform a fundamental rights impact assessment under Article 27(1).")
    out, notes = flag_fria_overapplication(ans)
    assert not notes and out == ans


def test_public_service_scope_suppresses():
    ans = ("A FRIA is required here because you are a private entity providing "
           "public services within the meaning of Article 27(1).")
    _, notes = flag_fria_overapplication(ans)
    assert not notes


def test_annex_iii_point_5_scope_suppresses():
    ans = ("Deployers of Annex III point 5(b) credit-scoring systems must perform "
           "a fundamental rights impact assessment under Article 27.")
    _, notes = flag_fria_overapplication(ans)
    assert not notes


def test_creditworthiness_scope_suppresses():
    ans = ("Because the system assesses creditworthiness, the deployer must "
           "perform a fundamental rights impact assessment (Article 27).")
    _, notes = flag_fria_overapplication(ans)
    assert not notes


def test_explicit_not_all_deployers_hedge_suppresses():
    ans = ("A fundamental rights impact assessment is required, but not all "
           "deployers fall within Article 27.")
    _, notes = flag_fria_overapplication(ans)
    assert not notes


# ── silent: assertion is hedged/negated on its own line ──────────────────────

def test_conditional_line_not_flagged():
    ans = ("A fundamental rights impact assessment is required only where the "
           "deployer meets the Article 27 criteria.")
    _, notes = flag_fria_overapplication(ans)
    assert not notes


def test_negated_line_not_flagged():
    ans = "No fundamental rights impact assessment is required for this private deployer."
    _, notes = flag_fria_overapplication(ans)
    assert not notes


def test_if_required_conditional_not_flagged():
    # HQ_001 regression: "Conduct a FRIA if required" expressly conditions the
    # duty; it is not an unconditional over-application.
    ans = "- Conduct a fundamental rights impact assessment (FRIA) if required (**Article 27(1)**)."
    _, notes = flag_fria_overapplication(ans)
    assert not notes


def test_if_applicable_and_dpia_verb_not_flagged():
    # HQ_037 regression: the verb governs the DPIA, and the FRIA is qualified
    # "(if applicable)" — a well-scoped answer must not be flagged.
    ans = "Conduct a DPIA if required and cross-reference it in the FRIA (if applicable)."
    _, notes = flag_fria_overapplication(ans)
    assert not notes


def test_point_5d_is_not_the_frIA_scope():
    # HQ_004 regression: "Annex III, point 5(d)" is a different use case, not the
    # 5(b)/(c) FRIA scope — it must NOT suppress an otherwise over-broad claim.
    ans = ("For the system under Annex III, point 5(d), the deployer must conduct "
           "a fundamental rights impact assessment under Article 27.")
    _, notes = flag_fria_overapplication(ans)
    assert notes


def test_distant_public_authority_mention_does_not_suppress():
    # HQ_008 regression: a "Public Authorities Only" registration heading far from
    # the FRIA claim must not suppress it (answer-wide suppression was too coarse).
    ans = (
        "You must conduct a fundamental rights impact assessment (FRIA) under Article 27.\n"
        + "\n" * 6
        + "#### Registration (Public Authorities Only)\n- Register the system under Article 49."
    )
    _, notes = flag_fria_overapplication(ans)
    assert notes


def test_nearby_public_body_scope_suppresses_within_window():
    # The mirror of the above: a public-body scope statement within ±2 lines of
    # the claim DOES suppress.
    ans = ("Because you are a body governed by public law:\n\n"
           "- you must conduct a fundamental rights impact assessment under Article 27.")
    _, notes = flag_fria_overapplication(ans)
    assert not notes


# ── silent: no FRIA content at all ───────────────────────────────────────────

def test_no_fria_mention_is_untouched():
    ans = "The deployer must register the system under Article 49 and monitor it under Article 26."
    out, notes = flag_fria_overapplication(ans)
    assert not notes and out == ans


def test_empty_answer():
    assert flag_fria_overapplication("") == ("", [])


# ── idempotent: the appended caveat itself names the scope, so a second pass
#    finds the scope present and does not double-append ────────────────────────

def test_second_pass_does_not_double_append():
    ans = "The deployer must perform a fundamental rights impact assessment under Article 27."
    once, n1 = flag_fria_overapplication(ans)
    twice, n2 = flag_fria_overapplication(once)
    assert n1 and not n2
    assert twice.count(_CAVEAT) == 1


# ── the flag does not fire on a heading/blockquote-only FRIA mention ──────────

def test_heading_only_mention_not_flagged():
    ans = "## Fundamental rights impact assessment\n\nSee the analysis of Article 26 duties below."
    _, notes = flag_fria_overapplication(ans)
    assert not notes
