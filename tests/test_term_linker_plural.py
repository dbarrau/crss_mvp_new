"""USES_TERM linking must match plural occurrences (canonicalization/term_linker).

A defined term is stored singular ("information society service") but the text
overwhelmingly uses the plural ("information society services"). The matcher must
link the plural to the singular term — otherwise a predominantly-plural term gets
zero USES_TERM edges and is silently disconnected from every provision that uses
it (found by scripts/audit_defined_term_connectivity.py). The subtlety: the
plural ``s?`` sits OUTSIDE a capturing group so ``re.findall`` yields the singular
term (the dict key), not the plural matched text.
"""
from canonicalization.term_linker import _build_term_regex, _find_uses


def test_plural_occurrence_links_to_the_singular_term():
    terms = [{"id": "t1", "term": "information society service", "src_prov_id": "def"}]
    rx = _build_term_regex(terms)
    provs = [{"id": "p1", "text": "the offering of information society services to a child"}]
    assert _find_uses(provs, terms, rx) == [{"prov_id": "p1", "term_id": "t1"}]


def test_singular_occurrence_still_links():
    terms = [{"id": "t1", "term": "specimen receptacle", "src_prov_id": "def"}]
    rx = _build_term_regex(terms)
    provs = [{"id": "p1", "text": "a specimen receptacle is a device"}]
    assert _find_uses(provs, terms, rx) == [{"prov_id": "p1", "term_id": "t1"}]


def test_defining_provision_is_not_linked_to_itself():
    terms = [{"id": "t1", "term": "importer", "src_prov_id": "pdef"}]
    rx = _build_term_regex(terms)
    provs = [{"id": "pdef", "text": "'importer' means a natural or legal person"}]
    assert _find_uses(provs, terms, rx) == []


def test_word_boundary_prevents_substring_false_match():
    # "importers-club" must not match, nor a longer word ending in the term.
    terms = [{"id": "t1", "term": "importer", "src_prov_id": "def"}]
    rx = _build_term_regex(terms)
    provs = [{"id": "p1", "text": "the transporter arranged shipping"}]  # 'transporter' ⊅ 'importer'
    assert _find_uses(provs, terms, rx) == []


def test_regex_matches_both_forms():
    rx = _build_term_regex([{"term": "authorised representative"}])
    assert rx.search("the authorised representatives shall")      # plural
    assert rx.search("an authorised representative is appointed")  # singular
