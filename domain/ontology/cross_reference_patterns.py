"""
Cross-reference patterns for EU legislative texts (Regulations / Directives).

Extracted from and validated against: EU AI Act (CELEX 32024R1689, EN).
Also applicable to: EU MDR (32017R0745), EU IVDR (32017R0746), and similar
instruments following the standard EUR-Lex drafting conventions.

Four categories are defined, each as a compiled ``re.Pattern`` with named
capture groups.  The module also exposes ``ALL_PATTERNS``, an ordered mapping
from category name to pattern, suitable for pipeline use.

-----------------------------------------------------------------------
CAVEATS FOR GRAPH-BUILDING PIPELINES
-----------------------------------------------------------------------
1. **Footnote noise** – parsed text often contains inline citation markers
   such as ``( 10 )`` (with spaces inside) immediately after a reference.
   Strip these first — requires at least one space on each side of the digit
   so that definition point numbers like ``(49)`` are not accidentally stripped:
       FOOTNOTE_MARKER = re.compile(r'\\s*\\(\\s+\\d+\\s+\\)')
2. **Priority order** – apply RELATIVE_REF before EXPLICIT_REF; a string
   like "paragraph 3, second subparagraph" is unambiguously relative in
   context but also matches the explicit article branch partially.
3. **Chained references** – "Article 4, point (14) of Regulation (EU)
   2016/679" yields both an EXPLICIT_REF match AND an EXTERNAL_REF match
   within the same span.  Your linker should detect the ``of`` connector
   and chain them.
4. **Treaty references** – "Article 4(2) TEU" and "Article 16 TFEU" are
   structural-law external references.  Extend EXTERNAL_REF with the
   optional ``(?P<treaty>TEU|TFEU|Charter)`` branch to capture these, or
   use a separate dedicated pattern if treaty refs must be typed differently
   in the graph.
-----------------------------------------------------------------------
"""

import re

# ---------------------------------------------------------------------------
# Category 1 — Explicit References
# ---------------------------------------------------------------------------
EXPLICIT_REF = re.compile(
    r"""
    (?:
      # ── "Section X of Annex Y" (reversed order) ─────────────────────────
      Section\s+(?P<sec_of_annex>[A-Z]|\d+)
      (?:,?\s*points?\s+(?P<sec_annex_pt>\d+(?:\([a-z]\))?))?  # optional point
      \s+of\s+Annex\s+(?P<annex_of_sec>[IVX]+|\d+)
    |
      # ── "point N of Annex Y" (reversed order) ──────────────────────────
      point\s+(?P<pt_of_annex>\d+(?:\([a-z]\))?)
      \s+of\s+Annex\s+(?P<annex_of_pt>[IVX]+|\d+)
    |
      # ── Annex branch ──────────────────────────────────────────────────────
      Annex\s+(?P<annex>[IVX]+|\d+)
      (?:,?\s*Section\s+(?P<section>[A-Z]|\d+))?
      (?:,?\s*points?\s+(?P<annex_pt>\d+(?:\([a-z]\))?))?  # "point 2" or "points 1"
    |
      # ── Article branch ────────────────────────────────────────────────────
      Articles?\s+(?P<article>\d+)
      (?:\((?P<para>\d+)\))?
      (?:,?\s*(?P<ordinal>first|second|third|fourth|fifth)\s+subparagraph)?
      (?:,?\s*point\s+\((?P<point>[a-z0-9]+)\)(?:\((?P<subpoint>[a-z0-9]+)\))?)?
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)
"""
Matches explicit, self-contained cross-references to a specific Article or
Annex location, optionally qualified by paragraph, ordinal subparagraph, and
lettered/numbered point.

Named groups
------------
sec_of_annex : Section id in "Section X of Annex Y" form, e.g. "A", "1"
sec_annex_pt : Optional point after section in reversed form
annex_of_sec : Annex roman/number in "Section X of Annex Y" form
pt_of_annex  : Point number in "point N of Annex Y" form
annex_of_pt  : Annex roman/number in "point N of Annex Y" form
annex       : Roman or Arabic numeral of the Annex, e.g. "III", "I", "IV"
section     : Section within the Annex, e.g. "A", "B", "1"
annex_pt    : Point number inside the Annex, e.g. "8", "1(a)"
article     : Article number, e.g. "5", "9", "26"
para        : Paragraph number inside the article, e.g. "1", "2"
ordinal     : Ordinal qualifier of a subparagraph, e.g. "first", "second"
point       : Lettered/numbered point, e.g. "g", "h", "14"
subpoint    : Second-level point, e.g. "iii"

Test strings (sourced from 32024R1689)
---------------------------------------
"Article 5(1), first subparagraph, point (g)"
    → article='5', para='1', ordinal='first', point='g'
"Article 9(2), point (g)"
    → article='9', para='2', point='g'
"Article 4, point (14)"
    → article='4', point='14'
"Article 16(6)"
    → article='16', para='6'
"Annex III"
    → annex='III'

Source locations in 32024R1689/EN/parsed.json
----------------------------------------------
- Recital (40), line 870  : "Article 5(1), first subparagraph, point (g)"
- Recital (14), line 506  : "Article 4, point (14) of Regulation (EU) 2016/679"
- Recital (136), line 2214: "Article 16(6) of Regulation (EU) 2022/2065"
- Art. 6(2), line 4109    : "Annex III"
"""


# ---------------------------------------------------------------------------
# Category 2 — Range References
# ---------------------------------------------------------------------------
RANGE_REF = re.compile(
    r"""
    (?P<kind>Articles?|paragraphs?|points?)\s+
    (?P<start>\d+(?:\(\d+\))?)               # e.g. "102"  or  "5(2)"
    (?:
      \s+to\s+
      (?P<end>\d+(?:\(\d+\))?|\(\d+\))       # e.g. "109", "5(6)", or bare "(6)"
    |
      (?P<middle>(?:\s*,\s*\d+(?:\(\d+\))?)*)  # optional middle terms: ", 14, 15"
      \s+and\s+
      (?P<last>\d+(?:\(\d+\))?)              # final term of enumeration
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)
"""
Matches range or enumerated references to consecutive or co-cited provisions.

Two sub-forms are recognised:
  • Span:        "Articles 102 to 109", "Article 5(2) to (6)"
  • Enumeration: "Articles 5 and 6",    "paragraphs 6 and 7"

Named groups
------------
kind   : Provision type keyword — "Articles", "Article", "paragraphs", "points"
start  : First number, optionally with paragraph qualifier, e.g. "102", "5(2)"
end    : Last number of a span range, e.g. "109", "(6)"
middle : Comma-separated middle terms in an enumeration, e.g. ", 14, 15"
last   : Final number of an enumerated list (mutually exclusive with 'end')

Test strings (sourced from 32024R1689)
---------------------------------------
"Articles 102 to 109"
    → kind='Articles', start='102', end='109'
"Article 5(2) to (6)"
    → kind='Article', start='5(2)', end='(6)'
"paragraphs 2 to 5"
    → kind='paragraphs', start='2', end='5'
"Articles 5 and 6"
    → kind='Articles', start='5', last='6'
"paragraphs 6 and 7"
    → kind='paragraphs', start='6', last='7'
"Articles 13, 14 and 16"
    → kind='Articles', start='13', middle=', 14', last='16'
"Articles 8, 9, 10 and 11"
    → kind='Articles', start='8', middle=', 9, 10', last='11'

Source locations in 32024R1689/EN/parsed.json
----------------------------------------------
- Art. 2(3),     line 3240: "Articles 102 to 109"
- Recital (40),  line 870 : "Article 5(2) to (6)"
- Art. 10(1),    line 5180: "paragraphs 2 to 5"
- Recital (121), line 2004: "Articles 5 and 6"
- Art. 6(8),     line 4288: "paragraphs 6 and 7"
"""


# ---------------------------------------------------------------------------
# Category 3 — Relative References
# ---------------------------------------------------------------------------
RELATIVE_REF = re.compile(
    r"""
    (?:
      # ── Anaphoric "this" without a number ─────────────────────────────
      # "this paragraph", "this subparagraph", "this Article"
      # Must appear as standalone phrases (not "this paragraph 3")
      (?P<this_kw>this)\s+(?P<this_target>paragraph|subparagraph|Article)
      (?!\s*\d)              # negative lookahead: no digit follows
    |
      # ── Anaphoric: "of this Article" / "of this paragraph" ────────────
      of\s+this\s+(?P<of_this>Article|paragraph)
      (?!\s*\d)
    |
      # ── Ordinal subparagraph: "the first subparagraph [of paragraph N]" ──
      (?P<the>the)\s+
      (?P<ordinal>first|second|third|fourth|fifth)\s+subparagraph
      (?:\s+of\s+(?:this\s+)?
        (?:paragraph\s+(?P<of_para>\d+)|Article)
      )?
    |
      # ── Point with locator: "point (X)(Y) of [the Nth subparagraph / ..." ─
      point\s+\((?P<pt_letter>[a-z])\)(?:\((?P<pt_sub>[a-z0-9]+)\))?
      \s+
      (?:
          of\s+(?P<of_clause>
              (?:the\s+)?(?:first|second|third|fourth|fifth)\s+subparagraph
            | (?:this\s+)?(?:paragraph|Article)
          )
        | (?P<thereof>thereof)
      )
    |
      # ── Paragraph qualifier: "paragraph N[, ordinal subparagraph][, point (X)]" ─
      (?P<para_kw>paragraph)\s+(?P<para>\d+)
      (?:,\s*(?P<sub_ord>first|second|third|fourth|fifth)\s+subparagraph)?
      (?:,\s*point\s+\((?P<para_pt>[a-z])\))?
    |
      # ── Anaphoric locator: "referred to in / pursuant to / in accordance with ..." ─
      (?P<qualifier>referred\s+to\s+in|pursuant\s+to|in\s+accordance\s+with)
      \s+(?:this\s+)?(?P<ref_kw>paragraph|Article)\s+(?P<ref_num>\d+)
      (?:,\s*(?P<ref_sub_ord>first|second|third|fourth|fifth)\s+subparagraph)?
      (?:,\s*point\s+\((?P<ref_pt>[a-z0-9]+)\)(?:\((?P<ref_subpt>[a-z0-9]+)\))?)?
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)
"""
Matches positional or anaphoric references that locate a provision relative to
the current context, without repeating the full Article number.

Named groups
------------
this_kw   : Literal "this" — signals anaphoric self-reference
this_target : "paragraph", "subparagraph", or "Article" (no digit)
of_this   : "Article" or "paragraph" in "of this Article/paragraph"
the       : Literal "the" — signals ordinal-subparagraph form
ordinal   : Ordinal word — "first" … "fifth"
of_para   : Paragraph number when "of paragraph N" follows the ordinal
pt_letter : Primary letter of a point reference, e.g. "h", "g"
pt_sub    : Sub-level letter/numeral, e.g. "iii"
of_clause : Locator phrase after "of", e.g. "the first subparagraph"
thereof   : Literal "thereof" (back-reference)
para_kw   : Literal "paragraph"
para      : Paragraph number, e.g. "1", "3"
sub_ord   : Ordinal of a subparagraph within the paragraph
para_pt   : Point letter within the paragraph, e.g. "h"
qualifier : Anaphoric trigger — "referred to in", "pursuant to",
            "in accordance with"
ref_kw    : "paragraph" or "Article"
ref_num   : Number of the referred provision
ref_sub_ord : Ordinal subparagraph after anaphoric locator, e.g. "first"
ref_pt    : Point label after anaphoric locator, e.g. "h" or "49"
ref_subpt : Sub-point label after ref_pt, e.g. "c" from "point (49)(c)"

Test strings (sourced from 32024R1689)
---------------------------------------
"this paragraph"
    → this_kw='this', this_target='paragraph'
"this Article"
    → this_kw='this', this_target='Article'
"this subparagraph"
    → this_kw='this', this_target='subparagraph'
"of this Article"
    → of_this='Article'
"the first subparagraph"
    → the='the', ordinal='first'
"paragraph 1, first subparagraph, point (h)"
    → para_kw='paragraph', para='1', sub_ord='first', para_pt='h'
"point (h) of the first subparagraph"
    → pt_letter='h', of_clause='the first subparagraph'
"referred to in paragraph 3"
    → qualifier='referred to in', ref_kw='paragraph', ref_num='3'
"referred to in paragraph 1, first subparagraph, point (h)"
    → qualifier='referred to in', ref_kw='paragraph', ref_num='1',
      ref_sub_ord='first', ref_pt='h'
"paragraph 3, second subparagraph"
    → para_kw='paragraph', para='3', sub_ord='second'

Source locations in 32024R1689/EN/parsed.json
----------------------------------------------
- Art. 6(3),  line 4126: "The first subparagraph shall apply where..."
- Art. 5(1),  line 3490: "Point (h) of the first subparagraph is without..."
- Art. 5(2),  line 3838: "...paragraph 1, first subparagraph, point (h)..."
- Art. 5(3),  line 3891: "...paragraph 1, first subparagraph, point (h)..."
- Art. 6(6),  line 4254: "...paragraph 3, second subparagraph, of this Article..."
"""


# ---------------------------------------------------------------------------
# Category 4 — External References
# ---------------------------------------------------------------------------
EXTERNAL_REF = re.compile(
    r"""
    (?:(?P<institution>Council|Commission)\s+)?
    (?P<doc_type>Regulation|Directive|Decision)\s+
    (?:\((?P<series>EU|EC|EEC|Euratom)\)\s+)?
    (?:No\s+)?
    (?P<number>\d{1,4}/\d{1,4})
    (?:/(?P<suffix>EC|EU|EEA|EEC|Euratom))?
    """,
    re.VERBOSE | re.IGNORECASE,
)
"""
Matches citations to other EU legislative instruments — Regulations,
Directives, and Decisions — identified by their official CELEX-style number.

Named groups
------------
institution : Optional issuing institution, e.g. "Council"
doc_type    : Instrument type — "Regulation", "Directive", "Decision"
series      : EU family identifier in parentheses — "EU", "EC", "EEC",
              "Euratom"
number      : Year/sequence number in N/YYYY format, e.g. "2016/679",
              "765/2008", "300/2008"
suffix      : Suffix appended after a second slash, e.g. "EC", "EU", "EEC"

Test strings (sourced from 32024R1689)
---------------------------------------
"Regulation (EU) 2016/679"
    → doc_type='Regulation', series='EU', number='2016/679'
"Directive 2002/58/EC"
    → doc_type='Directive', number='2002/58', suffix='EC'
"Regulation (EC) No 765/2008"
    → doc_type='Regulation', series='EC', number='765/2008'
"Directive (EU) 2016/797"
    → doc_type='Directive', series='EU', number='2016/797'
"Directive 85/374/EEC"
    → doc_type='Directive', number='85/374', suffix='EEC'

Source locations in 32024R1689/EN/parsed.json
----------------------------------------------
- Recital (10), line 450 : "Regulation (EU) 2016/679", "Directive 2002/58/EC"
- Recital (9),  line 436 : "Regulation (EC) No 765/2008"
- Recital (49), line 996 : "Directive (EU) 2016/797", "Directive 85/374/EEC"
- Document,     line 11  : "Regulations (EC) No 300/2008, (EU) No 167/2013..."
"""


# ---------------------------------------------------------------------------
# Category 5 — Amendment Relationships
# ---------------------------------------------------------------------------
AMENDED_BY = re.compile(
    r"""
    (?P<base_type>Regulation|Directive|Decision)\s+
    (?:\((?P<base_series>EU|EC|EEC|Euratom)\)\s+)?
    (?:No\s+)?
    (?P<base_number>\d{1,4}/\d{1,4})
    (?:/(?P<base_suffix>EC|EU|EEA|EEC|Euratom))?
    \s+as\s+amended\s+by\s+
    (?P<amending_type>Regulation|Directive|Decision)\s+
    (?:\((?P<amending_series>EU|EC|EEC|Euratom)\)\s+)?
    (?:No\s+)?
    (?P<amending_number>\d{1,4}/\d{1,4})
    (?:/(?P<amending_suffix>EC|EU|EEA|EEC|Euratom))?
    """,
    re.VERBOSE | re.IGNORECASE,
)
"""
Matches "Directive X as amended by Regulation Y" constructs that establish
an amendment relationship between two EU legislative instruments.

Named groups
------------
base_type        : Instrument type of the base act
base_series      : EU family identifier of the base act
base_number      : CELEX-style number of the base act
base_suffix      : Suffix of the base act
amending_type    : Instrument type of the amending act
amending_series  : EU family identifier of the amending act
amending_number  : CELEX-style number of the amending act
amending_suffix  : Suffix of the amending act

Test strings
------------
"Directive 2001/83/EC as amended by Regulation (EU) 2017/745"
    → base_type='Directive', base_number='2001/83', base_suffix='EC',
      amending_type='Regulation', amending_series='EU', amending_number='2017/745'
"Regulation (EC) No 765/2008 as amended by Regulation (EU) 2019/1020"
    → base_type='Regulation', base_series='EC', base_number='765/2008',
      amending_type='Regulation', amending_series='EU', amending_number='2019/1020'
"""


# ---------------------------------------------------------------------------
# Footnote noise stripper (apply before matching)
# ---------------------------------------------------------------------------
FOOTNOTE_MARKER = re.compile(r"\s*\(\s+\d+\s+\)")
"""
Strips inline footnote citation markers like ``( 10 )`` or ``(7)`` that appear
in EUR-Lex parsed text immediately after cross-references, e.g.:

    "Regulation (EU) 2019/1020 of the European Parliament ( 9 ) ,..."

Usage::

    clean_text = FOOTNOTE_MARKER.sub("", raw_text)
"""


# ---------------------------------------------------------------------------
# Convenience registry (priority order: relative → explicit → range → external)
# ---------------------------------------------------------------------------
ALL_PATTERNS: dict[str, re.Pattern] = {
    "relative": RELATIVE_REF,
    "explicit": EXPLICIT_REF,
    "range": RANGE_REF,
    "external": EXTERNAL_REF,
    "amended_by": AMENDED_BY,
}
"""
Ordered mapping of category name → compiled pattern.

Apply in the order given: RELATIVE_REF is checked first to avoid
ambiguous partial matches by EXPLICIT_REF on constructs like
"paragraph 3, second subparagraph".

Usage example::

    import re
    from domain.ontology.cross_reference_patterns import ALL_PATTERNS, FOOTNOTE_MARKER

    def extract_refs(text: str) -> list[dict]:
        text = FOOTNOTE_MARKER.sub("", text)
        results = []
        for category, pattern in ALL_PATTERNS.items():
            for m in pattern.finditer(text):
                results.append({
                    "category": category,
                    "span": m.span(),
                    "match": m.group(),
                    "groups": {k: v for k, v in m.groupdict().items() if v},
                })
        return results
"""
