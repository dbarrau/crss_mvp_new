#!/usr/bin/env python
"""Audit for paraphrase-misattribution: a claim attached to "Article N(p)" whose
substance actually lives in a DIFFERENT paragraph of the same article.

The runtime faithfulness stage (application/_faithfulness.py) catches displaced
*verbatim quotes* — but only at article-family granularity, and only for text in
quote marks. A *paraphrased* claim that names a specific sub-paragraph escapes on
two counts: it is never extracted as a quote, and even if it were, `_base_ref_family`
collapses "Article 33(4)" and "Article 33(5)" to one family, so intra-article
displacement is invisible. The observed failure (Sep 2026): CRSS wrote "Article
33(4) specifies that Eudamed's public parts must be presented in a user-friendly
and easily-searchable format" — but 33(4) is the data-entry paragraph; the
user-friendly duty is Article 33(5), subparagraph 2. The root cause was a context
render that erased the paragraph boundary (fixed in _context._render_subtree,
commit d4020cc); this sweep is the standing detector so the class stays measured.

What it does: for every "Article N(p)" reference in an answer, it extracts the
surrounding claim, then scores that claim (distinctive 2-3 word shingles, graph =
ground truth) against the CITED paragraph and against every SIBLING paragraph of
the same article. It flags a candidate only when the cited paragraph matches
poorly AND exactly the pattern of a sibling matching clearly better — the same
"positive displacement proof" the runtime guard uses.

    python scripts/audit_paraphrase_attribution.py --run eval/runs/artifact_v1.json
    python scripts/audit_paraphrase_attribution.py --run eval/runs/quality_v9_panel.json --verbose

IMPORTANT — this is a REVIEW AID, not an auto-repair, and deliberately so. Lexical
grounding cannot tell "the obligation lives here" from "this paragraph name-drops
the topic": enumeration/QMS paragraphs (e.g. MDR 10(9) lists every quality-system
process; AI Act 53/55 cross-reference each other) lexically dominate their
siblings and produce false flags. On the 40-answer eval corpus (936 adjudications)
every above-threshold flag was, on manual review, a correct citation. So an
auto-repointer built on this signal would corrupt correct citations far more often
than it fixes a real (rare, tail) defect. Every flag here MUST be adjudicated by a
human against the printed graph text before any action. Fix the *rendering* that
misleads the model (the root cause), not the citation after the fact.

Requires Neo4j (the loaded graph) and MDR/IVDR/AI Act/GDPR ingested.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from neo4j import GraphDatabase

from domain.legislation_catalog import LEGISLATION

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))

# Regulations this sweep adjudicates (operative articles/paragraphs live here).
_ADJUDICABLE = {"32017R0745", "32017R0746", "32024R1689", "32016R0679"}
# name/number substrings -> celex, for detecting which regulation an answer is about
_NAME_TO_CELEX: dict[str, str] = {}
for _celex, _meta in LEGISLATION.items():
    if _celex not in _ADJUDICABLE:
        continue
    if _meta.get("number"):
        _NAME_TO_CELEX[_meta["number"]] = _celex           # "2017/745"
for _k, _c in {"MDR": "32017R0745", "IVDR": "32017R0746",
               "AI Act": "32024R1689", "GDPR": "32016R0679"}.items():
    _NAME_TO_CELEX[_k] = _c

_ID_RE = re.compile(r"^(\d{5}[A-Z]\d{4})_(\d{3})\.(\d{3})(?:_.*)?$")
_REF_RE = re.compile(r"Article\s+(\d+)\((\d+)\)")
_WS = re.compile(r"\s+")

# Default displacement-proof thresholds (conservative; tuned so the only signal
# that survives is a paragraph that plainly does NOT contain the claim while a
# single sibling plainly does). Loosening these floods the output with the
# enumeration-paragraph false positives described in the module docstring.
_CITED_MAX = 0.15      # cited paragraph must match the claim POORLY
_SIB_MIN = 0.30        # a sibling must match the claim CLEARLY
_DELTA_MIN = 0.20      # and win by this margin

_STOP = set(
    "the a an and or of to in on for with by shall be is are as that this which "
    "such at from into under within not no all any its their his her may must "
    "other where when who whom been being have has had will each per point "
    "article annex paragraph subparagraph referred pursuant accordance member "
    "states state commission relevant appropriate concerned".split()
)


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "").lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return _WS.sub(" ", t).strip()


def _shingles(t: str) -> set[tuple]:
    w = [x for x in _norm(t).split() if x not in _STOP and len(x) > 2]
    return set(zip(w, w[1:])) | set(zip(w, w[1:], w[2:]))


def _score(claim_sh: set[tuple], node_text: str) -> float:
    if not claim_sh:
        return 0.0
    return len(claim_sh & _shingles(node_text)) / len(claim_sh)


def _sentence_around(text: str, idx: int) -> str:
    lo = max(text.rfind(".", 0, idx), text.rfind("\n", 0, idx),
             text.rfind(":", 0, idx), text.rfind(";", 0, idx)) + 1
    hi = [p for p in (text.find(".", idx), text.find("\n", idx)) if p != -1]
    return text[lo:(min(hi) if hi else len(text))].strip()


def _load_article_paragraphs(driver, db: str | None) -> dict:
    """{(celex, art:int) -> {para:int -> combined paragraph+children text}}."""
    art_para: dict = defaultdict(lambda: defaultdict(list))
    with (driver.session(database=db) if db else driver.session()) as s:
        q = ("MATCH (c:Provision) WHERE c.celex IN $celexes AND c.text IS NOT NULL "
             "RETURN c.id AS id, c.text AS text")
        for r in s.run(q, celexes=list(_ADJUDICABLE)):
            m = _ID_RE.match(r["id"] or "")
            if not m:
                continue
            celex, art, para = m.group(1), int(m.group(2)), int(m.group(3))
            t = (r["text"] or "").strip()
            if t:
                art_para[(celex, art)][para].append(t)
    return art_para


def _answers_from_run(path: Path):
    data = json.loads(path.read_text())
    rows = data.get("results") or data.get("cases") or []
    for r in rows:
        ans = r.get("final") or r.get("answer") or ""
        if ans:
            yield r.get("id", "?"), r.get("question", ""), ans


def audit(run_path: Path, art_para: dict, *, verbose: bool = False) -> list[dict]:
    flags: list[dict] = []
    examined = 0
    for cid, question, ans in _answers_from_run(run_path):
        cand = {c for name, c in _NAME_TO_CELEX.items() if name in ans or name in question}
        cand = cand or set(_ADJUDICABLE)
        for m in _REF_RE.finditer(ans):
            art, para = int(m.group(1)), int(m.group(2))
            claim = _sentence_around(ans, m.start())
            csh = _shingles(_REF_RE.sub("", claim))
            if len(csh) < 4:
                continue
            for celex in cand:
                paras = art_para.get((celex, art))
                if not paras or para not in paras:
                    continue
                examined += 1
                cited = _score(csh, " ".join(paras[para]))
                sib = [(p, _score(csh, " ".join(t))) for p, t in paras.items() if p != para]
                if not sib:
                    continue
                bp, bs = max(sib, key=lambda x: x[1])
                if cited <= _CITED_MAX and bs >= _SIB_MIN and bs - cited >= _DELTA_MIN:
                    flags.append(dict(case=cid, celex=celex, art=art, cited=para,
                                      cited_score=round(cited, 2), better=bp,
                                      better_score=round(bs, 2), claim=claim[:240],
                                      cited_text=" ".join(paras[para])[:240],
                                      better_text=" ".join(paras[bp])[:240]))
    print(f"# {run_path.name}: {examined} sub-paragraph attributions adjudicated, "
          f"{len(flags)} candidate(s) for review\n")
    for f in flags:
        print(f"[{f['case']}] {f['celex']} Article {f['art']}({f['cited']}) "
              f"score={f['cited_score']}  →  better: ({f['better']}) score={f['better_score']}")
        print(f"    claim : {f['claim']}")
        if verbose:
            print(f"    cited : {f['cited_text']}")
            print(f"    better: {f['better_text']}")
        print()
    if flags:
        print("Review each against the printed graph text — lexical flags over-report "
              "on enumeration/cross-reference paragraphs (see module docstring). Do NOT "
              "auto-repoint.")
    return flags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True,
                    help="eval-run JSON with results[].{final|answer} (e.g. eval/runs/artifact_v1.json)")
    ap.add_argument("--verbose", action="store_true",
                    help="also print the cited and better paragraph texts")
    args = ap.parse_args()
    if not args.run.exists():
        print(f"run file not found: {args.run}", file=sys.stderr)
        return 2
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    )
    try:
        art_para = _load_article_paragraphs(driver, os.environ.get("NEO4J_DATABASE"))
    finally:
        driver.close()
    flags = audit(args.run, art_para, verbose=args.verbose)
    # Non-zero exit signals "candidates present" for CI wiring, but this is a
    # review aid: a flag is a prompt to look, not a proven defect.
    return 1 if flags else 0


if __name__ == "__main__":
    raise SystemExit(main())
