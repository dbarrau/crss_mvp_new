# crss/ingestion/run_pipeline.py

"""
Pipeline Orchestrator
=====================

This module serves as the primary entry point for the CRSS data ingestion
engine. It coordinates the execution of scraping and parsing tasks,
ensuring that raw regulatory data is correctly fetched, stored, and
transformed into the unified Knowledge Graph format.

The module handles:
1.  Command-line interface (CLI) processing.
2.  Dynamic directory creation for data persistence.
3.  Sequence management between :mod:`.scrape` and :mod:`.parse`.

Supports two document families:

- **EUR-Lex regulations** (CELEX IDs like ``32017R0745``): scraped from
  EUR-Lex as HTML and parsed into ``parsed.json``.
- **Guidance documents** (MDCG IDs like ``MDCG_2020_3`` and AI Office IDs
  like ``AI_OFFICE_2025_TRANSPARENCY_ART50``): parsed from PDFs via
  LlamaParse v2 into clean markdown, using a per-doc parse profile.

Attributes:
    DEFAULT_DOC (str): The fallback document identifier (MDR 2017/745) used
        if none is provided via CLI.
    DEFAULT_LANG (str): The fallback language code ('EN') used if none
        is provided via CLI.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .scrape.scrape import scrape_document
from .parse.dispatcher import parse_document
from domain.legislation_catalog import LEGISLATION
from domain.guidance_catalog import GUIDANCE_DOCUMENTS, get_parse_profile

# Load .env from project root (needed for LLAMA_CLOUD_API_KEY, etc.)
_env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_env_path, override=False)

DEFAULT_DOC = "32017R0745"
DEFAULT_LANG = "EN"

# Backward-compatible alias
DEFAULT_CELEX = DEFAULT_DOC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── EUR-Lex regulation pipeline ───────────────────────────────────────────

def _run_legislation(celex: str, lang: str) -> Optional[Path]:
    """Scrape + parse an EUR-Lex legislative act (existing flow)."""
    base_dir = Path(__file__).resolve().parents[1]
    reg_dir = base_dir / "data" / "legislation" / celex / lang

    reg_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = reg_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    scrape_celex = LEGISLATION[celex].get("source_celex", celex)

    # The preamble supplement must never be mistaken for the main document
    # (e.g. after `rm raw.html` in the re-scrape flow).
    html_candidates = sorted(
        f for f in raw_dir.glob("*.html") if f.name != "raw_preamble.html"
    )
    if html_candidates:
        html_file = html_candidates[0]
        logger.info("Using existing HTML: %s", html_file)
    else:
        try:
            html_file = scrape_document(scrape_celex, lang, raw_dir)
            logger.info("Scraped HTML to: %s", html_file)
        except Exception as e:
            logger.exception("Scraping failed for %s %s: %s", celex, lang, e)
            return None

    # Consolidated EUR-Lex texts (CONSLEG) omit the preamble entirely — the
    # recitals exist only in the *original* act, whose CELEX is this catalog
    # entry's canonical key. Fetch it once as a preamble supplement so the
    # parser can graft the recitals into the consolidated body (GDPR: 173,
    # MDR/IVDR: 101 each). Best-effort: parsing proceeds without recitals if
    # the fetch fails.
    if scrape_celex != celex:
        preamble_file = raw_dir / "raw_preamble.html"
        if not preamble_file.exists():
            try:
                scrape_document(celex, lang, raw_dir, filename="raw_preamble.html")
                logger.info("Scraped preamble supplement to: %s", preamble_file)
            except Exception as e:
                logger.warning(
                    "Preamble supplement scrape failed for %s (%s) — "
                    "parsing without recitals.", celex, e,
                )

    try:
        json_file = parse_document(html_file, lang, celex, reg_dir)
    except Exception as e:
        logger.exception("Parsing failed for %s: %s", html_file, e)
        return None

    logger.info("Pipeline completed. HTML: %s, JSON: %s", html_file, json_file)
    return json_file


# ── Guidance pipeline (MDCG, AI Office, …) ────────────────────────────────

def _run_guidance(doc_id: str, lang: str) -> Optional[Path]:
    """Parse a guidance PDF via LlamaParse v2, then structure into parsed.json.

    Family-agnostic: the LlamaParse prompt + post-processing come from the
    per-doc parse profile (``mdcg``, ``ai_office``, …), so a new guidance
    publisher needs only a catalog entry, not a new pipeline.
    """
    from .parse.guidance.mdcg_parser import parse_guidance_pdf
    from .parse.guidance.mdcg_structurer import write_parsed_json
    from .parse.guidance.profiles import get_profile

    meta = GUIDANCE_DOCUMENTS[doc_id]
    profile = get_profile(get_parse_profile(doc_id))
    base_dir = Path(__file__).resolve().parents[1]

    doc_dir = base_dir / "data" / "guidance" / doc_id / lang
    raw_dir = doc_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pdf_filename = meta["pdf_filename"]
    pdf_path = raw_dir / pdf_filename

    if not pdf_path.exists():
        # Try to download if download_url is available
        download_url = meta.get("download_url")
        if download_url:
            from .scrape.download_guidance import download_guidance_pdf
            logger.info("PDF not found locally. Attempting download...")
            pdf_path = download_guidance_pdf(download_url, pdf_filename, raw_dir)
            if not pdf_path:
                logger.error("Download failed for %s", doc_id)
                return None
        else:
            logger.error(
                "PDF not found at %s and no download_url in catalog. "
                "Either place the PDF there manually or add a download_url to "
                "the guidance catalog entry for %s.",
                pdf_path,
                doc_id,
            )
            return None

    # Skip re-parsing if clean markdown already exists
    clean_md = doc_dir / f"{pdf_path.stem}_clean.md"
    if clean_md.exists():
        logger.info("Clean markdown already exists: %s (delete to re-parse)", clean_md)
    else:
        try:
            result = parse_guidance_pdf(
                pdf_path=pdf_path, output_dir=doc_dir, profile=profile
            )
        except Exception as e:
            logger.exception("Guidance parsing failed for %s: %s", doc_id, e)
            return None

        output_files = result.get("output_files", {})
        clean_md = Path(output_files.get("clean_markdown", str(clean_md)))

    # Structure clean markdown into parsed.json
    parsed_json = doc_dir / "parsed.json"
    try:
        write_parsed_json(
            md_path=clean_md,
            doc_id=doc_id,
            doc_name=meta["name"],
            lang=lang,
            output_path=parsed_json,
            scheme=profile.section_scheme,
        )
        logger.info("Structured %s → %s", clean_md.name, parsed_json)
    except Exception as e:
        logger.exception("Structuring failed for %s: %s", doc_id, e)
        return None

    logger.info(
        "Guidance pipeline completed for %s (profile=%s). Outputs in %s",
        doc_id, profile.name, doc_dir,
    )
    return parsed_json


# ── Public entry point ────────────────────────────────────────────────────

def run(doc_id: str, lang: str) -> Optional[Path]:
    """
    Execute the full data pipeline for a regulation or MDCG guidance document.

    Accepts both EUR-Lex CELEX identifiers (e.g. ``32017R0745``) and
    guidance document IDs (MDCG or AI Office, e.g. ``MDCG_2020_3`` /
    ``AI_OFFICE_2025_TRANSPARENCY_ART50``), routing to the appropriate
    pipeline automatically.

    :param doc_id: Document identifier (CELEX ID or guidance doc ID).
    :param lang: ISO language code (e.g. ``EN``).
    """
    if doc_id in GUIDANCE_DOCUMENTS:
        return _run_guidance(doc_id, lang)

    if doc_id in LEGISLATION:
        return _run_legislation(doc_id, lang)

    logger.error(
        "Unknown document identifier: %s. "
        "Must be a CELEX ID from legislation_catalog or a guidance ID from "
        "the guidance catalog (MDCG / AI Office).",
        doc_id,
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CRSS ingestion pipeline — regulations and guidance (MDCG / AI Office)",
    )
    parser.add_argument(
        "--doc",
        default=DEFAULT_DOC,
        help="Document identifier — CELEX ID (e.g. 32017R0745) or guidance doc ID "
             "(e.g. MDCG_2020_3, AI_OFFICE_2025_TRANSPARENCY_ART50)",
    )
    # Deprecated alias kept for backward compatibility
    parser.add_argument("--celex", dest="doc", help=argparse.SUPPRESS)
    parser.add_argument("--lang", default=DEFAULT_LANG)

    args = parser.parse_args()

    run(args.doc, args.lang)
