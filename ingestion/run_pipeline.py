# crss_mvp/crss/ingestion/run_pipeline.py

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

Attributes:
    DEFAULT_CELEX (str): The fallback CELEX ID (MDR 2017/745) used if none
        is provided via CLI.
    DEFAULT_LANG (str): The fallback language code ('EN') used if none
        is provided via CLI.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from .scrape.scrape import scrape_document
from .parse import parse_document
from domain.regulations_catalog import REGULATIONS

DEFAULT_CELEX = "32017R0745"
DEFAULT_LANG = "EN"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(celex: str, lang: str) -> Optional[Path]:
    """
    Executes the full data pipeline: scraping, saving, and parsing an EU regulation.

    This function establishes the project directory structure, creates the
    necessary nested folders for the specific CELEX ID and language, and
    sequentially calls the scraper and parser modules.

    The directory structure created follows this pattern:
    `regulations/<celex>/<lang>/`

    :param celex: The unique CELEX identifier of the EU document (e.g., '32017R0745').
    :param lang: The ISO language code (e.g., 'EN', 'ES').

    Note:
        This function uses relative path resolution based on the location of
        the script file to ensure portability across different installation
        paths.

    Example:
        >>> run("32017R0745", "EN")
        Pipeline completed:
        HTML: .../regulations/32017R0745/EN/raw.html
        JSON: .../regulations/32017R0745/EN/graph.json
    """
    # Validate CELEX against catalog before creating any directories
    if celex not in REGULATIONS:
        logger.error("Unknown CELEX identifier: %s. Aborting pipeline.", celex)
        return None

    base_dir = Path(__file__).resolve().parents[1]
    # use data/regulations per project conventions
    reg_dir = base_dir / "data" / "regulations" / celex / lang

    # create directories only after CELEX validated
    reg_dir.mkdir(parents=True, exist_ok=True)

    # raw HTML directory
    raw_dir = reg_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Discover existing HTML files first
    html_candidates = sorted(raw_dir.glob("*.html"))
    if html_candidates:
        html_file = html_candidates[0]
        logger.info("Using existing HTML: %s", html_file)
    else:
        try:
            html_file = scrape_document(celex, lang, raw_dir)
            logger.info("Scraped HTML to: %s", html_file)
        except Exception as e:
            logger.exception("Scraping failed for %s %s: %s", celex, lang, e)
            return None

    # Parse and write parsed output under reg_dir
    try:
        json_file = parse_document(html_file, lang, celex, reg_dir)
    except Exception as e:
        logger.exception("Parsing failed for %s: %s", html_file, e)
        return None

    logger.info("Pipeline completed. HTML: %s, JSON: %s", html_file, json_file)
    return json_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--celex", default=DEFAULT_CELEX)
    parser.add_argument("--lang", default=DEFAULT_LANG)

    args = parser.parse_args()

    run(args.celex, args.lang)
