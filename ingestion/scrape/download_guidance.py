"""Download guidance PDFs from the EC Health portal.

Unlike EUR-Lex regulations which require Playwright for JavaScript rendering,
EC Health guidance documents are direct PDF downloads that can be fetched with
simple HTTP requests.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def download_guidance_pdf(
    download_url: str,
    pdf_filename: str,
    out_dir: Path,
) -> Optional[Path]:
    """
    Download a guidance PDF from the EC Health portal.

    Args:
        download_url: Full download URL from EC Health portal
            (e.g. https://health.ec.europa.eu/document/download/UUID_en?filename=...)
        pdf_filename: Destination filename (e.g. 'mdcg_2020_3_rev1.pdf')
        out_dir: Directory to save the PDF (typically data/guidance/{doc_id}/EN/raw/)

    Returns:
        Path to the downloaded PDF file, or None if download failed.

    Example:
        >>> url = "https://health.ec.europa.eu/document/download/800e8e87...?filename=mdcg_2020-3_en_1.pdf"
        >>> download_guidance_pdf(url, "mdcg_2020_3_rev1.pdf", Path("data/guidance/MDCG_2020_3/EN/raw"))
    """
    out_file = out_dir / pdf_filename

    if out_file.exists():
        logger.info("PDF already exists: %s", out_file)
        return out_file

    logger.info("Downloading from: %s", download_url)

    try:
        # Use stream=True for large files to avoid loading entire file into memory
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        # Write in chunks
        with open(out_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)

        logger.info("Downloaded PDF to: %s (%.1f KB)", out_file, out_file.stat().st_size / 1024)
        return out_file

    except requests.exceptions.RequestException as e:
        logger.error("Failed to download %s: %s", download_url, e)
        return None
