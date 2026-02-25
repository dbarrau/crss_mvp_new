# crss_mvp/crss/ingestion/scrape.py

from playwright.sync_api import sync_playwright
from pathlib import Path

def scrape_document(celex: str, lang: str, out_dir: Path) -> Path:
    """
    Scrapes an HTML document from EUR-Lex using its CELEX identifier.

    This function uses Playwright to navigate to the EUR-Lex portal,
    waits for the network to become idle to ensure the content is fully loaded,
    and saves the raw HTML to the specified directory.

    Args:
        celex: The unique CELEX identifier of the EU document (e.g., '32024R0590').
        lang: The language code for the document (e.g., 'EN', 'ES', 'FR').
        out_dir: A pathlib.Path object pointing to the directory where
            the 'raw.html' file should be saved.

    Returns:
        Path: The path to the newly created 'raw.html' file.

    Raises:
        playwright.errors.Error: If the browser fails to launch or the
            page fails to load.
        OSError: If the output directory is not writable.
    """
    url = f"https://eur-lex.europa.eu/legal-content/{lang}/TXT/HTML/?uri=CELEX:{celex}"

    # in case we later scrape xml files
    xml_url = f"https://eur-lex.europa.eu/legal-content/{lang}/TXT/XML/?uri=CELEX:{celex}"

    out_file = out_dir / "raw.html"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")

        html = page.content()
        out_file.write_text(html, encoding="utf-8")

        browser.close()

    return out_file
