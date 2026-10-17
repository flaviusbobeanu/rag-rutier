"""Download the consolidated law texts from legislatie.just.ro into data/raw/.

Each consolidated version of a law has its own page ID on the Portal Legislativ.
The page contains a link to a "Forma printabilă" (printable form): clean HTML where
articles, paragraphs and amendment notes are marked with CSS classes (S_ART, S_ALN,
S_NTA, ...). We save that HTML as-is; parsing happens in parse_html.py.

The list of laws and their page IDs is in sources.py.
"""

import re
import time
import urllib.request

from src.config import settings
from src.ingest.sources import DOCUMENTS

BASE_URL = "https://legislatie.just.ro"


def fetch(url: str) -> str:
    """Download a URL and return its body as text."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (rag-rutier student project)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def download_document(file_stem: str, page_id: int) -> None:
    """Find the printable-form link on a document page and save that HTML to data/raw/."""
    page = fetch(f"{BASE_URL}/Public/DetaliiDocument/{page_id}")
    # The site writes this href without quotes: href=/Public/FormaPrintabila/...
    match = re.search(r'href="?(/Public/FormaPrintabila/[^"\s>]+)', page)
    if match is None:
        raise RuntimeError(f"No printable-form link found on page {page_id}")

    html = fetch(BASE_URL + match.group(1))
    path = settings.raw_dir / f"{file_stem}.html"
    path.write_text(html, encoding="utf-8")
    print(f"Saved {path.name} ({len(html) // 1024} KB)")


def main() -> None:
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    for doc in DOCUMENTS:
        download_document(doc.file_stem, doc.page_id)
        time.sleep(1)  # be polite to a public government server


if __name__ == "__main__":
    main()
