"""Turn the parsed laws into chunks with metadata and save them to data/processed/chunks.jsonl.

Chunking rule: one chunk per article; if an article is longer than MAX_ARTICLE_CHARS,
one chunk per paragraph (alineat) instead. Each chunk's text starts with a header line
("OUG 195/2002, art. 101 alin. (1) (Capitolul VII - ...)") so it makes sense on its own.

Run: python -m src.ingest.build_chunks
"""

import json
from collections import Counter

from src.config import settings
from src.ingest.parse_html import Article, parse_law
from src.ingest.sources import DOCUMENTS, LawDocument

MAX_ARTICLE_CHARS = 1500


def make_chunk(doc: LawDocument, article: Article, alineat: str | None, body: str) -> dict:
    """Build one chunk: header + text, plus metadata used for citations and filtering."""
    reference = f"art. {article.number}" + (f" alin. ({alineat})" if alineat else "")
    context = " | ".join(part for part in (article.capitol, article.sectiune) if part)
    header = f"{doc.citation}, {reference}" + (f" ({context})" if context else "")

    chunk_id = f"{doc.id_prefix}_art{article.number}" + (f"_alin{alineat}" if alineat else "")
    return {
        "chunk_id": chunk_id.replace("^", "_"),
        "text": f"{header}\n{body}",
        "lege": doc.citation,
        "capitol": article.capitol,
        "sectiune": article.sectiune,
        "articol": article.number,
        "alineat": alineat,
        "versiune": doc.version,
    }


def chunk_article(doc: LawDocument, article: Article) -> list[dict]:
    """Split one article into chunks: whole article if short enough, else one per alineat."""
    if len(article.text) <= MAX_ARTICLE_CHARS or not article.alineate:
        return [make_chunk(doc, article, None, article.text)]

    chunks = [make_chunk(doc, article, None, article.intro)] if article.intro else []
    for number, text in article.alineate:
        chunks.append(make_chunk(doc, article, number, text))
    return chunks


def main() -> None:
    all_chunks = []
    print(f"{'law':<14} {'in HTML':>8} {'kept':>6} {'chunks':>7} {'avg chars':>10} {'max chars':>10}")
    for doc in DOCUMENTS:
        path = settings.raw_dir / f"{doc.file_stem}.html"
        in_html = path.read_text(encoding="utf-8").count('class="S_ART"')
        articles = parse_law(path)
        chunks = [chunk for article in articles for chunk in chunk_article(doc, article)]
        all_chunks += chunks

        lengths = [len(chunk["text"]) for chunk in chunks]
        print(
            f"{doc.citation:<14} {in_html:>8} {len(articles):>6} {len(chunks):>7}"
            f" {sum(lengths) // len(lengths):>10} {max(lengths):>10}"
        )

    duplicates = [cid for cid, n in Counter(c["chunk_id"] for c in all_chunks).items() if n > 1]
    if duplicates:
        print(f"WARNING: {len(duplicates)} duplicate chunk IDs, e.g. {duplicates[:5]}")

    too_long = sum(len(c["text"]) > 2 * MAX_ARTICLE_CHARS for c in all_chunks)
    print(f"\nChunks longer than {2 * MAX_ARTICLE_CHARS} chars (long single paragraphs): {too_long}")

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.processed_dir / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main()
