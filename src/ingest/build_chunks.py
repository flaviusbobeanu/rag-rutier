"""Turn the parsed laws into chunks with metadata and save them to data/processed/chunks.jsonl.

Chunking rule: one chunk per article; if an article is longer than MAX_ARTICLE_CHARS,
one chunk per paragraph (alineat) instead. Each chunk's text starts with a header line
("OUG 195/2002, art. 101 alin. (1) (Capitolul VII - ...)") so it makes sense on its own.

Variant (--split-lists): a paragraph that is still too long and is a list of letters/points is split
into one chunk per item, each repeating the paragraph's intro sentence.

Run:
    python -m src.ingest.build_chunks
    python -m src.ingest.build_chunks --split-lists --out chunks_split.jsonl
"""

import argparse
import json
import re
from collections import Counter

from src.config import settings
from src.ingest.parse_html import Article, parse_law
from src.ingest.sources import DOCUMENTS, LawDocument

MAX_ARTICLE_CHARS = 1500
# With --split-lists, a list paragraph longer than this is split per item. Lower than
# MAX_ARTICLE_CHARS because even a medium list usually mixes unrelated offences.
SPLIT_LIST_CHARS = 800


# A list item line: a letter "e) ..." or a numbered point "19. ...".
LIST_ITEM = re.compile(r"^(?:([a-z](?:\^\d+)?)\)|(\d+(?:\^\d+)?)\.)\s")


def make_chunk(
    doc: LawDocument, article: Article, alineat: str | None, body: str, item: str | None = None
) -> dict:
    """Build one chunk: header + text, plus metadata used for citations and filtering.

    `item` is a letter/point inside the paragraph, e.g. "lit. e)" or "lit. a) pct. 3".
    """
    reference = (
        f"art. {article.number}" + (f" alin. ({alineat})" if alineat else "") + (f" {item}" if item else "")
    )
    context = " | ".join(part for part in (article.capitol, article.sectiune) if part)
    header = f"{doc.citation}, {reference}" + (f" ({context})" if context else "")

    chunk_id = f"{doc.id_prefix}_art{article.number}" + (f"_alin{alineat}" if alineat else "")
    if item:
        chunk_id += "_" + re.sub(r"\W+", "", item.replace(" ", "_").replace(".", ""))
    chunk = {
        "chunk_id": chunk_id.replace("^", "_"),
        "text": f"{header}\n{body}",
        "lege": doc.citation,
        "capitol": article.capitol,
        "sectiune": article.sectiune,
        "articol": article.number,
        "alineat": alineat,
        "versiune": doc.version,
    }
    if item:
        chunk["subdiviziune"] = item
    return chunk


def split_list(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a paragraph into its intro and its list items.

    Returns (intro, [(item label, item text)]). A point under a letter ("a) 2 puncte ...: 3. centura")
    gets label "lit. a) pct. 3" and the letter's line as extra context, since that line holds the
    information the point depends on (here: the number of penalty points).
    """
    intro: list[str] = []
    items: list[tuple[str, str]] = []
    letter: tuple[str, str] | None = None  # (letter, line) of a letter that has points under it
    lines = text.split("\n")
    for i, line in enumerate(lines):
        match = LIST_ITEM.match(line)
        if match is None:
            if items:  # text after a list item continues that item
                label, item_text = items[-1]
                items[-1] = (label, f"{item_text}\n{line}")
            else:
                intro.append(line)
        elif match.group(1):  # a letter
            next_match = LIST_ITEM.match(lines[i + 1]) if i + 1 < len(lines) else None
            if next_match and next_match.group(2):  # points follow: keep the letter as context only
                letter = (match.group(1), line)
            else:
                letter = None
                items.append((f"lit. {match.group(1)})", line))
        elif letter:  # a point under a letter
            items.append((f"lit. {letter[0]}) pct. {match.group(2)}", f"{letter[1]}\n{line}"))
        else:  # a point directly in the paragraph
            items.append((f"pct. {match.group(2)}", line))
    return "\n".join(intro), items


def chunk_paragraph(
    doc: LawDocument, article: Article, alineat: str | None, text: str, split_lists: bool
) -> list[dict]:
    """One chunk for the paragraph, or (split_lists=True and the paragraph is a long list) one per item,
    each repeating the paragraph's intro so the item keeps its meaning (e.g. the sanction)."""
    if split_lists and len(text) > SPLIT_LIST_CHARS:
        intro, items = split_list(text)
        if len(items) >= 2:
            return [
                make_chunk(doc, article, alineat, f"{intro}\n{item_text}" if intro else item_text, label)
                for label, item_text in items
            ]
    return [make_chunk(doc, article, alineat, text)]


def chunk_article(doc: LawDocument, article: Article, split_lists: bool = False) -> list[dict]:
    """Split one article into chunks: whole article if short enough, else one per alineat.
    With split_lists, long paragraphs that are lists are further split per letter/point."""
    # With split_lists, any article long enough to contain a list to split goes paragraph by paragraph,
    # so items of different paragraphs (each with its own "a)", "b)") are never mixed up.
    limit = SPLIT_LIST_CHARS if split_lists else MAX_ARTICLE_CHARS
    if len(article.text) <= limit or not article.alineate:
        return chunk_paragraph(doc, article, None, article.text, split_lists)

    chunks = [make_chunk(doc, article, None, article.intro)] if article.intro else []
    for number, text in article.alineate:
        chunks += chunk_paragraph(doc, article, number, text, split_lists)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-lists", action="store_true", help="split long list paragraphs per item")
    parser.add_argument("--out", default=settings.chunks_file, help="output file name in data/processed/")
    args = parser.parse_args()

    all_chunks = []
    print(f"{'law':<14} {'in HTML':>8} {'kept':>6} {'chunks':>7} {'avg chars':>10} {'max chars':>10}")
    for doc in DOCUMENTS:
        path = settings.raw_dir / f"{doc.file_stem}.html"
        in_html = path.read_text(encoding="utf-8").count('class="S_ART"')
        articles = parse_law(path)
        chunks = [chunk for article in articles for chunk in chunk_article(doc, article, args.split_lists)]
        all_chunks += chunks

        lengths = [len(chunk["text"]) for chunk in chunks]
        print(
            f"{doc.citation:<14} {in_html:>8} {len(articles):>6} {len(chunks):>7}"
            f" {sum(lengths) // len(lengths):>10} {max(lengths):>10}"
        )

    duplicates = [cid for cid, n in Counter(c["chunk_id"] for c in all_chunks).items() if n > 1]
    if duplicates:  # chunk IDs are database keys, so this is an error, not a warning
        raise SystemExit(f"ERROR: {len(duplicates)} duplicate chunk IDs, e.g. {duplicates[:5]}")

    too_long = sum(len(c["text"]) > 2 * MAX_ARTICLE_CHARS for c in all_chunks)
    print(f"\nChunks longer than {2 * MAX_ARTICLE_CHARS} chars (long single paragraphs): {too_long}")

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.processed_dir / args.out
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main()
