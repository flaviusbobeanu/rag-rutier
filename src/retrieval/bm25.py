"""Keyword search over the chunks with BM25.

BM25 ranks chunks by how many of the question's words they contain, weighting rare words more.
It complements embeddings: exact terms like "art. 102", "ITP" or "15 puncte" match reliably.
The index is small (891 chunks), so it is built in memory from chunks.jsonl on first use.
"""

import json
import re
import unicodedata
from functools import lru_cache

import snowballstemmer
from rank_bm25 import BM25Okapi

from src.config import settings
from src.retrieval.store import RetrievedChunk

STEMMER = snowballstemmer.stemmer("romanian")

# Very common words (written without diacritics) that carry no meaning for search, plus words that
# are typical for questions but rare in law text ("primesc", "pățesc"). Removed before stemming.
STOPWORDS = {
    "a", "al", "ale", "am", "ar", "as", "au", "ca", "care", "cat", "cata", "cate", "cati", "ce", "cine",
    "cu", "cum", "dar", "daca", "de", "din", "este", "eu", "fi", "fie", "in", "intr", "la", "le", "lui",
    "mai", "ma", "mi", "ne", "nu", "o", "or", "ori", "pe", "pentru", "pot", "prin", "sa", "se", "si",
    "sunt", "sau", "un", "una", "unei", "unui", "primesc", "patesc", "trebuie", "voie", "imi", "meu", "mea",
}  # fmt: skip

# Old texts use cedilla letters (ş, ţ); normalize them to the comma forms (ș, ț) the stemmer expects.
CEDILLA = str.maketrans({"ş": "ș", "ţ": "ț", "Ş": "Ș", "Ţ": "Ț"})


def strip_diacritics(word: str) -> str:
    """ "plâng" -> "plang", so queries typed without diacritics still match."""
    return "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop stopwords, stem, then remove diacritics."""
    words = re.findall(r"\w+", text.lower().translate(CEDILLA))
    words = [word for word in words if strip_diacritics(word) not in STOPWORDS]
    return [strip_diacritics(stem) for stem in STEMMER.stemWords(words)]


@lru_cache(maxsize=1)
def get_index() -> tuple[BM25Okapi, list[dict]]:
    """Build the BM25 index once and keep it in memory."""
    path = settings.processed_dir / settings.chunks_file
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return BM25Okapi([tokenize(chunk["text"]) for chunk in chunks]), chunks


def bm25_search(question: str, k: int = 5) -> list[RetrievedChunk]:
    """Return the k chunks with the highest BM25 score for the question."""
    index, chunks = get_index()
    scores = index.get_scores(tokenize(question))
    best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        RetrievedChunk(
            chunk_id=chunks[i]["chunk_id"],
            text=chunks[i]["text"],
            metadata={
                key: value or "" for key, value in chunks[i].items() if key not in ("text", "chunk_id")
            },
            score=float(scores[i]),
        )
        for i in best
    ]
