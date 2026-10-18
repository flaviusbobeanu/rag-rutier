"""Find explicit article references in a question ("art. 100 alin. (3) din OUG 195/2002") and fetch
those chunks directly by metadata.

Embeddings and BM25 both handle article numbers poorly (to an embedding, "art. 100" and "art. 102"
look almost the same), but a reference is unambiguous, so an exact lookup is the right tool.
"""

import re

from src.config import settings
from src.retrieval.store import RetrievedChunk, get_client

# How each law can be written in a question. Checked in order; the first match wins.
LAW_PATTERNS = [
    ("OG 2/2001", r"\bO\.?G\.?\s*(?:nr\.?\s*)?2/2001|ordonan[țţt]a\s+(?:guvernului\s+)?(?:nr\.?\s*)?2/2001"),
    ("HG 1391/2006", r"\bH\.?G\.?\s*(?:nr\.?\s*)?1391/2006|regulament"),
    ("OUG 195/2002", r"\bO\.?U\.?G\.?\s*(?:nr\.?\s*)?195/2002|codul\s+rutier"),
]
# Without a law name, a reference almost always means the traffic code (that is what fines cite).
DEFAULT_LAW = "OUG 195/2002"

# "art. 100", "articolul 28^1", "art 102 alin. (3)", "art. 100 alineatul 3", "art. 100 alin. (3) lit. h)"
ARTICLE_RE = re.compile(
    r"\bart(?:icolul|\.)?\s*(\d+(?:\^\d+)?)"
    r"(?:\s*,?\s*alin(?:eatul|\.)?\s*\(?(\d+(?:\^\d+)?)\)?)?"
    r"(?:\s*,?\s*lit(?:era|\.)?\s*([a-z](?:\^\d+)?)\)?)?",
    re.IGNORECASE,
)


def find_references(question: str) -> list[tuple[str, str, str, str]]:
    """Return (lege, articol, alineat, litera) for every article mentioned; "" for parts not given."""
    law = next(
        (name for name, pattern in LAW_PATTERNS if re.search(pattern, question, re.IGNORECASE)), DEFAULT_LAW
    )
    return [(law, art, alin, lit.lower()) for art, alin, lit in ARTICLE_RE.findall(question)]


def narrow(chunks: list[RetrievedChunk], key: str, wanted: str) -> list[RetrievedChunk]:
    """Keep the chunks whose metadata[key] matches `wanted`; keep all if none match or nothing is wanted."""
    if not wanted:
        return chunks
    if key == "subdiviziune":  # "lit. h)" or "lit. h) pct. 2"
        matching = [c for c in chunks if c.metadata.get(key, "").startswith(f"lit. {wanted})")]
    else:
        matching = [c for c in chunks if c.metadata.get(key) == wanted]
    return matching or chunks


def fetch_article(lege: str, articol: str, alineat: str = "", litera: str = "") -> list[RetrievedChunk]:
    """The chunks of one article, narrowed to the paragraph and letter if given and stored separately."""
    collection = get_client().get_collection(settings.collection_name)
    result = collection.get(where={"$and": [{"lege": lege}, {"articol": articol}]})
    chunks = [
        RetrievedChunk(chunk_id=cid, text=doc, metadata=meta, score=1.0)
        for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]
    return narrow(narrow(chunks, "alineat", alineat), "subdiviziune", litera)


def lookup_references(question: str, k: int = 5) -> list[RetrievedChunk]:
    """Fetch the chunks of the articles referenced in the question (at most k)."""
    found = [chunk for reference in find_references(question) for chunk in fetch_article(*reference)]
    return found[:k]
