"""One entry point for retrieval: vector search, BM25, or both combined (hybrid), with articles
named explicitly in the question ("art. 100 din OUG 195/2002") looked up directly and put first.

Hybrid uses Reciprocal Rank Fusion (RRF): each chunk gets 1 / (RRF_K + rank) from every list it
appears in, and the sums decide the final order. RRF only looks at ranks, so it does not matter that
cosine similarities (0-1) and BM25 scores (0-30+) are on completely different scales.
"""

from src.config import settings
from src.retrieval.bm25 import bm25_search
from src.retrieval.references import lookup_references
from src.retrieval.rerank import rerank
from src.retrieval.store import RetrievedChunk, search

RRF_K = 60  # standard value from the original RRF paper; dampens the advantage of rank 1 over rank 2
CANDIDATES = 20  # how many results to take from each retriever before fusing
RERANK_CANDIDATES = 10  # how many results the cross-encoder re-orders (20 was slower, not better)


def reciprocal_rank_fusion(rankings: list[list[RetrievedChunk]], k: int) -> list[RetrievedChunk]:
    """Merge several ranked lists into one. The returned score is the RRF score."""
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1 / (RRF_K + rank)
            chunks.setdefault(chunk.chunk_id, chunk)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [
        RetrievedChunk(chunks[cid].chunk_id, chunks[cid].text, chunks[cid].metadata, scores[cid])
        for cid in best
    ]


def ranked_search(question: str, k: int, mode: str) -> list[RetrievedChunk]:
    """Similarity search with the chosen mode: "vector", "bm25" or "hybrid"."""
    if mode == "vector":
        return search(question, k)
    if mode == "bm25":
        return bm25_search(question, k)
    if mode == "hybrid":
        n = max(k, CANDIDATES)
        return reciprocal_rank_fusion([search(question, n), bm25_search(question, n)], k)
    raise ValueError(f"Unknown retrieval mode: {mode!r} (use 'vector', 'bm25' or 'hybrid')")


def retrieve(
    question: str,
    k: int = 5,
    mode: str | None = None,
    use_references: bool = True,
    use_reranker: bool | None = None,
) -> list[RetrievedChunk]:
    """Top-k chunks: articles referenced explicitly in the question first, then similarity search
    (optionally re-ordered by a cross-encoder)."""
    mode = mode or settings.retrieval_mode
    use_reranker = settings.use_reranker if use_reranker is None else use_reranker
    if use_reranker:
        ranked = rerank(question, ranked_search(question, max(k, RERANK_CANDIDATES), mode), k)
    else:
        ranked = ranked_search(question, k, mode)
    if not use_references:
        return ranked
    referenced = lookup_references(question, k)
    seen = {chunk.chunk_id for chunk in referenced}
    return (referenced + [chunk for chunk in ranked if chunk.chunk_id not in seen])[:k]
