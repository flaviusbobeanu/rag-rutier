"""Re-order retrieved chunks with a cross-encoder (default: BAAI/bge-reranker-v2-m3).

The embedding model compares a question vector with chunk vectors computed separately. A cross-encoder
reads question and chunk together and scores how well the chunk answers the question: more accurate,
but too slow for the whole corpus, so it only re-orders the top candidates from the fast search.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.store import RetrievedChunk


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load the cross-encoder once (it is about as large as the embedding model)."""
    # 512 tokens covers the start of every chunk, where the topic is; 1024 was slower with no gain (eval/results.md).
    return CrossEncoder(settings.reranker_model, max_length=512)


def rerank(question: str, chunks: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
    """Return the k chunks the cross-encoder scores highest. The score becomes the reranker score."""
    if not chunks:
        return []
    scores = get_reranker().predict([(question, chunk.text) for chunk in chunks], batch_size=8)
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)[:k]
    return [RetrievedChunk(c.chunk_id, c.text, c.metadata, float(score)) for c, score in ranked]
