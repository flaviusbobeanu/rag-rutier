"""Turn text into embedding vectors with a sentence-transformers model (default: BAAI/bge-m3)."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import settings

# bge-m3 accepts up to 8192 tokens, but attention cost grows with length. Our longest
# chunks are ~1500 tokens, so 2048 keeps everything without wasting time on padding.
MAX_SEQ_LENGTH = 2048


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it (loading takes several seconds)."""
    model = SentenceTransformer(settings.embedding_model)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def embed(texts: list[str], kind: str = "query", show_progress: bool = False) -> list[list[float]]:
    """Embed questions (kind="query") or chunks (kind="passage"). Vectors are normalized,
    so dot product = cosine similarity."""
    if "e5" in settings.embedding_model:
        # The e5 models were trained with these prefixes and work worse without them.
        texts = [f"{kind}: {text}" for text in texts]
    vectors = get_model().encode(
        texts, batch_size=8, normalize_embeddings=True, show_progress_bar=show_progress
    )
    return vectors.tolist()
