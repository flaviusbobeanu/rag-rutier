"""Access to the persistent ChromaDB collection and vector search over it."""

from dataclasses import dataclass

import chromadb

from src.config import settings
from src.retrieval.embeddings import embed


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict  # lege, articol, alineat, capitol, sectiune, versiune
    score: float  # cosine similarity, higher = more similar


def get_client() -> chromadb.ClientAPI:
    """Open (or create) the Chroma database folder on disk."""
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def search(question: str, k: int = 5) -> list[RetrievedChunk]:
    """Return the k chunks whose embeddings are closest to the question's embedding."""
    collection = get_client().get_collection(settings.collection_name)
    result = collection.query(query_embeddings=embed([question]), n_results=k)
    return [
        RetrievedChunk(chunk_id=cid, text=doc, metadata=meta, score=1 - dist)
        for cid, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
