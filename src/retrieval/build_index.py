"""Embed all chunks from data/processed/chunks.jsonl and store them in a ChromaDB collection.

The collection is deleted and rebuilt from scratch each time, so the index always matches
the current chunks file.

Run: python -m src.retrieval.build_index
"""

import json
import time

from src.config import settings
from src.retrieval.embeddings import embed
from src.retrieval.store import get_client


def load_chunks() -> list[dict]:
    """Read all chunks from the JSONL file produced by src.ingest.build_chunks."""
    path = settings.processed_dir / settings.chunks_file
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def to_chroma_metadata(chunk: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool, so None becomes ""."""
    return {key: value or "" for key, value in chunk.items() if key not in ("text", "chunk_id")}


def main() -> None:
    chunks = load_chunks()
    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(set(ids)) != len(ids):  # check before the slow embedding step, not after it
        raise SystemExit(f"Duplicate chunk IDs in {settings.chunks_file}; fix the chunking first.")
    print(f"Embedding {len(chunks)} chunks with {settings.embedding_model} ...")
    start = time.perf_counter()
    vectors = embed([chunk["text"] for chunk in chunks], kind="passage", show_progress=True)
    print(f"Done in {time.perf_counter() - start:.0f}s, vector size {len(vectors[0])}")

    client = get_client()
    if settings.collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(settings.collection_name)
    # "cosine" tells Chroma to rank by cosine distance (1 - cosine similarity).
    collection = client.create_collection(settings.collection_name, metadata={"hnsw:space": "cosine"})

    batch = 500  # Chroma limits how many items one add() call may contain
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        collection.add(
            ids=[chunk["chunk_id"] for chunk in part],
            embeddings=vectors[i : i + batch],
            documents=[chunk["text"] for chunk in part],
            metadatas=[to_chroma_metadata(chunk) for chunk in part],
        )
    print(
        f"Stored {collection.count()} chunks in {settings.chroma_dir} (collection '{settings.collection_name}')"
    )


if __name__ == "__main__":
    main()
