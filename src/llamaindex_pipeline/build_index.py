"""Build the LlamaIndex version of the index from our chunks (same chunks, own Chroma collection).

LlamaIndex's default text splitters cut by size, which would split articles in half, so we keep our
legal chunking and turn each chunk into a TextNode. LlamaIndex then embeds and stores the nodes.

Run: python -m src.llamaindex_pipeline.build_index
"""

import time

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.generation.prompts import citation_from_metadata
from src.llamaindex_pipeline.pipeline import COLLECTION, embed_model
from src.retrieval.build_index import load_chunks
from src.retrieval.store import get_client


def chunk_to_node(chunk: dict) -> TextNode:
    """One TextNode per chunk. Metadata is not embedded; the LLM sees only the citation tag above the text,
    so the embedding input and the prompt look exactly like in the manual pipeline."""
    metadata = {key: value or "" for key, value in chunk.items() if key not in ("text", "chunk_id")}
    metadata["citare"] = citation_from_metadata(metadata)
    return TextNode(
        id_=chunk["chunk_id"],
        text=chunk["text"],
        metadata=metadata,
        excluded_embed_metadata_keys=list(metadata),
        excluded_llm_metadata_keys=[key for key in metadata if key != "citare"],
        metadata_template="{value}",
        text_template="{metadata_str}\n{content}",
    )


def main() -> None:
    nodes = [chunk_to_node(chunk) for chunk in load_chunks()]
    client = get_client()
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    start = time.perf_counter()
    storage = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
    VectorStoreIndex(nodes, storage_context=storage, embed_model=embed_model(), show_progress=True)
    print(f"Indexed {collection.count()} nodes into '{COLLECTION}' in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
