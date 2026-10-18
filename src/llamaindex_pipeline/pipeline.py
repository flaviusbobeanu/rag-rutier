"""Answer questions with LlamaIndex: retriever -> query engine -> Claude.

Mapping to the manual pipeline (src/generation/answer.py):
    store.search()            -> VectorIndexRetriever
    references.lookup_*()     -> the same retriever with MetadataFilters (lege, articol)
    build_user_prompt + LLM   -> RetrieverQueryEngine with our prompt as a PromptTemplate
    memory.rewrite_question() -> a condense prompt + llm.predict (what CondenseQuestionChatEngine does)
The result is converted to our Answer class, so the same eval scripts work for both pipelines.
"""

from functools import lru_cache

from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import settings
from src.generation.answer import DEFAULT_K, Answer
from src.generation.prompts import SYSTEM_PROMPT
from src.memory.rewrite import REWRITE_PROMPT, format_history
from src.retrieval.embeddings import MAX_SEQ_LENGTH
from src.retrieval.references import find_references
from src.retrieval.store import RetrievedChunk, get_client

COLLECTION = "laws_llamaindex"

QA_TEMPLATE = PromptTemplate("<context>\n{context_str}\n</context>\n\nÎntrebare: {query_str}")
CONDENSE_TEMPLATE = PromptTemplate(
    REWRITE_PROMPT + "\n\n<conversation>\n{chat_history}\n</conversation>\n\nLatest question: {question}"
)


@lru_cache(maxsize=1)
def embed_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(
        model_name=settings.embedding_model, normalize=True, max_length=MAX_SEQ_LENGTH
    )


@lru_cache(maxsize=1)
def llm() -> Anthropic:
    # LlamaIndex leaves out `temperature` for current Claude models, which reject it.
    return Anthropic(
        model=settings.anthropic_model,
        max_tokens=16000,
        api_key=settings.anthropic_api_key or None,
        system_prompt=SYSTEM_PROMPT,
        additional_kwargs={"output_config": {"effort": settings.anthropic_effort}},
    )


@lru_cache(maxsize=1)
def index() -> VectorStoreIndex:
    """Open the existing collection built by src.llamaindex_pipeline.build_index (no re-embedding)."""
    store = ChromaVectorStore(chroma_collection=get_client().get_collection(COLLECTION))
    return VectorStoreIndex.from_vector_store(store, embed_model=embed_model())


class ReferenceAwareRetriever(BaseRetriever):
    """Vector retrieval, with articles named in the question ("art. 100 din OUG 195/2002") fetched first
    through metadata filters."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def filtered(self, query_bundle: QueryBundle, filters: list[MetadataFilter]) -> list[NodeWithScore]:
        """Vector search restricted to nodes whose metadata matches all filters."""
        retriever = index().as_retriever(similarity_top_k=self.k, filters=MetadataFilters(filters=filters))
        return retriever.retrieve(query_bundle)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        nodes: list[NodeWithScore] = []
        for lege, articol, alineat, _ in find_references(query_bundle.query_str):
            article = [MetadataFilter(key="lege", value=lege), MetadataFilter(key="articol", value=articol)]
            paragraph = article + [MetadataFilter(key="alineat", value=alineat)] if alineat else []
            # Try the named paragraph first; fall back to the whole article if it is not stored separately.
            for filters in filter(None, (paragraph, article)):
                found = self.filtered(query_bundle, filters)
                if found:
                    nodes += found
                    break
        seen = {node.node_id for node in nodes}
        similar = index().as_retriever(similarity_top_k=self.k).retrieve(query_bundle)
        return (nodes + [node for node in similar if node.node_id not in seen])[: self.k]


def condense(question: str, history: list[dict]) -> str:
    """Rewrite a follow-up into a standalone question (the core of LlamaIndex's CondenseQuestionChatEngine)."""
    if not history:
        return question
    rewritten = llm().predict(CONDENSE_TEMPLATE, chat_history=format_history(history), question=question)
    return rewritten.strip().strip('"') or question


def ask(question: str, history: list[dict] | None = None, k: int = DEFAULT_K) -> Answer:
    """LlamaIndex version of src.generation.answer.ask(); returns the same Answer type."""
    standalone = condense(question, history or [])
    engine = RetrieverQueryEngine.from_args(
        ReferenceAwareRetriever(k), llm=llm(), text_qa_template=QA_TEMPLATE, response_mode="compact"
    )
    response = engine.query(standalone)
    sources = [
        RetrievedChunk(node.node_id, node.node.get_content(), dict(node.node.metadata), node.score or 0.0)
        for node in response.source_nodes
    ]
    return Answer(
        question=question, text=str(response).strip(), sources=sources, standalone_question=standalone
    )
