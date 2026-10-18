# LlamaIndex vs. the manual pipeline

The project has the same RAG pipeline twice: written by hand (`src/retrieval`, `src/generation`, `src/memory`)
and with LlamaIndex (`src/llamaindex_pipeline`). Both use the same chunks, embedding model, prompt and LLM, so
any difference in the eval comes from the framework, not from the ingredients.

## What each LlamaIndex piece does, and its manual equivalent

| Step | Manual version | LlamaIndex | What LlamaIndex does under the hood |
|---|---|---|---|
| Unit of text | a dict from `chunks.jsonl` | `TextNode` | A node is text + metadata + an ID, plus settings for *which metadata the embedding model sees* and *which the LLM sees*. We hide all metadata from the embedding and show the LLM only the citation tag, so both pipelines embed and prompt exactly the same text. |
| Embedding + storage | `embed()` then `collection.add()` | `VectorStoreIndex(nodes, storage_context=...)` | Calls the embedding model in batches, serializes each node into Chroma's metadata (an extra `_node_content` field holding the whole node as JSON), and stores the vectors. |
| Similarity search | `store.search()` | `index.as_retriever(similarity_top_k=k)` | Embeds the query with the same model, calls Chroma's `query()`, and rebuilds `NodeWithScore` objects from the stored JSON. |
| Article references | Chroma `where=` filter in `references.py` | `MetadataFilters` on the retriever | Translates the filters into the same Chroma `where` clause. We still parse the reference with our own regex; LlamaIndex has no notion of "art. 100 alin. (3)". |
| Combining both | `retrieve()` in `search.py` | a small `BaseRetriever` subclass | Any custom retrieval logic goes into `_retrieve()`; the rest of LlamaIndex accepts it like a built-in retriever. |
| Prompt + LLM call | `build_user_prompt()` + `llm.generate()` | `RetrieverQueryEngine` + `PromptTemplate` | The *response synthesizer* fills `{context_str}` with the node texts and `{query_str}` with the question. In `compact` mode it packs as many nodes as fit into one prompt; if they did not fit, it would make several calls and *refine* the answer step by step. With 5 chunks and a 1M-token model, it is always one call. |
| Follow-up questions | `rewrite_question()` | condense prompt + `llm.predict()` | This is what `CondenseQuestionChatEngine` does internally: format the history into a prompt, ask the LLM for a standalone question, then run the query engine on it. We call the two steps directly so the rewritten question can be shown and evaluated. |
| LLM client | `AnthropicLLM` (our 40 lines) | `llama_index.llms.anthropic.Anthropic` | Wraps the same `anthropic` SDK. It knows which Claude models reject `temperature` and leaves it out for them. Extra API parameters (our `effort` setting) go through `additional_kwargs`. |

## Verified equivalence

- Retrieval: on all 41 single-turn test questions, the LlamaIndex retriever returns **the same top 5 chunks** as
  the manual one (41/41). The first version differed on 2 questions because it did not narrow an article
  reference down to the named paragraph; after adding that filter, the results matched exactly.
- End-to-end scores: see `eval/results.md` (Phase 6).

## What we kept from the manual version, and why

- **Chunking.** LlamaIndex's default `SentenceSplitter` cuts text into fixed-size pieces, which would split an
  article in half and break the citations. Legal text needs structure-aware chunking, so nodes are built from our
  own chunks.
- **Reference parsing, prompt, refusal rule, citation parsing, version footer.** These are domain logic, not
  plumbing. A framework can't know what "art. 100 alin. (3) lit. h)" means.

## Trade-offs

- **For LlamaIndex:** less glue code for standard pieces; swapping the vector database, LLM or embedding model is
  one line; built-in pieces we could add later (rerankers as `node_postprocessors`, streaming, async, tracing).
- **For the manual version:** every step is visible and debuggable; no hidden prompts (the default LlamaIndex
  prompts are in English and would have had to be overridden anyway); fewer dependencies; and building it first
  is what made it possible to check that the framework does the same thing.
