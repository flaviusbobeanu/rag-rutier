# CLAUDE.md: RAG assistant for Romanian traffic law

## Project overview

A legal assistant that answers questions about Romanian traffic law **strictly** from a corpus of Romanian laws, and always cites the exact source (law, article, paragraph). It is a RAG (Retrieval-Augmented Generation) system, not a plain chatbot.

Extra features that make the project complex:
- Accuracy evaluation with **RAGAS** plus a retrieval hit-rate metric
- **Conversation memory** (follow-up questions work)
- **PDF upload** (e.g. a traffic fine, "proces-verbal") analyzed on the spot

This is a **portfolio/CV project** for tech internship interviews, so code quality, a clear README and measured evaluation results matter.

## About the developer: IMPORTANT

- 2nd-year AI student. **This is my first real project.** The goal is to learn as much as possible, not just to get working code.
- **Teaching mode:**
  - Before each step, briefly explain *what* we are doing and *why*.
  - When you introduce a new concept (embeddings, chunking, BM25, query rewriting, etc.), explain it in 2–4 simple sentences.
  - Prefer small, readable code over clever code. Avoid unnecessary abstractions.
  - After a step is done, tell me how to run or test it myself, and what I should look at in the output.
  - Do one phase at a time. Do not jump ahead or build several phases at once unless I ask.
  - If I make a design choice that is a bad idea, tell me and explain why.
- When a phase is finished, update its checkboxes in the **Phases** section below.

## Core rules for the assistant's behavior (the product)

1. Answer **only** from retrieved legal text. Never from the LLM's general knowledge.
2. Every claim gets a citation in the form `[OUG 195/2002, art. 102 alin. (3)]`.
3. If the retrieved context does not contain the answer, say so clearly. Do not guess.
4. Each answer shows which version (consolidation date) of the law it relies on.
5. The UI shows a disclaimer: this is not legal advice and does not replace a lawyer.
6. Uploaded fine PDFs contain personal data. Process them in memory and do not store them permanently.

## Tech stack

- **Language:** Python 3.11+
- **RAG framework:** first a minimal version **without a framework** (to understand the mechanics), then **LlamaIndex**
- **Embeddings:** multilingual model with good Romanian support: `BAAI/bge-m3` (default) or `intfloat/multilingual-e5-large`, via `sentence-transformers`
- **Vector DB:** **ChromaDB** (local, persistent). Qdrant only if there is a concrete reason.
- **Keyword search:** BM25 (`rank-bm25`) for hybrid search, since queries like "art. 102" need exact matching
- **LLM:** OpenAI API (small model) by default, configurable to a local model via **Ollama**. Keep the LLM behind one small interface so it can be swapped.
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **PDF:** `pdfplumber` / `pypdf` for text PDFs, `pytesseract` with the Romanian language pack (`ron`) for scanned ones
- **Evaluation:** RAGAS plus a custom hit-rate/recall@k script
- **Config:** `.env` file (never committed) loaded with `python-dotenv` or `pydantic-settings`

## Data sources

Download from the **Portal Legislativ** (legislatie.just.ro), always in the **consolidated form** ("forma consolidată"):

| Document | Role |
|---|---|
| OUG 195/2002 (Codul rutier) | Core traffic code |
| HG 1391/2006 | Implementing regulation of OUG 195/2002 |
| OG 2/2001 | Contravention regime: contesting fines, deadlines, paying half the minimum |

- For the first version, the files are downloaded **manually** and saved in `data/raw/`. Do not write an aggressive scraper.
- Record the consolidation date of each document in `data/raw/SOURCES.md`.

## Project structure

```
rag-rutier/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/            # original downloaded laws + SOURCES.md
│   └── processed/      # chunks as JSON/JSONL with metadata
├── src/
│   ├── config.py
│   ├── ingest/         # parsing + chunking
│   ├── retrieval/      # embeddings, Chroma, BM25, hybrid search
│   ├── generation/     # prompts, LLM client, answer + citations
│   ├── memory/         # conversation history + query rewriting
│   ├── documents/      # PDF upload, text extraction, OCR
│   └── api/            # FastAPI app
├── frontend/           # Streamlit app
├── eval/               # test questions + evaluation scripts + results
├── notebooks/          # experiments
└── tests/              # pytest
```

## Key design decisions

- **Chunking per article** (split into paragraphs if an article is too long). Never fixed-size character chunks that cut an article in half.
- Every chunk carries metadata:
  ```json
  {"text": "...", "lege": "OUG 195/2002", "capitol": "...", "articol": "102",
   "alineat": "3", "versiune": "YYYY-MM-DD", "chunk_id": "oug195_art102_alin3"}
  ```
- Article detection is regex-based (e.g. `Art\.\s*(\d+\^?\d*)`). Handle articles with superscripts like `art. 102^1`.
- **Hybrid retrieval:** vector search + BM25, combined (e.g. Reciprocal Rank Fusion).
- **Query rewriting:** before retrieval, rewrite follow-up questions into standalone questions using the conversation history.
- Every experiment (chunking variant, embedding model, hybrid on/off) is evaluated on the same test set and its results are logged in `eval/results.md`.

## Phases

Work through these **in order**. Get an end-to-end ugly version working early (Phase 3), then improve.

### Phase 0: Setup
- [x] Git repo, virtual env, `requirements.txt`, `.env.example`, folder structure
- [x] `.gitignore` (including `.env`, `data/processed/`, Chroma DB folder, `.venv/`)

### Phase 1: Data
- [x] Laws downloaded into `data/raw/` (printable-form HTML, via `python -m src.ingest.download`)
- [x] `data/raw/SOURCES.md` with links and consolidation dates

### Phase 2: Parsing and chunking
- [x] Clean the raw text (headers, footers, notes about amendments)
- [x] Split by article/paragraph with metadata into `data/processed/chunks.jsonl`
- [x] Sanity check: count of articles per law, inspect a few random chunks
- Known limitations: annexes (road sign images/tables) are skipped; 11 chunks are > 3000 chars
  (e.g. art. 6 definitions, long lists of contraventions). Splitting those per point is a Phase 5 experiment.

### Phase 3: Minimal RAG without a framework
- [ ] Embed chunks, store them in ChromaDB
- [ ] `ask(question)` → retrieve top-k → prompt → LLM → answer with citations
- [ ] Refusal when the context is insufficient
- [ ] Simple CLI to ask questions

### Phase 4: Evaluation
- [ ] `eval/questions.jsonl`: 30–50 hand-written questions with expected answer and expected article(s)
- [ ] Retrieval hit rate / recall@k script
- [ ] RAGAS: faithfulness, answer relevancy, context precision, context recall
- [ ] Baseline results in `eval/results.md`

### Phase 5: Better retrieval
- [ ] BM25 + hybrid search
- [ ] Compare embedding models / chunking variants, log the results
- [ ] (Optional) reranker

### Phase 6: Move to LlamaIndex
- [ ] Re-implement the pipeline with LlamaIndex, same eval results or better
- [ ] Explain to me what LlamaIndex does under the hood versus my manual version

### Phase 7: Backend (FastAPI)
- [ ] `POST /ask` (question + history → answer + sources)
- [ ] `POST /upload` (PDF → extracted info + explanation)
- [ ] `GET /health`

### Phase 8: Conversation memory
- [ ] History handling + query rewriting
- [ ] Add follow-up questions to the eval set

### Phase 9: PDF upload (fines)
- [ ] Text extraction + OCR fallback for scans
- [ ] Extract the violated article, amount and date, then retrieve and explain (including contest/payment options from OG 2/2001)
- [ ] No permanent storage of uploaded files

### Phase 10: Frontend and polish
- [ ] Streamlit chat UI with visible sources and disclaimer
- [ ] Dockerfile
- [ ] README: architecture diagram, how to run, evaluation results table, limitations
- [ ] (Optional) Demo on Hugging Face Spaces

## Coding conventions

- Type hints everywhere, short docstrings on public functions
- Formatting with `ruff format`, linting with `ruff`
- Tests with `pytest` for parsing/chunking and retrieval at minimum
- No secrets in code. API keys only in `.env`.
- Small, focused commits with clear English messages (e.g. `feat: chunk laws by article`)

## Commands

Fill these in as the project grows:

```bash
# setup
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell (Windows)
pip install -r requirements.txt

# ingest
python -m src.ingest.download        # fetch law HTML into data/raw/ (already done; rerun only to update versions)
python -m src.ingest.build_chunks    # HTML -> data/processed/chunks.jsonl, prints stats per law
python -m src.ingest.inspect_chunks  # 3 random chunks; or pass a number, or a chunk_id
python -m src.retrieval.build_index

# ask from CLI
python -m src.cli "Ce amendă primesc dacă trec pe roșu?"

# evaluation
python -m eval.run_eval

# backend / frontend
uvicorn src.api.main:app --reload
streamlit run frontend/app.py

# tests
pytest
```
