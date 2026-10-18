"""HTTP API for the traffic-law assistant.

Run:  uvicorn src.api.main:app --reload
Docs: http://127.0.0.1:8000/docs (try the endpoints from the browser)
"""

from functools import lru_cache
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import settings
from src.documents.extract import OCRUnavailableError, extract_text
from src.documents.fine import analyze_fine, describe
from src.generation.answer import DEFAULT_K, Answer, ask
from src.generation.llm import LLM, get_llm
from src.generation.prompts import citation
from src.ingest.sources import DOCUMENTS
from src.retrieval.store import get_client

DISCLAIMER = "Acesta nu este un sfat juridic și nu înlocuiește un avocat."

app = FastAPI(title="RAG Rutier", description="Answers about Romanian traffic law, with citations.")


@lru_cache(maxsize=1)
def llm() -> LLM:
    """One LLM client for the whole app (it keeps a connection pool)."""
    return get_llm()


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000, examples=["Ce sancțiune primesc dacă trec pe roșu?"])
    history: list[Message] = Field(default_factory=list, description="Earlier messages, oldest first")
    k: int = Field(DEFAULT_K, ge=1, le=20, description="How many law chunks to retrieve")


class Source(BaseModel):
    citation: str
    lege: str
    articol: str
    alineat: str
    versiune: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    standalone_question: str
    refused: bool
    sources: list[Source]
    versions: dict[str, str]
    disclaimer: str = DISCLAIMER


def to_response(answer: Answer) -> AskResponse:
    sources = [
        Source(
            citation=citation(chunk),
            lege=chunk.metadata["lege"],
            articol=chunk.metadata["articol"],
            alineat=chunk.metadata.get("alineat", ""),
            versiune=chunk.metadata["versiune"],
            score=round(chunk.score, 4),
            text=chunk.text,
        )
        for chunk in answer.sources
    ]
    return AskResponse(
        answer=answer.text,
        standalone_question=answer.standalone_question,
        refused=answer.refused,
        sources=sources,
        versions=answer.versions,
    )


@app.get("/health")
def health() -> dict:
    """Is the index loaded, and which model and law versions are in use?"""
    chunks = get_client().get_collection(settings.collection_name).count()
    return {
        "status": "ok",
        "llm": llm().name,
        "collection": settings.collection_name,
        "chunks": chunks,
        "law_versions": {doc.citation: doc.version for doc in DOCUMENTS},
    }


MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class FineFactsResponse(BaseModel):
    articles: list[str]
    amount_lei: int | None
    penalty_points: int | None
    date: str | None
    text_method: str


class UploadResponse(BaseModel):
    facts: FineFactsResponse
    explanation: AskResponse | None
    message: str = ""
    disclaimer: str = DISCLAIMER


@app.post("/upload")
def upload(file: UploadFile) -> UploadResponse:
    """Upload a fine (proces-verbal) as PDF: get the key facts and an explanation with citations.

    The file is processed in memory and never saved. Only the article, amount, points and date are used
    to build the question for the LLM; personal data from the document is never sent to it.
    """
    pdf_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The PDF is larger than 10 MB")
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=415, detail="The file is not a PDF")

    try:
        text, method = extract_text(pdf_bytes)
        analysis = analyze_fine(text, method, llm=llm())
    except OCRUnavailableError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except anthropic.APIStatusError as error:
        raise HTTPException(status_code=502, detail=f"LLM API error: {error.message}") from error
    finally:
        del pdf_bytes  # drop our reference right away; nothing was written to disk

    facts = FineFactsResponse(
        articles=[describe(reference) for reference in analysis.facts.articles],
        amount_lei=analysis.facts.amount_lei,
        penalty_points=analysis.facts.penalty_points,
        date=analysis.facts.date,
        text_method=analysis.text_method,
    )
    if analysis.answer is None:
        message = "Nu am găsit în document niciun articol din OUG 195/2002 sau HG 1391/2006."
        return UploadResponse(facts=facts, explanation=None, message=message)
    return UploadResponse(facts=facts, explanation=to_response(analysis.answer))


@app.post("/ask")
def ask_endpoint(request: AskRequest) -> AskResponse:
    """Answer a question (optionally a follow-up, with the earlier conversation in `history`)."""
    history = [message.model_dump() for message in request.history]
    try:
        answer = ask(request.question, history=history, k=request.k, llm=llm())
    except anthropic.APIStatusError as error:
        raise HTTPException(status_code=502, detail=f"LLM API error: {error.message}") from error
    except anthropic.APIConnectionError as error:
        raise HTTPException(status_code=503, detail="Could not reach the LLM API") from error
    return to_response(answer)
