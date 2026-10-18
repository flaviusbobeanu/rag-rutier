"""Plug our own judge LLM (Claude) and embeddings (bge-m3) into RAGAS.

RAGAS metrics ask the judge for JSON matching a Pydantic model (e.g. a list of statements and
verdicts). Its built-in Anthropic wrapper sends `temperature`/`top_p`, which current Claude models
reject, so this adapter calls Claude with structured outputs instead: the API itself guarantees the
reply matches the Pydantic schema.
"""

import asyncio
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM

from src.config import settings
from src.retrieval.embeddings import embed

T = TypeVar("T", bound=BaseModel)


class ClaudeJudge(InstructorBaseRagasLLM):
    """A RAGAS judge LLM backed by the Claude API."""

    def __init__(self, model: str, effort: str = "low", max_concurrency: int = 4) -> None:
        self.model = model
        self.effort = effort  # judging is a simple task; low effort keeps it fast and cheap
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or None)
        self.semaphore = asyncio.Semaphore(max_concurrency)  # stay under the API rate limit

    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        async with self.semaphore:
            response = await self.client.messages.parse(
                model=self.model,
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}],
                output_format=response_model,
                output_config={"effort": self.effort},
            )
        if response.parsed_output is None:
            raise RuntimeError(f"Judge returned no parsable output (stop_reason={response.stop_reason})")
        return response.parsed_output

    def generate(self, prompt: str, response_model: type[T]) -> T:
        return asyncio.run(self.agenerate(prompt, response_model))


class LocalEmbeddings(BaseRagasEmbedding):
    """RAGAS embeddings using the same sentence-transformers model as our retrieval."""

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return embed([text])[0]

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_text(text)

    def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return embed(texts)

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return self.embed_texts(texts)
