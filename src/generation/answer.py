"""The RAG pipeline: question -> retrieve chunks -> prompt -> LLM -> answer with citations."""

import re
from dataclasses import dataclass

from src.generation.llm import LLM, get_llm
from src.generation.prompts import REFUSAL, SYSTEM_PROMPT, build_user_prompt
from src.memory.rewrite import rewrite_question
from src.retrieval.search import retrieve
from src.retrieval.store import RetrievedChunk

DEFAULT_K = 5

# Matches citations like "[OUG 195/2002, art. 102 alin. (3)]" or "[OG 2/2001, art. 28^1]".
# The model sometimes adds the letter or point, e.g. "... alin. (3) lit. d)]" or "... pct. 19]"; those are
# allowed but not captured, since we track sources at the paragraph level.
# Paragraph ranges like "alin. (1)-(2)" or "alin. (1) și (2)" are allowed too; the first paragraph is kept.
CITATION_RE = re.compile(
    r"\[([A-Z]+ \d+/\d{4}), art\. (\d+(?:\^\d+)?)"
    r"(?: alin\. \((\d+(?:\^\d+)?)\)(?:\s*(?:-|–|și|,)\s*\(\d+(?:\^\d+)?\))*)?"
    r"(?: lit\. [a-z](?:\^\d+)?\))?(?: pct\. \d+(?:\^\d+)?)?\]"
)


@dataclass
class Answer:
    question: str
    text: str
    sources: list[RetrievedChunk]  # everything that was put in the prompt
    standalone_question: str = ""  # the question after rewriting with the history (= question if no history)

    @property
    def refused(self) -> bool:
        return REFUSAL in self.text

    @property
    def citations(self) -> list[tuple[str, str, str]]:
        """(lege, articol, alineat) for every citation in the answer, in order, without duplicates."""
        return list(dict.fromkeys(CITATION_RE.findall(self.text)))

    @property
    def versions(self) -> dict[str, str]:
        """Consolidation date of every law the answer cites, e.g. {"OUG 195/2002": "2026-08-25"}."""
        cited_laws = {lege for lege, _, _ in self.citations}
        return {
            chunk.metadata["lege"]: chunk.metadata["versiune"]
            for chunk in self.sources
            if chunk.metadata["lege"] in cited_laws
        }


def ask(
    question: str, history: list[dict] | None = None, k: int = DEFAULT_K, llm: LLM | None = None
) -> Answer:
    """Answer a question about traffic law from the top-k retrieved chunks.

    `history` is the earlier conversation as [{"role": "user"|"assistant", "content": "..."}]. If given,
    the question is first rewritten into a standalone question, which is then used for retrieval and answer.
    """
    llm = llm or get_llm()
    standalone = rewrite_question(question, history or [], llm)
    chunks = retrieve(standalone, k=k)
    text = llm.generate(SYSTEM_PROMPT, build_user_prompt(standalone, chunks))
    return Answer(question=question, text=text, sources=chunks, standalone_question=standalone)
