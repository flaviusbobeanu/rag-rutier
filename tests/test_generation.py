"""Tests for prompt building, citation parsing and retrieval metrics (no model or database needed)."""

from eval.retrieval_eval import score_question
from src.generation.answer import Answer
from src.generation.prompts import REFUSAL, build_user_prompt, citation
from src.retrieval.references import find_references
from src.retrieval.store import RetrievedChunk


def make_chunk(lege: str, articol: str, alineat: str = "", versiune: str = "2026-08-25") -> RetrievedChunk:
    metadata = {"lege": lege, "articol": articol, "alineat": alineat, "versiune": versiune}
    return RetrievedChunk(chunk_id=f"{lege}_{articol}", text="text", metadata=metadata, score=0.5)


def test_citation_with_and_without_alineat() -> None:
    assert citation(make_chunk("OUG 195/2002", "102", "3")) == "[OUG 195/2002, art. 102 alin. (3)]"
    assert citation(make_chunk("OG 2/2001", "28^1")) == "[OG 2/2001, art. 28^1]"


def test_user_prompt_contains_citation_tags_and_question() -> None:
    prompt = build_user_prompt("Ce viteză?", [make_chunk("OUG 195/2002", "49")])
    assert "[OUG 195/2002, art. 49]" in prompt
    assert prompt.endswith("Întrebare: Ce viteză?")


def test_answer_parses_citations_and_versions_of_cited_laws_only() -> None:
    sources = [make_chunk("OUG 195/2002", "100", "3"), make_chunk("OG 2/2001", "31", versiune="2026-03-27")]
    text = (
        "Amenda din clasa a II-a [OUG 195/2002, art. 100 alin. (3)] și suspendare 30 de zile "
        "[OUG 195/2002, art. 100 alin. (3)]. Vezi și [OUG 195/2002, art. 109^1]."
    )
    answer = Answer(question="q", text=text, sources=sources)
    assert answer.citations == [("OUG 195/2002", "100", "3"), ("OUG 195/2002", "109^1", "")]
    assert answer.versions == {"OUG 195/2002": "2026-08-25"}  # OG 2/2001 was retrieved but not cited
    assert not answer.refused


def test_citations_with_letter_or_point() -> None:
    text = "[OUG 195/2002, art. 100 alin. (3) lit. d)] și [OUG 195/2002, art. 101 alin. (1) pct. 19]"
    answer = Answer(question="q", text=text, sources=[])
    assert answer.citations == [("OUG 195/2002", "100", "3"), ("OUG 195/2002", "101", "1")]


def test_citations_with_paragraph_range() -> None:
    text = "[HG 1391/2006, art. 52 alin. (1)-(2)] și [OG 2/2001, art. 31 alin. (1) și (2)]"
    answer = Answer(question="q", text=text, sources=[])
    assert answer.citations == [("HG 1391/2006", "52", "1"), ("OG 2/2001", "31", "1")]


def test_refusal_detected() -> None:
    assert Answer(question="q", text=REFUSAL, sources=[]).refused


def test_score_question() -> None:
    expected = {("OUG 195/2002", "36"), ("OUG 195/2002", "108")}
    retrieved = [
        make_chunk("OUG 195/2002", "99"),
        make_chunk("OUG 195/2002", "36"),
        make_chunk("HG 1391/2006", "97"),
    ]
    scores = score_question(expected, retrieved)
    assert scores["hit@1"] == 0 and scores["hit@3"] == 1
    assert scores["recall@3"] == 0.5
    assert scores["rr"] == 0.5


def test_find_references() -> None:
    assert find_references("Ce prevede art. 102 alin. (3) lit. e) din OUG 195/2002?") == [
        ("OUG 195/2002", "102", "3", "e")
    ]
    assert find_references("Ce spune articolul 31 din OG 2/2001?") == [("OG 2/2001", "31", "", "")]
    assert find_references("art. 142 din regulament") == [("HG 1391/2006", "142", "", "")]
    assert find_references("art 28^1 alin 2") == [("OUG 195/2002", "28^1", "2", "")]
    assert find_references("Cât e amenda?") == []


def test_citation_includes_list_item() -> None:
    chunk = make_chunk("OUG 195/2002", "101", "3")
    chunk.metadata["subdiviziune"] = "lit. e)"
    assert citation(chunk) == "[OUG 195/2002, art. 101 alin. (3) lit. e)]"
