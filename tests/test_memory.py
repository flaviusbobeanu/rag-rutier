"""Tests for rewriting follow-up questions (with a fake LLM, no API calls)."""

from src.memory.rewrite import MAX_TURNS, rewrite_question


class FakeLLM:
    name = "fake"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[str] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append(user)
        return self.reply


def test_no_history_returns_question_without_calling_llm() -> None:
    llm = FakeLLM("should not be used")
    assert rewrite_question("Ce viteză am voie în localitate?", [], llm) == "Ce viteză am voie în localitate?"
    assert llm.calls == []


def test_follow_up_is_rewritten_using_history() -> None:
    llm = FakeLLM('"Ce sancțiune primesc dacă trec pe roșu și produc un accident?"\n')
    history = [
        {"role": "user", "content": "Ce sancțiune primesc dacă trec pe roșu?"},
        {"role": "assistant", "content": "Amendă clasa a II-a și suspendare 30 de zile."},
    ]
    result = rewrite_question("Și dacă produc un accident?", history, llm)
    assert result == "Ce sancțiune primesc dacă trec pe roșu și produc un accident?"  # quotes stripped
    assert "Utilizator: Ce sancțiune primesc dacă trec pe roșu?" in llm.calls[0]
    assert "Latest question: Și dacă produc un accident?" in llm.calls[0]


def test_only_recent_history_is_sent() -> None:
    llm = FakeLLM("q")
    history = [{"role": "user", "content": f"mesaj {i}"} for i in range(MAX_TURNS + 4)]
    rewrite_question("și?", history, llm)
    assert "mesaj 0" not in llm.calls[0] and f"mesaj {MAX_TURNS + 3}" in llm.calls[0]
