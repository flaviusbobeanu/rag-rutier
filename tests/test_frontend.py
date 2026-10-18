"""Streamlit UI tests with AppTest (runs the script without a browser). The backend is faked by replacing
httpx.get/httpx.post, so neither the API nor an LLM is needed."""

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

HEALTH = {"chunks": 891, "llm": "fake", "law_versions": {"OUG 195/2002": "2026-08-25"}}
ANSWER = {
    "answer": "Amendă clasa a II-a [OUG 195/2002, art. 100 alin. (3)].",
    "standalone_question": "Ce amendă primesc dacă trec pe roșu?",
    "refused": False,
    "sources": [{"citation": "[OUG 195/2002, art. 100 alin. (3)]", "score": 0.71, "text": "(3) ..."}],
    "versions": {"OUG 195/2002": "2026-08-25"},
}


def response(method: str, url: str, status: int, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request(method, url))


@pytest.fixture
def requests(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Fake backend; returns the list of JSON bodies sent to /ask."""
    sent: list[dict] = []
    st.cache_data.clear()  # health() is cached between tests otherwise
    monkeypatch.setattr(httpx, "get", lambda url, **kwargs: response("GET", url, 200, HEALTH))

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        sent.append(kwargs.get("json"))
        return response("POST", url, 200, ANSWER)

    monkeypatch.setattr(httpx, "post", fake_post)
    return sent


def test_shows_disclaimer_and_server_status(requests: list[dict]) -> None:
    at = AppTest.from_file("../frontend/app.py").run()
    assert "nu este un sfat juridic" in at.warning[0].value
    assert "891 fragmente" in at.sidebar.success[0].value


def test_question_shows_answer_versions_and_sources(requests: list[dict]) -> None:
    at = AppTest.from_file("../frontend/app.py").run()
    at.chat_input[0].set_value("Ce amendă primesc dacă trec pe roșu?").run()
    assert ANSWER["answer"] in [m.value for m in at.markdown]
    assert any("forma consolidată la 2026-08-25" in c.value for c in at.caption)
    assert at.expander[0].label == "Surse (1)"


def test_follow_up_sends_history(requests: list[dict]) -> None:
    at = AppTest.from_file("../frontend/app.py").run()
    at.chat_input[0].set_value("Ce amendă primesc dacă trec pe roșu?").run()
    at.chat_input[0].set_value("Și dacă fac și accident?").run()
    assert [m["role"] for m in requests[1]["history"]] == ["user", "assistant"]


def test_server_down_shows_error(requests: list[dict], monkeypatch: pytest.MonkeyPatch) -> None:
    def down(url: str, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", down)
    at = AppTest.from_file("../frontend/app.py").run()
    at.chat_input[0].set_value("Ce amendă primesc?").run()
    assert "Nu pot contacta serverul" in at.error[0].value
    assert at.session_state.messages == []  # the failed question is not kept in the conversation
