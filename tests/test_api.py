"""API tests with FastAPI's TestClient. The LLM and retrieval are faked, so no index or API key is needed."""

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.documents import extract as extract_module
from src.documents import fine as fine_module
from src.documents.extract import OCRUnavailableError
from src.generation import answer as answer_module
from src.retrieval.store import RetrievedChunk
from tests.test_documents import make_pdf

CHUNK = RetrievedChunk(
    chunk_id="oug195_art100_alin3",
    text="OUG 195/2002, art. 100 alin. (3)\n(3) ... amenda clasa a II-a ...",
    metadata={"lege": "OUG 195/2002", "articol": "100", "alineat": "3", "versiune": "2026-08-25"},
    score=0.71,
)


class FakeLLM:
    name = "fake"

    def generate(self, system: str, user: str) -> str:
        if "Latest question" in user:  # the query-rewrite prompt
            return "Ce sancțiune primesc dacă trec pe roșu și fac un accident?"
        return "Amendă clasa a II-a [OUG 195/2002, art. 100 alin. (3)]."


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(main, "llm", lambda: FakeLLM())
    monkeypatch.setattr(answer_module, "retrieve", lambda question, k: [CHUNK])
    return TestClient(main.app)


def test_ask_returns_answer_sources_versions(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "Ce sancțiune primesc dacă trec pe roșu?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Amendă clasa a II-a")
    assert body["sources"][0]["citation"] == "[OUG 195/2002, art. 100 alin. (3)]"
    assert body["versions"] == {"OUG 195/2002": "2026-08-25"}
    assert body["refused"] is False
    assert "sfat juridic" in body["disclaimer"]


def test_follow_up_is_rewritten(client: TestClient) -> None:
    history = [
        {"role": "user", "content": "Ce sancțiune primesc dacă trec pe roșu?"},
        {"role": "assistant", "content": "Amendă clasa a II-a."},
    ]
    response = client.post("/ask", json={"question": "Și dacă fac un accident?", "history": history})
    assert (
        response.json()["standalone_question"] == "Ce sancțiune primesc dacă trec pe roșu și fac un accident?"
    )


def test_invalid_request_is_rejected(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert (
        client.post("/ask", json={"question": "ok?", "history": [{"role": "x", "content": "y"}]}).status_code
        == 422
    )


def test_upload_pdf_returns_facts_and_explanation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fine_module, "fetch_article", lambda *reference: [CHUNK])
    pdf = make_pdf(["fapta prevazuta de art. 100 alin. (3) lit. d) din OUG 195/2002", "amenda de 1.215 lei"])
    response = client.post("/upload", files={"file": ("pv.pdf", pdf, "application/pdf")})
    assert response.status_code == 200
    body = response.json()
    assert body["facts"]["articles"] == ["art. 100 alin. (3) lit. d) din OUG 195/2002"]
    assert body["facts"]["amount_lei"] == 1215
    assert body["explanation"]["sources"][0]["citation"] == "[OUG 195/2002, art. 100 alin. (3)]"


def test_upload_rejects_non_pdf(client: TestClient) -> None:
    response = client.post("/upload", files={"file": ("pv.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_upload_scan_without_tesseract_returns_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_ocr(page: object) -> str:
        raise OCRUnavailableError("This PDF is a scan and needs OCR. Install Tesseract")

    monkeypatch.setattr(extract_module, "ocr_page", no_ocr)
    response = client.post("/upload", files={"file": ("scan.pdf", make_pdf([]), "application/pdf")})
    assert response.status_code == 422
    assert "Tesseract" in response.json()["detail"]
