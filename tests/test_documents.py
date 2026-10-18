"""Tests for PDF text extraction and fine analysis. The fine below is fictional."""

import pytest

from src.config import settings
from src.documents import extract as extract_module
from src.documents import fine as fine_module
from src.documents.extract import OCRUnavailableError, extract_text
from src.documents.fine import analyze_fine, build_question, extract_facts
from src.retrieval.store import RetrievedChunk

FINE_TEXT = """PROCES-VERBAL DE CONSTATARE A CONTRAVENȚIEI seria PCC nr. 0000000
Încheiat astăzi 14.03.2026, ora 10:20. Subsemnatul agent Ion Agentul, am constatat că
numitul Popescu Ionel Fictiv, CNP 1900101000000, domiciliat în București, Str. Exemplului nr. 1,
la data de 14.03.2026 a condus autoturismul cu nr. de înmatriculare B-00-XYZ și a folosit
telefonul mobil fără dispozitiv mâini libere, concomitent cu nerespectarea unei reguli de circulație,
faptă prevăzută de art. 100 alin. (3) lit. h) din O.U.G. nr. 195/2002 și sancționată de art. 100 alin. 3
din OUG 195/2002 cu amendă în cuantum de 2.025 lei și 4 puncte de penalizare.
Contravenientul poate achita jumătate din minimul amenzii conform art. 28 din OG 2/2001."""


def test_extract_facts_from_fine() -> None:
    facts = extract_facts(FINE_TEXT)
    assert facts.articles == [("OUG 195/2002", "100", "3", "h"), ("OUG 195/2002", "100", "3", "")]
    assert facts.amount_lei == 2025
    assert facts.penalty_points == 4
    assert facts.date == "2026-03-14"


def test_question_contains_no_personal_data() -> None:
    question = build_question(extract_facts(FINE_TEXT))
    assert "art. 100 alin. (3) lit. h) din OUG 195/2002" in question
    for personal in ("Popescu", "1900101000000", "B-00-XYZ", "Exemplului", "PCC"):
        assert personal not in question


def test_analyze_fine_uses_violated_and_procedure_articles(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[tuple] = []

    def fake_fetch(lege: str, articol: str, alineat: str = "", litera: str = "") -> list[RetrievedChunk]:
        fetched.append((lege, articol))
        meta = {"lege": lege, "articol": articol, "alineat": alineat, "versiune": "2026-08-25"}
        return [RetrievedChunk(f"{lege}_{articol}_{alineat}{litera}", "text", meta, 1.0)]

    class FakeLLM:
        name = "fake"

        def generate(self, system: str, user: str) -> str:
            assert "Popescu" not in user and "1900101000000" not in user
            return "Explicație [OUG 195/2002, art. 100 alin. (3)]."

    monkeypatch.setattr(fine_module, "fetch_article", fake_fetch)
    analysis = analyze_fine(FINE_TEXT, llm=FakeLLM())
    assert ("OUG 195/2002", "100") in fetched
    assert ("OG 2/2001", "28") in fetched and ("OUG 195/2002", "118") in fetched
    assert analysis.answer is not None and analysis.answer.citations


def test_no_article_found_gives_no_answer() -> None:
    assert analyze_fine("Un document fără nicio referință la lege.", llm=None).answer is None


def make_pdf(lines: list[str]) -> bytes:
    """A minimal one-page PDF with text (ASCII only), built by hand so no PDF library is needed."""
    # PDF strings are wrapped in ( ), so parentheses inside the text must be escaped.
    escaped = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
    stream = "BT /F1 12 Tf 50 750 Td 14 TL " + " ".join(f"({line}) '" for line in escaped) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf, offsets = "%PDF-1.4\n", []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n{body}\nendobj\n"
    xref = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n" + "".join(
        f"{o:010d} 00000 n \n" for o in offsets
    )
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    return pdf.encode("latin-1")


def test_extract_text_from_text_pdf() -> None:
    lines = ["PROCES-VERBAL DE CONSTATARE A CONTRAVENTIEI", "fapta prevazuta de art. 100 alin. (3) lit. d)",
             "din OUG 195/2002, amenda in cuantum de 1.215 lei"]  # fmt: skip
    text, method = extract_text(make_pdf(lines))
    assert method == "text"
    assert "art. 100 alin. (3) lit. d)" in text
    assert extract_facts(text).amount_lei == 1215


def test_page_without_text_goes_to_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    # A page with no text layer is what a scan looks like; the fake OCR stands in for Tesseract.
    monkeypatch.setattr(extract_module, "ocr_page", lambda page: FINE_TEXT)
    text, method = extract_text(make_pdf([]))
    assert method == "ocr"
    assert extract_facts(text).articles[0] == ("OUG 195/2002", "100", "3", "h")


def test_missing_tesseract_gives_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "tesseract_cmd", "no-such-tesseract-program")
    with pytest.raises(OCRUnavailableError, match="Tesseract"):
        extract_text(make_pdf([]))
