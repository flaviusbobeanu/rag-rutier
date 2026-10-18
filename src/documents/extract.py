"""Get the text out of an uploaded PDF, entirely in memory.

Digitally generated PDFs contain text, which pdfplumber reads directly. Scanned PDFs are only images,
so if a page has (almost) no text we render it and run OCR with Tesseract and its Romanian model.
"""

import io

import pdfplumber

from src.config import settings

MIN_CHARS_PER_PAGE = 50  # less than this on a page means it is probably a scan
OCR_RESOLUTION = 300  # dpi; Tesseract works best around 300


class OCRUnavailableError(RuntimeError):
    """Raised when a scanned PDF needs OCR but Tesseract is not installed."""


def ocr_page(page: pdfplumber.page.Page) -> str:
    """Render one page to an image and read it with Tesseract (Romanian language model)."""
    import pytesseract  # imported here so text PDFs work even without OCR set up

    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    image = page.to_image(resolution=OCR_RESOLUTION).original
    try:
        return pytesseract.image_to_string(image, lang="ron")
    except (pytesseract.TesseractNotFoundError, pytesseract.TesseractError) as error:
        raise OCRUnavailableError(
            "This PDF is a scan and needs OCR. Install Tesseract with the Romanian language pack "
            "(see README) and make sure `tesseract` is on PATH."
        ) from error


def extract_text(pdf_bytes: bytes) -> tuple[str, str]:
    """Return (text, method) where method is "text", "ocr" or "text+ocr". Nothing is written to disk
    by this function (note: pytesseract itself uses a temporary image file that it deletes right away)."""
    pages: list[str] = []
    methods: set[str] = set()
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if len(text.strip()) < MIN_CHARS_PER_PAGE:
                text = ocr_page(page)
                methods.add("ocr")
            else:
                methods.add("text")
            pages.append(text)
    return "\n".join(pages), "+".join(sorted(methods, reverse=True)) or "text"
