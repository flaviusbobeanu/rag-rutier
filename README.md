# RAG Rutier

A legal assistant that answers questions about Romanian traffic law strictly from the official law texts (OUG 195/2002, HG 1391/2006, OG 2/2001) and cites the exact article and paragraph for every claim.

> Not legal advice. This tool does not replace a lawyer.

Work in progress. Architecture, setup instructions and evaluation results will be added as the project grows.

## OCR for scanned fines (optional)

Uploaded fines that are digital PDFs are read directly. Scanned PDFs (images only) need OCR with
[Tesseract](https://github.com/tesseract-ocr/tesseract) and its Romanian language model:

1. Install Tesseract for Windows (the installer from UB Mannheim, linked from the Tesseract docs). In the
   installer, under *Additional language data*, select **Romanian**.
2. Add the install folder (usually `C:\Program Files\Tesseract-OCR`) to `PATH`, open a new terminal, and check:
   `tesseract --list-langs` should list `ron`. Instead of editing `PATH` you can also put
   `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` in `.env`.

To analyze a fine from the terminal: `python -m src.cli --fine path\to\pv.pdf`.

Without Tesseract, uploading a scanned PDF returns a clear error; everything else works.

Privacy: uploaded files are processed in memory and never saved. Only the violated article, amount, penalty
points and date are extracted (locally, with regular expressions) and used to build the question for the LLM;
names, CNP, addresses and plate numbers from the document are never sent to it.
