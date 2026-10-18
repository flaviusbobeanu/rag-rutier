"""Analyze a traffic fine (proces-verbal de contravenție): extract the key facts, then explain them.

Privacy: the PDF contains personal data (name, CNP, address, plate number). Only the violated article,
the amount, the penalty points and the date are extracted, locally, with regular expressions. The LLM
only ever sees a question built from those facts, never the document text. Nothing is stored.
"""

import re
from dataclasses import dataclass, field

from src.generation.answer import Answer
from src.generation.llm import LLM, get_llm
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retrieval.references import ARTICLE_RE, DEFAULT_LAW, LAW_PATTERNS, fetch_article
from src.retrieval.store import RetrievedChunk

# Laws whose articles describe offences. References to OG 2/2001 in a fine are procedural (rights, deadlines).
OFFENCE_LAWS = {"OUG 195/2002", "HG 1391/2006"}

# Procedure that applies to every traffic fine: paying half the minimum, and contesting it.
PROCEDURE_ARTICLES = [
    ("OG 2/2001", "28", "1", ""),
    ("OUG 195/2002", "118", "1", ""),
    ("OG 2/2001", "31", "", ""),
]

AMOUNT_RE = re.compile(r"(\d{1,3}(?:[.\s]\d{3})+|\d+)(?:,\d{1,2})?\s*(?:lei|ron)\b", re.IGNORECASE)
POINTS_RE = re.compile(r"(\d+)\s*puncte\s*(?:de\s*)?penalizare", re.IGNORECASE)
DATE_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")


@dataclass
class FineFacts:
    articles: list[tuple[str, str, str, str]] = field(
        default_factory=list
    )  # (lege, articol, alineat, litera)
    amount_lei: int | None = None
    penalty_points: int | None = None
    date: str | None = None  # YYYY-MM-DD, the first date in the document (usually the date of the offence)


@dataclass
class FineAnalysis:
    facts: FineFacts
    text_method: str  # "text", "ocr" or "text+ocr"
    answer: Answer | None  # None if no article could be found


def law_after(text: str, position: int) -> str:
    """The law named right after a reference ("art. 100 ... din OUG 195/2002"), else the traffic code."""
    window = text[position : position + 80]
    for name, pattern in LAW_PATTERNS:
        if re.match(rf"[^.;]*?(?:{pattern})", window, re.IGNORECASE):
            return name
    return DEFAULT_LAW


def extract_facts(text: str) -> FineFacts:
    """Find the violated articles, amount, penalty points and date in the fine's text."""
    facts = FineFacts()
    for match in ARTICLE_RE.finditer(text):
        articol, alineat, litera = match.groups()
        reference = (law_after(text, match.end()), articol, alineat or "", (litera or "").lower())
        if reference[0] in OFFENCE_LAWS and reference not in facts.articles:
            facts.articles.append(reference)

    # The fine's amount is the first amount after the word "amendă"; fall back to the first amount at all.
    fine_word = re.search(r"amend", text, re.IGNORECASE)
    amount = AMOUNT_RE.search(text, fine_word.start() if fine_word else 0) or AMOUNT_RE.search(text)
    if amount:
        facts.amount_lei = int(re.sub(r"[.\s]", "", amount.group(1)))
    if points := POINTS_RE.search(text):
        facts.penalty_points = int(points.group(1))
    if date := DATE_RE.search(text):
        day, month, year = date.groups()
        facts.date = f"{year}-{int(month):02d}-{int(day):02d}"
    return facts


def describe(reference: tuple[str, str, str, str]) -> str:
    lege, articol, alineat, litera = reference
    return (
        f"art. {articol}"
        + (f" alin. ({alineat})" if alineat else "")
        + (f" lit. {litera})" if litera else "")
        + f" din {lege}"
    )


def build_question(facts: FineFacts) -> str:
    """A question that contains the facts of the fine but no personal data."""
    question = "Am primit un proces-verbal de contravenție pentru fapta prevăzută de "
    question += ", ".join(describe(ref) for ref in facts.articles)
    if facts.amount_lei:
        question += f", cu o amendă de {facts.amount_lei} lei"
    if facts.penalty_points:
        question += f" și {facts.penalty_points} puncte de penalizare"
    return question + (
        ". Explică-mi ce faptă este, ce sancțiuni prevede legea pentru ea și ce opțiuni am: "
        "dacă pot plăti jumătate din minimul amenzii și în ce termen, și cum și în ce termen pot contesta."
    )


def analyze_fine(text: str, text_method: str = "text", llm: LLM | None = None) -> FineAnalysis:
    """Extract the facts from the fine's text and explain them using the law."""
    facts = extract_facts(text)
    if not facts.articles:
        return FineAnalysis(facts=facts, text_method=text_method, answer=None)

    chunks: list[RetrievedChunk] = []
    for reference in facts.articles[:3] + PROCEDURE_ARTICLES:
        chunks += [
            chunk for chunk in fetch_article(*reference) if chunk.chunk_id not in {c.chunk_id for c in chunks}
        ]

    question = build_question(facts)
    llm = llm or get_llm()
    text_answer = llm.generate(SYSTEM_PROMPT, build_user_prompt(question, chunks))
    answer = Answer(question=question, text=text_answer, sources=chunks, standalone_question=question)
    return FineAnalysis(facts=facts, text_method=text_method, answer=answer)
