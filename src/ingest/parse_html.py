"""Parse the printable-form HTML from legislatie.just.ro into clean articles.

The site marks every part of a law with a CSS class:
    S_CAP chapter, S_SEC section, S_ART article, S_ALN paragraph (alineat),
    S_PCT numbered point, S_LIT letter, S_NTA editorial note, S_ANX annex.
Each of them has a _TTL child (title, e.g. "Articolul 5" or "(1)"), a _BDY child (content)
and sometimes a _DEN child (name, e.g. a chapter's name).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Comment, Tag

# Amendment notes look like "(la 01-09-2014, Pct. 1 ... a fost modificat de ...)" and are
# separate elements. Older ones look like "----------- Alin. (6) ... a fost modificat de ..."
# and are glued to the end of the law text, so they are cut out of the text instead.
AMENDMENT_NOTE = re.compile(r"\s*\(la \d{2}-\d{2}-\d{4}")
OLD_AMENDMENT_NOTE = re.compile(r"\s*-{5,}.*")
REPEALED = re.compile(r"(?i)(abrogat|eliminat)[ăe]?\s*[.;]?")
ARTICLE_TITLE = re.compile(r"Articolul\s+(\d+(?:\^\d+)?)")
ALINEAT_TITLE = re.compile(r"\((\d+(?:\^\d+)?)\)")

# Old cedilla letters (ş ţ) -> correct Romanian comma letters (ș ț), so the same word
# is always spelled the same way. Otherwise keyword search sees two different words.
CEDILLA_TO_COMMA = str.maketrans("şţŞŢ", "șțȘȚ")

# Marks the line breaks we want to keep. We can't use "\n", because the HTML source
# also contains newlines that mean nothing.
LINE_BREAK = "\x1e"


@dataclass
class Article:
    number: str  # "102" or "102^1"
    capitol: str  # e.g. "Capitolul VII - Răspunderea juridică", "" if none
    sectiune: str  # e.g. "Secțiunea 1 - ...", "" if none
    intro: str  # article text that is not inside a numbered paragraph
    alineate: list[tuple[str, str]] = field(default_factory=list)  # [("1", "(1) ..."), ...]

    @property
    def text(self) -> str:
        """The full article text: intro followed by all paragraphs."""
        parts = [self.intro] if self.intro else []
        parts += [text for _, text in self.alineate]
        return "\n".join(parts)


def normalize(text: str) -> str:
    """Fix Romanian letters, collapse whitespace, drop old notes, remove spaces before punctuation."""
    text = " ".join(text.translate(CEDILLA_TO_COMMA).split())
    text = OLD_AMENDMENT_NOTE.sub("", text)
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    return re.sub(r"([(\[])\s+", r"\1", text)


def clean_text(element: Tag) -> str:
    """Text of an element, with each point ("1.") and letter ("a)") on its own line."""
    for title in element.select(".S_PCT_TTL, .S_LIT_TTL"):
        title.insert_before(LINE_BREAK)
    lines = (normalize(line) for line in element.get_text(" ").split(LINE_BREAK))
    return "\n".join(line for line in lines if line)


def heading(element: Tag | None, cls: str) -> str:
    """Title and name of a chapter or section, e.g. "Capitolul I - Dispoziții generale"."""
    if element is None:
        return ""
    parts = [element.find(class_=f"{cls}_TTL"), element.find(class_=f"{cls}_DEN")]
    return " - ".join(text for text in (normalize(p.get_text(" ")) for p in parts if p) if text)


def remove_noise(soup: BeautifulSoup) -> None:
    """Delete everything that is not law text: notes, annexes, amendment notes, repealed items."""
    for element in soup.select(".S_NTA, .S_ANX, [class$='_SHORT']"):
        element.decompose()

    for par in soup.select(".S_PAR"):
        if not par.decomposed and AMENDMENT_NOTE.match(par.get_text()):
            par.decompose()

    # Deepest elements first, so a paragraph whose letters were all repealed is seen as empty.
    for cls in ("S_LIT", "S_PCT", "S_ALN"):
        for element in soup.select(f".{cls}"):
            body = element.find(class_=f"{cls}_BDY")
            if body is not None and REPEALED.fullmatch(normalize(body.get_text(" "))):
                element.decompose()


def parse_html(html: str) -> list[Article]:
    """Parse one law's HTML into articles. Fully repealed articles are skipped."""
    soup = BeautifulSoup(html, "html.parser")
    remove_noise(soup)

    articles = []
    for art in soup.select(".S_ART"):
        title, body = art.find(class_="S_ART_TTL"), art.find(class_="S_ART_BDY")
        match = ARTICLE_TITLE.search(title.get_text()) if title else None
        if match is None or body is None:
            continue

        # Walk the body in order. Each "(n)" starts a new alineat; the letters and points
        # after it in the HTML are siblings, not children, but they belong to that alineat.
        intro_parts: list[str] = []
        alineate_parts: list[tuple[str, list[str]]] = []
        for child in body.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, Tag) and "S_ALN" in child.get("class", []):
                aln_title = child.find(class_="S_ALN_TTL")
                aln_match = ALINEAT_TITLE.search(aln_title.get_text()) if aln_title else None
                if aln_match:
                    alineate_parts.append((aln_match.group(1), [clean_text(child)]))
                    continue
            text = clean_text(child) if isinstance(child, Tag) else normalize(child)
            if text:
                (alineate_parts[-1][1] if alineate_parts else intro_parts).append(text)

        intro = "\n".join(intro_parts)
        alineate = [(number, "\n".join(parts)) for number, parts in alineate_parts]
        if not alineate and (not intro or REPEALED.fullmatch(intro)):
            continue

        articles.append(
            Article(
                number=match.group(1),
                capitol=heading(art.find_parent(class_="S_CAP"), "S_CAP"),
                sectiune=heading(art.find_parent(class_="S_SEC"), "S_SEC"),
                intro=intro,
                alineate=alineate,
            )
        )
    return articles


def parse_law(path: Path) -> list[Article]:
    """Parse a law HTML file from data/raw/."""
    return parse_html(path.read_text(encoding="utf-8"))
