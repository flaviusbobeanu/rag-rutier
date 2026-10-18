"""Tests for parsing the law HTML and chunking articles."""

from src.ingest.build_chunks import chunk_article, split_list
from src.ingest.parse_html import Article, parse_html
from src.ingest.sources import LawDocument

# A tiny law in the same format as legislatie.just.ro, with the noise we need to remove.
SAMPLE_HTML = """
<span class="S_CAP"><span class="S_CAP_TTL">Capitolul I</span><span class="S_CAP_DEN">Dispoziţii generale</span>
<span class="S_CAP_BDY">
  <span class="S_ART"><span class="S_ART_TTL">Articolul 5^1</span><span class="S_ART_BDY">
    <span class="S_ALN"><span class="S_ALN_TTL">(1)</span><span class="S_ALN_BDY">
      Conducătorii sunt obligaţi să respecte <span class="S_LGI">alin. (2)</span> , astfel:
      <span class="S_LIT"><span class="S_LIT_TTL">a)</span><span class="S_LIT_BDY">să oprească;
        <span class="S_PAR">(la 01-02-2020, Lit. a) a fost modificată de OUG nr. 1/2020)</span>
      </span><span class="S_LIT_SHORT">...</span></span>
      <span class="S_LIT"><span class="S_LIT_TTL">b)</span><span class="S_LIT_BDY">abrogată;
        <span class="S_PAR">(la 01-02-2021, Lit. b) a fost abrogată de OUG nr. 2/2021)</span>
      </span></span>
    </span></span>
    <span class="S_ALN"><span class="S_ALN_TTL">(2)</span><span class="S_ALN_BDY">Text doi.</span></span>
    <span class="S_NTA"><span class="S_NTA_TTL">Notă</span><span class="S_NTA_PAR">Notă editorială.</span></span>
  </span></span>
  <span class="S_ART"><span class="S_ART_TTL">Articolul 6</span><span class="S_ART_BDY">abrogat.
    <span class="S_PAR">(la 01-01-2019, Art. 6 a fost abrogat de Legea nr. 3/2019)</span>
  </span></span>
  <span class="S_ART"><span class="S_ART_TTL">Articolul 7</span><span class="S_ART_BDY">
    <span class="S_ALN"><span class="S_ALN_TTL">(1)</span><span class="S_ALN_BDY">Sancțiunile sunt:</span></span>
    <span class="S_LIT"><span class="S_LIT_TTL">a)</span><span class="S_LIT_BDY">amenda;</span></span>
    <span class="S_LIT"><span class="S_LIT_TTL">b)</span><span class="S_LIT_BDY">avertismentul.</span></span>
    <span class="S_ALN"><span class="S_ALN_TTL">(2)</span><span class="S_ALN_BDY">Text doi.
      ----------- Alin. (2) al art. 7 a fost modificat de Legea nr. 4/2012.</span></span>
  </span></span>
</span></span>
"""

DOC = LawDocument("test", "OUG 195/2002", "oug195", 0, "2026-01-01")


def test_skips_repealed_article_and_reads_superscript_number() -> None:
    articles = parse_html(SAMPLE_HTML)
    assert [a.number for a in articles] == ["5^1", "7"]


def test_letters_after_alineat_belong_to_it_and_old_notes_are_removed() -> None:
    article = parse_html(SAMPLE_HTML)[1]
    assert article.intro == ""
    assert article.alineate == [
        ("1", "(1) Sancțiunile sunt:\na) amenda;\nb) avertismentul."),
        ("2", "(2) Text doi."),
    ]


def test_cleans_text() -> None:
    article = parse_html(SAMPLE_HTML)[0]
    assert article.capitol == "Capitolul I - Dispoziții generale"
    assert article.alineate == [
        ("1", "(1) Conducătorii sunt obligați să respecte alin. (2), astfel:\na) să oprească;"),
        ("2", "(2) Text doi."),
    ]
    assert "a fost modificat" not in article.text
    assert "Notă" not in article.text


def test_short_article_is_one_chunk() -> None:
    chunks = chunk_article(DOC, parse_html(SAMPLE_HTML)[0])
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "oug195_art5_1"
    assert chunks[0]["alineat"] is None
    assert chunks[0]["text"].startswith("OUG 195/2002, art. 5^1 (Capitolul I - Dispoziții generale)\n")


def test_long_article_is_split_per_alineat() -> None:
    article = Article("10", "", "", "", [("1", "(1) " + "a" * 1000), ("2", "(2) " + "b" * 1000)])
    chunks = chunk_article(DOC, article)
    assert [c["chunk_id"] for c in chunks] == ["oug195_art10_alin1", "oug195_art10_alin2"]
    assert chunks[1]["text"].startswith("OUG 195/2002, art. 10 alin. (2)\n(2) bbb")


def test_split_list_keeps_intro_and_letter_context() -> None:
    text = (
        "(1) Se aplică puncte de penalizare, după cum urmează:\n"
        "a) 2 puncte pentru:\n"
        "1. fapta unu;\n"
        "2. fapta doi;\n"
        "b) 3 puncte pentru fapta trei;\n"
        "continuare a literei b."
    )
    intro, items = split_list(text)
    assert intro == "(1) Se aplică puncte de penalizare, după cum urmează:"
    assert [label for label, _ in items] == ["lit. a) pct. 1", "lit. a) pct. 2", "lit. b)"]
    assert items[0][1] == "a) 2 puncte pentru:\n1. fapta unu;"  # the letter line gives the points
    assert items[2][1].endswith("continuare a literei b.")
