"""The law documents used by the assistant: one entry per consolidated text in data/raw/."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LawDocument:
    file_stem: str  # file name in data/raw/ without ".html"
    citation: str  # how the law is cited in answers, e.g. "OUG 195/2002"
    id_prefix: str  # prefix for chunk IDs, e.g. "oug195"
    page_id: int  # page ID of this consolidated version on legislatie.just.ro
    version: str  # consolidation date, YYYY-MM-DD


# To update a law: pick the newest consolidation on its page, change page_id and version,
# rerun the download, and update data/raw/SOURCES.md.
DOCUMENTS: list[LawDocument] = [
    LawDocument("oug_195_2002", "OUG 195/2002", "oug195", 312713, "2026-08-25"),
    # The Regulament is the annex of HG 1391/2006; the HG itself has only a few articles.
    LawDocument("hg_1391_2006_regulament", "HG 1391/2006", "hg1391", 285008, "2024-07-19"),
    LawDocument("og_2_2001", "OG 2/2001", "og2", 307794, "2026-03-27"),
]
