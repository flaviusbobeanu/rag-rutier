# Sources

All texts come from the Portal Legislativ of the Romanian Ministry of Justice (legislatie.just.ro), in consolidated form ("forma consolidată"). They were downloaded on 2026-10-17 with `python -m src.ingest.download`, which saves the "Forma printabilă" HTML of each version.

| File | Act | Consolidation date | Page |
|---|---|---|---|
| `oug_195_2002.html` | OUG 195/2002 privind circulația pe drumurile publice, republicată (Codul rutier) | 2026-08-25 | https://legislatie.just.ro/Public/DetaliiDocument/312713 |
| `hg_1391_2006_regulament.html` | Regulamentul de aplicare a OUG 195/2002, aprobat prin HG 1391/2006 | 2024-07-19 | https://legislatie.just.ro/Public/DetaliiDocument/285008 |
| `og_2_2001.html` | OG 2/2001 privind regimul juridic al contravențiilor | 2026-03-27 | https://legislatie.just.ro/Public/DetaliiDocument/307794 |

Notes:
- HG 1391/2006 itself has only a few articles. The actual rules are in its annex, the Regulament, which the portal stores as a separate document. We download the Regulament.
- Each consolidation date has its own page ID. To update, open the act's page, pick the newest date under "Forma consolidată + istoric consolidări", and put that page ID in `src/ingest/download.py`.
- Consolidated texts are informative. The legally binding text is the one published in Monitorul Oficial.
