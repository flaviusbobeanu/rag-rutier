# Evaluation results

Every experiment is run on the same test set (`eval/questions.jsonl`) and logged here. Raw per-question output
for each run is in `eval/runs/`.

Test set: 46 questions. 41 answerable (37 everyday questions + 4 that name an article explicitly, q43-q46) and
5 out-of-scope ones that should be refused. With 41 questions, one question = 0.024, so differences below ~0.03
are noise.

## Retrieval

Command: `python -m eval.retrieval_eval --label <name> [--mode ...] [--no-refs] [--rerank]`. A hit means a chunk
from an expected article (law + article number) is in the top k. Only answerable questions count.

### Phase 4 baseline (37 everyday questions, before q43-q46 existed)

| Experiment | hit@1 | hit@3 | hit@5 | hit@10 | recall@5 | MRR |
|---|---|---|---|---|---|---|
| vector (bge-m3), per-article chunks | 0.51 | 0.89 | 0.92 | 0.97 | 0.89 | 0.692 |

### Phase 5 (41 answerable questions)

| Experiment | hit@1 | hit@3 | hit@5 | hit@10 | recall@5 | MRR | MRR everyday | MRR article refs |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.32 | 0.51 | 0.63 | 0.71 | 0.60 | 0.434 | 0.447 | 0.312 |
| hybrid (vector + bm25, RRF) | 0.44 | 0.71 | 0.83 | 0.90 | 0.80 | 0.590 | 0.643 | 0.098 |
| vector | 0.51 | 0.88 | 0.88 | 0.90 | 0.84 | 0.678 | 0.724 | 0.250 |
| hybrid + reference lookup | 0.54 | 0.80 | 0.90 | 0.95 | 0.88 | 0.678 | 0.643 | 1.000 |
| **vector + reference lookup** | **0.59** | **0.95** | **0.95** | **0.98** | **0.91** | **0.751** | 0.724 | **1.000** |

Findings:
- **BM25 and hybrid search did not help.** The test questions are everyday paraphrases, where embeddings are
  strong and word overlap is weak (users say "faruri", the law says "lumini de întâlnire"; "ITP" never appears in
  the law). A sweep over the BM25 weight in the fusion (0, 0.2, 0.3, 0.5, 1.0) never beat vector alone
  (MRR 0.692 → 0.686 / 0.690 / 0.682 / 0.622 on the everyday questions).
- **BM25 does not solve article references either**, contrary to the original plan: "art", "alin", "OUG", "195",
  "2002" occur in every chunk header, and "102" also occurs in every article that *refers to* art. 102.
- **A direct reference lookup does**: a regex finds "art. X [alin. (Y)] [lit. z)] [din <law>]" in the question and
  the chunks are fetched by metadata. All 4 article-reference questions now rank the right article first, and the
  everyday questions are unaffected.
- Kept as default: vector + reference lookup. Hybrid stays available (`RETRIEVAL_MODE=hybrid`).

## End-to-end (generation + RAGAS)

Command: `python -m eval.run_eval --label <name>`. Metric definitions are in the docstring of `eval/run_eval.py`.
LLM: `claude-haiku-5-5` (effort medium), RAGAS judge: `claude-haiku-5-5` (effort low).

| Date | Experiment | Questions | answered | refused_oos | cites_expected | citation_valid | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-17 | baseline: vector, per-article chunks, k=5 | 42 | 1.00 | 0.80 | 0.92* | 1.00 | 0.94 | 0.80 | 0.77 | 0.87 |

\* 0.95 after correcting the labels of q32/q33 (see below); the run itself scored the old labels.

Notes on the baseline:
- **citation_valid = 1.00**: no answer cited an article that was not in the retrieved context.
- **q32**: the model cited OUG 195/2002 art. 118 instead of the expected OG 2/2001 art. 31. The model was right:
  art. 118 is the special rule for traffic fines and takes precedence over the general rule. Labels of q32 and q33
  were corrected to accept both.
- **q40** (prison sentence for drunk driving, which is in the Criminal Code, not in the corpus) counts as "not
  refused", but the answer says clearly that the texts do not contain it and gives the related contravention
  rule. That is the intended partial-answer behaviour; the `refused_oos` metric only detects the exact refusal
  sentence, so it is stricter than the real behaviour.
- The remaining misses (q12, q16) are retrieval misses: the right article was not in the top 5.
