"""Measure how well retrieval finds the expected articles for each test question (no LLM needed).

Metrics, over the answerable questions in eval/questions.jsonl:
    hit@k     share of questions where at least one expected article is in the top k chunks
    recall@k  average share of a question's expected articles found in the top k chunks
    MRR       mean of 1/rank of the first chunk from an expected article (0 if not in the top 10)

A retrieved chunk matches if its law and article number match; the paragraph is ignored.

Run: python -m eval.retrieval_eval --label baseline [--mode vector|bm25|hybrid]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.config import settings
from src.retrieval.search import retrieve
from src.retrieval.store import RetrievedChunk

EVAL_DIR = Path(__file__).parent
KS = [1, 3, 5, 10]


def load_questions() -> list[dict]:
    """Read the test set from eval/questions.jsonl."""
    lines = (EVAL_DIR / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def article_key(chunk: RetrievedChunk) -> tuple[str, str]:
    return chunk.metadata["lege"], chunk.metadata["articol"]


def score_question(expected: set[tuple[str, str]], retrieved: list[RetrievedChunk]) -> dict:
    """Hit, recall and reciprocal rank for one question, given the ranked retrieved chunks."""
    keys = [article_key(chunk) for chunk in retrieved]
    result = {}
    for k in KS:
        found = expected & set(keys[:k])
        result[f"hit@{k}"] = float(bool(found))
        result[f"recall@{k}"] = len(found) / len(expected)
    first = next((rank for rank, key in enumerate(keys, start=1) if key in expected), None)
    result["rr"] = 1 / first if first else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label", default="baseline", help="name of this experiment, used in the output file"
    )
    parser.add_argument("--mode", default=settings.retrieval_mode, help="vector, bm25 or hybrid")
    parser.add_argument("--no-refs", action="store_true", help="disable the direct article-reference lookup")
    parser.add_argument("--rerank", action="store_true", help="re-order the top 20 with the cross-encoder")
    args = parser.parse_args()

    # Follow-up questions need the LLM rewrite step first, so they are only in the end-to-end eval.
    questions = [q for q in load_questions() if q["answerable"] and not q.get("history")]
    rows = []
    for q in questions:
        expected = {(s["lege"], s["articol"]) for s in q["expected_sources"]}
        retrieved = retrieve(
            q["question"],
            k=max(KS),
            mode=args.mode,
            use_references=not args.no_refs,
            use_reranker=args.rerank,
        )
        scores = score_question(expected, retrieved)
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "expected": sorted(expected),
                "retrieved": [f"{c.metadata['lege']} art. {c.metadata['articol']}" for c in retrieved],
                **scores,
            }
        )

    metric_names = [f"{m}@{k}" for m in ("hit", "recall") for k in KS] + ["rr"]
    summary = {name: sum(row[name] for row in rows) / len(rows) for name in metric_names}

    print(
        f"Retrieval eval '{args.label}' ({args.mode}, {settings.embedding_model}) on {len(rows)} questions\n"
    )
    print("| k | hit@k | recall@k |")
    print("|---|---|---|")
    for k in KS:
        print(f"| {k} | {summary[f'hit@{k}']:.2f} | {summary[f'recall@{k}']:.2f} |")
    print(f"\nMRR: {summary['rr']:.3f}")

    misses = [row for row in rows if row["hit@5"] == 0]
    print(f"\nQuestions with no expected article in the top 5 ({len(misses)}):")
    for row in misses:
        print(
            f"  {row['id']} {row['question']}\n      expected {row['expected']}\n      got      {row['retrieved'][:5]}"
        )

    out_dir = EVAL_DIR / "runs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"retrieval_{args.label}_{datetime.now():%Y%m%d_%H%M}.json"
    out = {
        "label": args.label,
        "mode": args.mode,
        "references": not args.no_refs,
        "reranker": args.rerank,
        "collection": settings.collection_name,
        "embedding_model": settings.embedding_model,
        "summary": summary,
        "questions": rows,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetails saved to {out_path}")


if __name__ == "__main__":
    main()
