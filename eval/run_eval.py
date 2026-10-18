"""End-to-end evaluation: answer the test questions, then score the answers.

Two groups of metrics:
  Our own (computed in code, free):
    answered        share of answerable questions that got an answer (not a refusal)
    refused_oos     share of out-of-scope questions that were correctly refused
    cites_expected  share of answered questions that cite at least one expected article
    citation_valid  share of all citations that point to an article that was in the retrieved context
  RAGAS (an LLM judge reads question, answer, contexts and reference answer):
    faithfulness       are the answer's claims supported by the retrieved contexts? (answered only)
    answer_relevancy   does the answer address the question? (answered only)
    context_precision  are the relevant chunks ranked high?
    context_recall     do the contexts contain everything in the reference answer?

Run:
    python -m eval.run_eval                                    # generate answers, then score
    python -m eval.run_eval eval/runs/answers_....jsonl        # score existing answers
    python -m eval.run_eval --limit 3 --no-ragas               # quick, cheap smoke test
"""

import argparse
import asyncio
import json
import math
from pathlib import Path

from eval.generate_answers import generate
from eval.ragas_adapters import ClaudeJudge, LocalEmbeddings
from src.config import settings
from src.generation.answer import DEFAULT_K


def mean(values: list[float]) -> float:
    """Average that ignores NaN (a metric that failed for one question)."""
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")


def own_metrics(rows: list[dict]) -> dict:
    """Refusal behaviour and citation quality, computed directly from the saved answers."""
    answerable = [r for r in rows if r["answerable"]]
    out_of_scope = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable if not r["refused"]]

    def cites_expected(row: dict) -> bool:
        expected = {(s["lege"], s["articol"]) for s in row["expected_sources"]}
        return any((lege, articol) in expected for lege, articol, _ in row["citations"])

    citations = [(r, c) for r in rows if not r["refused"] for c in r["citations"]]
    valid = [1 for r, (lege, articol, _) in citations if [lege, articol] in r["retrieved"]]
    return {
        "answered": len(answered) / len(answerable) if answerable else float("nan"),
        "refused_oos": mean([float(r["refused"]) for r in out_of_scope]),
        "cites_expected": mean([float(cites_expected(r)) for r in answered]),
        "citation_valid": len(valid) / len(citations) if citations else float("nan"),
    }


async def ragas_scores(row: dict, metrics: dict) -> dict:
    """Run the four RAGAS metrics on one answered question. A failing metric gives NaN."""
    calls = {
        "context_precision": lambda: metrics["context_precision"].ascore(
            user_input=row["question"], reference=row["expected_answer"], retrieved_contexts=row["contexts"]
        ),
        "context_recall": lambda: metrics["context_recall"].ascore(
            user_input=row["question"], retrieved_contexts=row["contexts"], reference=row["expected_answer"]
        ),
    }
    if not row["refused"]:
        calls["faithfulness"] = lambda: metrics["faithfulness"].ascore(
            user_input=row["question"], response=row["answer"], retrieved_contexts=row["contexts"]
        )
        calls["answer_relevancy"] = lambda: metrics["answer_relevancy"].ascore(
            user_input=row["question"], response=row["answer"]
        )

    scores = {}
    for name, call in calls.items():
        try:
            scores[name] = float((await call()).value)
        except Exception as error:  # noqa: BLE001 - one bad judge reply should not stop the whole eval
            print(f"  {row['id']} {name} failed: {error}")
            scores[name] = float("nan")
    return scores


async def run_ragas(rows: list[dict], judge_model: str) -> list[dict]:
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    judge = ClaudeJudge(judge_model)
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=LocalEmbeddings()),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }
    answerable = [r for r in rows if r["answerable"]]
    print(f"\nRAGAS on {len(answerable)} answerable questions, judge {judge_model} ...")
    return await asyncio.gather(*(ragas_scores(row, metrics) for row in answerable))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "answers", nargs="?", help="answers file from eval.generate_answers (omit to generate)"
    )
    parser.add_argument("--label", default="baseline")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--no-ragas", action="store_true", help="skip the (paid) RAGAS judge")
    parser.add_argument("--pipeline", choices=["manual", "llamaindex"], default="manual")
    parser.add_argument("--judge-model", default=settings.anthropic_model)
    args = parser.parse_args()

    path = Path(args.answers) if args.answers else generate(args.label, args.k, args.limit, args.pipeline)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    summary = own_metrics(rows)
    per_question = []
    if not args.no_ragas:
        per_question = asyncio.run(run_ragas(rows, args.judge_model))
        for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            summary[name] = mean([scores.get(name, float("nan")) for scores in per_question])

    print(f"\nResults for {path.name} ({rows[0]['llm']}, k={rows[0]['k']}):\n")
    print("| " + " | ".join(summary) + " |")
    print("|" + "---|" * len(summary))
    print("| " + " | ".join(f"{value:.2f}" for value in summary.values()) + " |")

    answerable_ids = [r["id"] for r in rows if r["answerable"]]
    out = {
        "answers_file": path.name,
        "llm": rows[0]["llm"],
        "judge_model": None if args.no_ragas else args.judge_model,
        "summary": summary,
        "ragas_per_question": dict(zip(answerable_ids, per_question)),
    }
    out_path = path.with_name(path.name.replace("answers_", "scores_").replace(".jsonl", ".json"))
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nScores saved to {out_path}")


if __name__ == "__main__":
    main()
