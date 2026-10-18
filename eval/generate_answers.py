"""Run the full RAG pipeline on every test question and save the answers with their contexts.

Kept separate from scoring so the (paid) answers can be scored again without regenerating them.

Run: python -m eval.generate_answers --label baseline [--pipeline llamaindex]
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from eval.retrieval_eval import EVAL_DIR, load_questions
from src.config import settings
from src.generation.answer import DEFAULT_K, Answer, ask
from src.generation.llm import get_llm


def generate(label: str, k: int = DEFAULT_K, limit: int | None = None, pipeline: str = "manual") -> Path:
    """Answer the test questions and write one JSON line per question. Returns the file path.

    pipeline: "manual" (src.generation.answer) or "llamaindex" (src.llamaindex_pipeline).
    """
    llm = get_llm()
    llm_name = f"{pipeline}/{llm.name}"
    if pipeline == "llamaindex":
        from src.llamaindex_pipeline.pipeline import ask as llamaindex_ask  # heavy import, only when used

    def answer_fn(question: str, history: list[dict] | None) -> Answer:
        if pipeline == "llamaindex":
            return llamaindex_ask(question, history=history, k=k)
        return ask(question, history=history, k=k, llm=llm)

    questions = load_questions()[:limit]
    out_dir = EVAL_DIR / "runs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"answers_{label}_{datetime.now():%Y%m%d_%H%M}.jsonl"

    print(
        f"Answering {len(questions)} questions with {llm_name}, k={k}, embeddings {settings.embedding_model}"
    )
    with out_path.open("w", encoding="utf-8") as f:
        for q in questions:
            start = time.perf_counter()
            answer = answer_fn(q["question"], q.get("history"))
            row = {
                **q,
                "llm": llm_name,
                "k": k,
                "standalone_question": answer.standalone_question,
                "answer": answer.text,
                "refused": answer.refused,
                "citations": answer.citations,
                "contexts": [chunk.text for chunk in answer.sources],
                "retrieved": [[c.metadata["lege"], c.metadata["articol"]] for c in answer.sources],
                "seconds": round(time.perf_counter() - start, 1),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            status = "REFUSED" if answer.refused else f"{len(answer.citations)} citations"
            print(f"  {q['id']} ({row['seconds']}s) {status}")
    print(f"\nSaved answers to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help="chunks retrieved per question")
    parser.add_argument("--limit", type=int, help="only the first N questions (for a cheap trial run)")
    parser.add_argument("--pipeline", choices=["manual", "llamaindex"], default="manual")
    args = parser.parse_args()
    path = generate(args.label, args.k, args.limit, args.pipeline)
    print(f"Score it with: python -m eval.run_eval {path.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
