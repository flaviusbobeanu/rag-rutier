"""Ask questions about Romanian traffic law from the terminal.

Run:
    python -m src.cli "Ce amendă primesc dacă trec pe roșu?"   # one question
    python -m src.cli                                          # interactive with memory, empty line to quit
    python -m src.cli -k 8 --sources "..."                     # more chunks, print their text
    python -m src.cli --fine pv.pdf                            # analyze a fine (proces-verbal) PDF
"""

import argparse
from pathlib import Path

from src.documents.extract import extract_text
from src.documents.fine import analyze_fine, describe
from src.generation.answer import DEFAULT_K, Answer, ask
from src.generation.prompts import citation

DISCLAIMER = "Atenție: acesta nu este un sfat juridic și nu înlocuiește un avocat."


def print_answer(answer: Answer, show_source_text: bool) -> None:
    print("\n" + answer.text + "\n")
    for lege, versiune in answer.versions.items():
        print(f"Versiune: {lege}, forma consolidată la {versiune}")

    print("\nFragmente găsite (scor):")
    for chunk in answer.sources:
        print(f"  {chunk.score:.3f}  {citation(chunk)}")
        if show_source_text:
            print("        " + chunk.text[:500].replace("\n", "\n        ") + "\n")
    print(f"\n{DISCLAIMER}")


def analyze_fine_file(path: Path, show_source_text: bool) -> None:
    """Print the facts extracted from a fine PDF and the explanation. The file is only read, never copied."""
    text, method = extract_text(path.read_bytes())
    analysis = analyze_fine(text, method)
    facts = analysis.facts
    print(f"Text citit prin: {method}")
    print("Articole încălcate: " + (", ".join(describe(ref) for ref in facts.articles) or "-"))
    print(
        f"Amendă: {facts.amount_lei or '-'} lei | Puncte: {facts.penalty_points or '-'} | Data: {facts.date or '-'}"
    )
    if analysis.answer is None:
        print("\nNu am găsit în document niciun articol din OUG 195/2002 sau HG 1391/2006.")
        return
    print_answer(analysis.answer, show_source_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question", nargs="*", help="the question (omit for interactive mode)")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help="how many chunks to retrieve")
    parser.add_argument("--sources", action="store_true", help="also print the text of retrieved chunks")
    parser.add_argument("--fine", type=Path, help="analyze a fine (proces-verbal) PDF instead of a question")
    args = parser.parse_args()

    if args.fine:
        analyze_fine_file(args.fine, args.sources)
        return
    if args.question:
        print_answer(ask(" ".join(args.question), k=args.k), args.sources)
        return

    history: list[dict] = []  # the conversation so far, so follow-up questions work
    while question := input("\nÎntrebare: ").strip():
        answer = ask(question, history=history, k=args.k)
        if answer.standalone_question != question:
            print(f"(căutat ca: {answer.standalone_question})")
        print_answer(answer, args.sources)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer.text}]


if __name__ == "__main__":
    main()
