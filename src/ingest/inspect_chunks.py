"""Print chunks from data/processed/chunks.jsonl to check them by eye.

Run:
    python -m src.ingest.inspect_chunks              # 3 random chunks
    python -m src.ingest.inspect_chunks 10           # 10 random chunks
    python -m src.ingest.inspect_chunks oug195_art101_alin1   # one chunk by ID
"""

import json
import random
import sys

from src.config import settings


def main() -> None:
    path = settings.processed_dir / "chunks.jsonl"
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    arg = sys.argv[1] if len(sys.argv) > 1 else "3"
    if arg.isdigit():
        selected = random.sample(chunks, int(arg))
    else:
        selected = [chunk for chunk in chunks if chunk["chunk_id"] == arg]
        if not selected:
            print(f"No chunk with ID {arg}")

    for chunk in selected:
        metadata = {key: value for key, value in chunk.items() if key != "text"}
        print("=" * 80)
        print(metadata)
        print("-" * 80)
        print(chunk["text"])


if __name__ == "__main__":
    main()
