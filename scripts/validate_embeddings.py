"""Report whether saved Gemini embeddings safely align with the chunk source."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embed import (  # noqa: E402
    DIMENSION,
    EmbeddingValidationError,
    index_existing_embeddings,
    load_chunks,
    load_embedding_records,
)


def main() -> None:
    """Print an embedding alignment report and exit nonzero for invalid progress."""

    try:
        chunks = load_chunks()
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        saved_records = load_embedding_records()
        embeddings_by_id = index_existing_embeddings(saved_records, chunks_by_id)
    except EmbeddingValidationError as exc:
        print("EMBEDDING VALIDATION RESULT: FAIL")
        print(f"Reason: {exc}")
        raise SystemExit(1) from exc

    missing = len(chunks) - len(embeddings_by_id)
    print("ADT U EMBEDDING VALIDATION")
    print(f"Total chunks: {len(chunks)}")
    print(f"Existing embeddings: {len(saved_records)}")
    print(f"Unique embedded chunk IDs: {len(embeddings_by_id)}")
    print(f"Missing embeddings: {missing}")
    print("Invalid embeddings: 0")
    print(f"Dimension: {DIMENSION}")
    print("EMBEDDING VALIDATION RESULT: PASS")


if __name__ == "__main__":
    main()
