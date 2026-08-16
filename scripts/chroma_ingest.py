"""Build the complete AdtU Chroma collection from validated Gemini embeddings."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import chromadb


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embed import (  # noqa: E402
    EmbeddingValidationError,
    index_existing_embeddings,
    load_chunks,
    load_embedding_records,
)


CHROMA_DIR = ROOT / "data" / "processed" / "chroma_db"
COLLECTION_NAME = "adtu_knowledge"
REQUIRED_METADATA_FIELDS = (
    "source_url",
    "category",
    "section",
    "last_updated",
    "is_table",
)


def metadata_for_chunk(chunk: dict[str, Any]) -> dict[str, str | bool]:
    """Return validated top-level chunk metadata without fabricating defaults."""

    chunk_id = chunk.get("chunk_id", "<missing chunk_id>")
    metadata: dict[str, str | bool] = {}

    for field in REQUIRED_METADATA_FIELDS:
        if field not in chunk:
            raise EmbeddingValidationError(
                f"Chunk {chunk_id} is missing required metadata field: {field}"
            )

        value = chunk[field]
        if field == "is_table":
            if not isinstance(value, bool):
                raise EmbeddingValidationError(
                    f"Chunk {chunk_id} has non-boolean is_table metadata."
                )
        elif not isinstance(value, str) or not value.strip():
            raise EmbeddingValidationError(
                f"Chunk {chunk_id} has empty or invalid {field} metadata."
            )

        metadata[field] = value

    return metadata


def prepare_records() -> tuple[
    list[str], list[str], list[dict[str, str | bool]], list[list[float]]
]:
    """Validate the complete corpus and prepare deterministic Chroma records."""

    chunks = load_chunks()
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    saved_records = load_embedding_records()
    embeddings_by_id = index_existing_embeddings(saved_records, chunks_by_id)
    missing_ids = [chunk["chunk_id"] for chunk in chunks if chunk["chunk_id"] not in embeddings_by_id]

    if missing_ids:
        raise RuntimeError(
            "Refusing to build adtu_knowledge with incomplete embeddings: "
            f"{len(embeddings_by_id)}/{len(chunks)} present; "
            f"{len(missing_ids)} missing."
        )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | bool]] = []
    vectors: list[list[float]] = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingValidationError(f"Chunk {chunk_id} has empty text.")

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(metadata_for_chunk(chunk))
        vectors.append(embeddings_by_id[chunk_id])

    return ids, documents, metadatas, vectors


def main() -> None:
    """Ingest the complete validated dataset into cosine-distance ChromaDB."""

    print("ADT U CHROMADB INGESTION")
    ids, documents, metadatas, vectors = prepare_records()
    print(f"Validated complete corpus: {len(ids)} records")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=vectors,
    )

    count = collection.count()
    if count != len(ids):
        raise RuntimeError(
            f"Chroma collection count mismatch: expected {len(ids)}, found {count}."
        )

    print(f"Collection: {COLLECTION_NAME}")
    print("Distance: cosine")
    print(f"Collection record count: {count}")


if __name__ == "__main__":
    main()
