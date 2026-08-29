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
    load_v2_chunks,
    load_v2_embedding_records,
    index_v2_existing_embeddings,
)
from derived_embeddings import (  # noqa: E402
    OVERSIZED_PARENT_IDS,
    build_derived_units,
    index_derived_embeddings,
    load_derived_embedding_records,
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

V2_METADATA_FIELDS = (
    "intent_category",
    "topic",
    "source_type",
    "verification_status",
    "effective_date",
    "published_date",
    "quality_status",
    "program",
    "specialization",
    "semester",
)


def metadata_for_chunk(chunk: dict[str, Any]) -> dict[str, str | bool]:
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
            if field == "source_url" and str(chunk_id).startswith("v2_") and isinstance(value, str):
                pass
            else:
                raise EmbeddingValidationError(
                    f"Chunk {chunk_id} has empty or invalid {field} metadata."
                )

        metadata[field] = value

    metadata["chunk_id"] = str(chunk_id)
    return metadata


def metadata_for_derived_unit(unit: dict[str, Any]) -> dict[str, str | bool | int]:
    """Return Chroma metadata that preserves child and canonical parent identity."""

    metadata = metadata_for_chunk({**unit, "chunk_id": unit["child_id"]})
    metadata.update(
        {
            "parent_chunk_id": str(unit["parent_chunk_id"]),
            "part_index": int(unit["part_index"]),
            "part_count": int(unit["part_count"]),
            "derivation_version": str(unit["derivation_version"]),
        }
    )
    return metadata


def metadata_for_v2_chunk(chunk: dict[str, Any]) -> dict[str, str | bool | int | float]:
    """Return Chroma metadata for V2 chunks, preserving optional fields."""

    metadata = metadata_for_chunk(chunk)
    for field in V2_METADATA_FIELDS:
        if field in chunk and chunk[field] is not None:
            value = chunk[field]
            if isinstance(value, (str, bool, int, float)):
                if isinstance(value, str) and not value.strip():
                    continue
                metadata[field] = value
    return metadata


def prepare_records() -> tuple[
    list[str], list[str], list[dict[str, str | bool | int]], list[list[float]]
]:
    """Validate complete direct/derived coverage and prepare Chroma records."""

    chunks = load_chunks()
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    saved_records = load_embedding_records()
    embeddings_by_id = index_existing_embeddings(saved_records, chunks_by_id)
    derived_units = build_derived_units(chunks)
    derived_records = load_derived_embedding_records()
    derived_embeddings_by_id = index_derived_embeddings(derived_records, derived_units)
    missing_ids = [
        chunk["chunk_id"]
        for chunk in chunks
        if chunk["chunk_id"] not in embeddings_by_id
        and chunk["chunk_id"] not in OVERSIZED_PARENT_IDS
    ]
    missing_derived_ids = [
        unit["child_id"]
        for unit in derived_units
        if unit["child_id"] not in derived_embeddings_by_id
    ]

    v1_ids = set(chunks_by_id)
    v2_chunks = load_v2_chunks(v1_ids=v1_ids)
    v2_chunks_by_id = {chunk["chunk_id"]: chunk for chunk in v2_chunks}
    v2_saved_records = load_v2_embedding_records()
    v2_embeddings_by_id = index_v2_existing_embeddings(
        v2_saved_records, v2_chunks_by_id
    )

    missing_v2_ids = [
        chunk["chunk_id"] for chunk in v2_chunks
        if chunk["chunk_id"] not in v2_embeddings_by_id
    ]

    if missing_ids or missing_derived_ids or missing_v2_ids:
        raise RuntimeError(
            "Refusing to build adtu_knowledge with incomplete embedding coverage: "
            f"{len(missing_ids)} direct canonical, "
            f"{len(missing_derived_ids)} derived child, and "
            f"{len(missing_v2_ids)} V2 records missing."
        )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | bool | int]] = []
    vectors: list[list[float]] = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        if chunk_id in OVERSIZED_PARENT_IDS:
            continue
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingValidationError(f"Chunk {chunk_id} has empty text.")

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(metadata_for_chunk(chunk))
        vectors.append(embeddings_by_id[chunk_id])

    for unit in derived_units:
        child_id = unit["child_id"]
        ids.append(child_id)
        documents.append(unit["text"])
        metadatas.append(metadata_for_derived_unit(unit))
        vectors.append(derived_embeddings_by_id[child_id])

    for chunk in v2_chunks:
        chunk_id = chunk["chunk_id"]
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingValidationError(f"Chunk {chunk_id} has empty text.")

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(metadata_for_v2_chunk(chunk))
        vectors.append(v2_embeddings_by_id[chunk_id])

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
