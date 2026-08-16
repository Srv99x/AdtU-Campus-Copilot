from __future__ import annotations

from pathlib import Path

import chromadb
import pytest

import embed
from scripts.chroma_ingest import metadata_for_chunk
from scripts.chroma_retrieval_test import (
    assert_complete_metadata,
    assert_official_hostel_result,
)


def chunk(chunk_id: str = "chunk-1") -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "text": "Official AdtU hostel fee information",
        "source_url": "https://adtu.in/hostel",
        "category": "fees",
        "section": "Hostel fees",
        "last_updated": "2026-08-16",
        "is_table": False,
    }


def vector() -> list[float]:
    return [0.25] * embed.DIMENSION


def test_existing_embeddings_are_indexed_by_chunk_id() -> None:
    source_chunk = chunk()
    indexed = embed.index_existing_embeddings(
        [{"chunk": source_chunk, "embedding": vector()}],
        {"chunk-1": source_chunk},
    )

    assert indexed == {"chunk-1": vector()}


def test_duplicate_embedding_chunk_id_is_rejected() -> None:
    source_chunk = chunk()
    records = [
        {"chunk": source_chunk, "embedding": vector()},
        {"chunk": source_chunk, "embedding": vector()},
    ]

    with pytest.raises(embed.EmbeddingValidationError, match="Duplicate embedding"):
        embed.index_existing_embeddings(records, {"chunk-1": source_chunk})


def test_unknown_or_mismatched_saved_chunk_is_rejected() -> None:
    source_chunk = chunk()
    unknown_chunk = chunk("unknown")

    with pytest.raises(embed.EmbeddingValidationError, match="nonexistent"):
        embed.index_existing_embeddings(
            [{"chunk": unknown_chunk, "embedding": vector()}],
            {"chunk-1": source_chunk},
        )

    changed_chunk = {**source_chunk, "text": "Unexpected text"}
    with pytest.raises(embed.EmbeddingValidationError, match="does not match"):
        embed.index_existing_embeddings(
            [{"chunk": changed_chunk, "embedding": vector()}],
            {"chunk-1": source_chunk},
        )


def test_invalid_vector_is_rejected() -> None:
    source_chunk = chunk()
    with pytest.raises(embed.EmbeddingValidationError, match="Invalid 768-dimension"):
        embed.index_existing_embeddings(
            [{"chunk": source_chunk, "embedding": [0.0] * 767}],
            {"chunk-1": source_chunk},
        )


def test_checkpoint_write_is_atomic_and_ordered(tmp_path: Path) -> None:
    first = chunk("chunk-1")
    second = chunk("chunk-2")
    records = embed.ordered_embedding_records(
        [first, second], {"chunk-2": vector(), "chunk-1": vector()}
    )
    output_file = tmp_path / "embeddings.json"

    embed.save_embedding_records(records, output_file)

    assert [record["chunk"]["chunk_id"] for record in records] == [
        "chunk-1",
        "chunk-2",
    ]
    assert not output_file.with_suffix(".tmp").exists()
    assert embed.load_embedding_records(output_file) == records


def test_quota_detection_avoids_blind_retries() -> None:
    quota_error = RuntimeError("429 RESOURCE_EXHAUSTED: TPM token quota exceeded")
    temporary_rate_error = RuntimeError("429 RESOURCE_EXHAUSTED: retry in 12s")

    assert embed.is_daily_or_token_quota_error(quota_error)
    assert not embed.is_daily_or_token_quota_error(temporary_rate_error)
    assert embed.is_rate_limit_error(temporary_rate_error)
    assert embed.retry_delay_seconds(temporary_rate_error, 1) == 12


def test_chroma_metadata_is_copied_from_top_level_chunk_fields() -> None:
    source_chunk = chunk()

    assert metadata_for_chunk(source_chunk) == {
        "source_url": "https://adtu.in/hostel",
        "category": "fees",
        "section": "Hostel fees",
        "last_updated": "2026-08-16",
        "is_table": False,
    }

    missing_metadata = dict(source_chunk)
    del missing_metadata["section"]
    with pytest.raises(embed.EmbeddingValidationError, match="section"):
        metadata_for_chunk(missing_metadata)


def test_chroma_cosine_collection_accepts_validated_records() -> None:
    source_chunk = chunk()
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="adtu_knowledge",
        configuration={"hnsw": {"space": "cosine"}},
    )
    collection.upsert(
        ids=["chunk-1"],
        documents=[str(source_chunk["text"])],
        metadatas=[metadata_for_chunk(source_chunk)],
        embeddings=[vector()],
    )

    result = collection.query(
        query_embeddings=[vector()],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )

    assert result["ids"] == [["chunk-1"]]
    assert result["metadatas"] == [[metadata_for_chunk(source_chunk)]]
    assert result["distances"] == [[pytest.approx(0.0, abs=1e-4)]]


def test_retrieval_result_checks_metadata_and_official_hostel_content() -> None:
    metadata = metadata_for_chunk(chunk())
    assert_complete_metadata(metadata, "chunk-1")
    assert_official_hostel_result(
        ["chunk-1"], ["Official AdtU hostel fee information"], [metadata]
    )
