"""Verify cosine retrieval against the complete AdtU Chroma collection."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import chromadb


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from query_embed import create_query_embedding  # noqa: E402


CHROMA_DIR = ROOT / "data" / "processed" / "chroma_db"
COLLECTION_NAME = "adtu_knowledge"
EXPECTED_RECORD_COUNT = 842
QUERY = "What is the hostel fee?"
REQUIRED_METADATA_FIELDS = (
    "source_url",
    "category",
    "section",
    "last_updated",
    "is_table",
)


def assert_complete_metadata(metadata: dict[str, Any], chunk_id: str) -> None:
    """Require the five immutable chunk metadata values in every result."""

    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            raise AssertionError(f"Result {chunk_id} is missing metadata: {field}")

        value = metadata[field]
        if field == "is_table":
            if not isinstance(value, bool):
                raise AssertionError(f"Result {chunk_id} has non-boolean is_table.")
        elif not isinstance(value, str) or not value.strip():
            raise AssertionError(f"Result {chunk_id} has empty {field} metadata.")


def assert_official_hostel_result(
    ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]
) -> None:
    """Require at least one result to be official AdtU hostel-fee information."""

    for chunk_id, document, metadata in zip(ids, documents, metadatas, strict=True):
        if (
            "hostel" in document.lower()
            and "fee" in document.lower()
            and "adtu.in" in metadata["source_url"].lower()
        ):
            return

    raise AssertionError(
        "No returned result contains official AdtU hostel-fee information."
    )


def main() -> None:
    """Embed the required query and display validated top Chroma results."""

    if not CHROMA_DIR.exists():
        raise RuntimeError(
            "Chroma database does not exist. Build it only after all 842 embeddings exist."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    record_count = collection.count()
    if record_count != EXPECTED_RECORD_COUNT:
        raise RuntimeError(
            "Retrieval test requires the complete collection: "
            f"expected {EXPECTED_RECORD_COUNT}, found {record_count}."
        )

    query_embedding = create_query_embedding(QUERY)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    if not ids:
        raise AssertionError("Chroma returned no retrieval results.")

    if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
        raise AssertionError("Chroma returned misaligned retrieval result fields.")

    print(f"Collection: {COLLECTION_NAME}")
    print("Distance: cosine")
    print(f"Query: {QUERY}")

    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances, strict=True),
        start=1,
    ):
        if not isinstance(metadata, dict):
            raise AssertionError(f"Result {chunk_id} has invalid metadata.")
        assert_complete_metadata(metadata, chunk_id)

        print(f"\nResult {rank}")
        print(f"chunk_id: {chunk_id}")
        print(f"distance: {distance}")
        print(f"source_url: {metadata['source_url']}")
        print(f"category: {metadata['category']}")
        print(f"section: {metadata['section']}")
        print(f"last_updated: {metadata['last_updated']}")
        print(f"is_table: {metadata['is_table']}")
        print(f"chunk text:\n{document}")

    assert_official_hostel_result(ids, documents, metadatas)
    print("\nHostel-fee retrieval verification: PASS")


if __name__ == "__main__":
    main()
