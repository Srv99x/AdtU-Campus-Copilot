"""Verify cosine retrieval against the complete AdtU Chroma collection."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import chromadb


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from query_embed import create_query_embedding  # noqa: E402
from derived_embeddings import OVERSIZED_PARENT_IDS, build_derived_units  # noqa: E402
from embed import load_chunks, load_v2_chunks  # noqa: E402
from scripts.chroma_ingest import V2_METADATA_FIELDS  # noqa: E402


CHROMA_DIR = ROOT / "data" / "processed" / "chroma_db"
COLLECTION_NAME = "adtu_knowledge"
REQUIRED_METADATA_FIELDS = (
    "chunk_id",
    "source_url",
    "category",
    "section",
    "last_updated",
    "is_table",
)


def expected_record_count() -> int:
    """Return the deterministic direct-canonical plus derived-child record count."""
    chunks = load_chunks()
    v1_count = len(chunks) - len(OVERSIZED_PARENT_IDS) + len(build_derived_units(chunks))
    v2_count = len(load_v2_chunks())
    return v1_count + v2_count


def assert_complete_metadata(metadata: dict[str, Any], chunk_id: str) -> None:
    """Require the immutable chunk metadata values in every result."""
    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            raise AssertionError(f"Result {chunk_id} is missing metadata: {field}")

        value = metadata[field]
        if field == "is_table":
            if not isinstance(value, bool):
                raise AssertionError(f"Result {chunk_id} has non-boolean is_table.")
        elif not isinstance(value, str) or not value.strip():
            if field == "source_url" and chunk_id.startswith("v2_") and isinstance(value, str):
                pass
            else:
                raise AssertionError(f"Result {chunk_id} has empty {field} metadata.")

    if chunk_id.startswith("v2_"):
        has_v2_metadata = False
        for field in V2_METADATA_FIELDS:
            if field in metadata:
                has_v2_metadata = True
                value = metadata[field]
                if not isinstance(value, (str, bool, int, float)):
                    raise AssertionError(f"Result {chunk_id} has invalid V2 metadata type for {field}.")
                if isinstance(value, str) and not value.strip():
                    raise AssertionError(f"Result {chunk_id} has empty string for {field}.")
        
        if not has_v2_metadata:
            raise AssertionError(f"Result {chunk_id} is missing all V2-specific metadata.")


def test_query(collection: chromadb.Collection, query_text: str, expected_chunk_id: str | None = None, expected_evidence: list[str] | None = None, negative_test: bool = False) -> None:
    """Run a query and assert expected evidence."""
    query_embedding = create_query_embedding(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=["documents", "metadatas", "distances"],
    )
    
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    if not ids:
        raise AssertionError(f"Chroma returned no retrieval results for '{query_text}'.")

    print(f"\nQuery: {query_text}")
    print(f"Top result ID: {ids[0]}")
    
    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances, strict=True), start=1
    ):
        assert_complete_metadata(metadata, chunk_id)
        
    if negative_test:
        for chunk_id, document in zip(ids, documents):
            if "hostel" in document.lower() and "fee" in document.lower():
                raise AssertionError(f"Negative test failed: found false positive hostel-fee in {chunk_id}")
        print(f"Negative check PASS: No false positive hostel-fee found.")
        return
        
    if not expected_chunk_id or not expected_evidence:
        raise ValueError("Must provide expected_chunk_id and expected_evidence for positive tests.")

    if expected_chunk_id not in ids:
        raise AssertionError(f"Expected chunk {expected_chunk_id} not found in top 10 results for '{query_text}'.")
        
    idx = ids.index(expected_chunk_id)
    doc_lower = documents[idx].lower()
    for ev in expected_evidence:
        if ev.lower() not in doc_lower:
            raise AssertionError(f"Missing evidence '{ev}' in chunk {expected_chunk_id} for '{query_text}'.")
            
    print(f"PASS: Found {expected_chunk_id} with expected evidence.")


def main() -> None:
    """Embed the required queries and display validated top Chroma results."""
    sys.stdout.reconfigure(encoding='utf-8')

    if not CHROMA_DIR.exists():
        raise RuntimeError("Chroma database does not exist. Build it only after all 957 embeddings exist.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    record_count = collection.count()
    expected_count = expected_record_count()
    if record_count != expected_count:
        raise RuntimeError(f"Retrieval test requires the complete collection: expected {expected_count}, found {record_count}.")

    print(f"Collection: {COLLECTION_NAME}")
    print("Distance: cosine")
    
    test_query(
        collection, 
        "What is the seat capacity of the boys H Block hostel?",
        expected_chunk_id="hostel-details-html_33",
        expected_evidence=["H Block", "152"]
    )
    
    test_query(
        collection, 
        "What is the total fee for BCA with IBM?",
        expected_chunk_id="v2_fees_6",
        expected_evidence=["Bachelor of Computer Application", "IBM", "520,000"]
    )
    
    test_query(
        collection, 
        "What scholarship is available for CBSE board students with 95%?",
        expected_chunk_id="v2_scholarships_0",
        expected_evidence=["95%", "100% SCHOLARSHIP"]
    )
    
    test_query(
        collection, 
        "What is the class routine for B.Tech CSE DS & AI IBM 2nd year 1st semester?",
        expected_chunk_id="v2_class_routine_14",
        expected_evidence=["Btech CSE [DS & AI]", "IBM", "2nd year 1st semester"]
    )
    
    test_query(
        collection, 
        "When are the holidays subject to change by Govt of Assam?",
        expected_chunk_id="v2_holiday_calendar_2",
        expected_evidence=["Dates of Holidays might change", "Govt. of Assam"]
    )
    
    test_query(
        collection, 
        "What is the hostel fee?",
        negative_test=True
    )
    
    print("\nAll retrieval smoke tests: PASS")


if __name__ == "__main__":
    main()
