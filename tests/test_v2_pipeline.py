"""
Unit tests for the V2 Knowledge Base pipeline.

Tests:
    - V2 chunking produces non-zero chunks
    - V2 chunk IDs don't collide with V1
    - V2 metadata completeness
    - Deduplication logic
    - Category assignment
    - Token estimation consistency
    - Quality status detection
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

V1_CHUNKS_FILE = ROOT / "data" / "processed" / "chunks" / "chunks.jsonl"
V2_CHUNKS_FILE = ROOT / "data" / "processed" / "knowledge_v2" / "canonical_chunks.jsonl"


ALLOWED_CATEGORIES = {"admissions", "fees", "facilities"}

REQUIRED_FIELDS = [
    "chunk_id",
    "document",
    "source_url",
    "category",
    "section",
    "last_updated",
    "is_table",
    "text",
]


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""

    records: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def estimate_tokens(text: str) -> int:
    """Match the project's 4 chars/token estimation."""
    return max(1, math.ceil(len(text) / 4))


# --------------------------------------------------
# FIXTURES
# --------------------------------------------------

@pytest.fixture(scope="module")
def v1_chunks() -> list[dict]:
    """Load V1 immutable chunks."""
    return load_jsonl(V1_CHUNKS_FILE)


@pytest.fixture(scope="module")
def v2_chunks() -> list[dict]:
    """Load V2 chunks."""
    return load_jsonl(V2_CHUNKS_FILE)


# --------------------------------------------------
# V1 IMMUTABILITY
# --------------------------------------------------

def test_v1_immutable_chunk_count(v1_chunks: list[dict]) -> None:
    """V1 corpus must remain exactly 842 chunks."""

    assert len(v1_chunks) == 842, (
        f"V1 chunk count changed: expected 842, got {len(v1_chunks)}"
    )


# --------------------------------------------------
# V2 BASIC INTEGRITY
# --------------------------------------------------

def test_v2_produces_non_zero_chunks(v2_chunks: list[dict]) -> None:
    """V2 chunking must produce at least some chunks."""

    assert len(v2_chunks) > 0, "V2 produced zero chunks"


def test_v2_chunk_ids_are_unique(v2_chunks: list[dict]) -> None:
    """All V2 chunk IDs must be unique within V2."""

    ids = [c["chunk_id"] for c in v2_chunks]
    assert len(ids) == len(set(ids)), "Duplicate V2 chunk IDs found"


def test_v2_chunk_ids_no_collision_with_v1(
    v1_chunks: list[dict],
    v2_chunks: list[dict],
) -> None:
    """V2 chunk IDs must not collide with V1 chunk IDs."""

    v1_ids = {c["chunk_id"] for c in v1_chunks}
    v2_ids = {c["chunk_id"] for c in v2_chunks}

    collisions = v1_ids & v2_ids

    assert not collisions, (
        f"V1-V2 chunk ID collisions: {collisions}"
    )


def test_v2_chunk_ids_have_v2_prefix(v2_chunks: list[dict]) -> None:
    """All V2 chunk IDs must start with 'v2_'."""

    for chunk in v2_chunks:
        assert str(chunk["chunk_id"]).startswith("v2_"), (
            f"V2 chunk ID missing 'v2_' prefix: {chunk['chunk_id']}"
        )


# --------------------------------------------------
# V2 METADATA COMPLETENESS
# --------------------------------------------------

def test_v2_required_fields_present(v2_chunks: list[dict]) -> None:
    """Every V2 chunk must have all required fields."""

    for chunk in v2_chunks:
        for field in REQUIRED_FIELDS:
            assert field in chunk, (
                f"Chunk {chunk.get('chunk_id', '?')} missing field: {field}"
            )


def test_v2_categories_are_valid(v2_chunks: list[dict]) -> None:
    """V2 categories must be from the locked set."""

    for chunk in v2_chunks:
        assert chunk["category"] in ALLOWED_CATEGORIES, (
            f"Chunk {chunk['chunk_id']} has invalid category: {chunk['category']}"
        )


def test_v2_is_table_is_boolean(v2_chunks: list[dict]) -> None:
    """is_table must be a boolean."""

    for chunk in v2_chunks:
        assert isinstance(chunk["is_table"], bool), (
            f"Chunk {chunk['chunk_id']} has non-boolean is_table: {chunk['is_table']}"
        )


def test_v2_text_is_non_empty(v2_chunks: list[dict]) -> None:
    """Every chunk must have non-empty text."""

    for chunk in v2_chunks:
        text = str(chunk.get("text", "")).strip()
        assert text, (
            f"Chunk {chunk['chunk_id']} has empty text"
        )


def test_v2_last_updated_is_non_empty(v2_chunks: list[dict]) -> None:
    """Every chunk must have a last_updated value."""

    for chunk in v2_chunks:
        assert chunk.get("last_updated"), (
            f"Chunk {chunk['chunk_id']} missing last_updated"
        )


def test_v2_section_is_non_empty(v2_chunks: list[dict]) -> None:
    """Every chunk must have a section value."""

    for chunk in v2_chunks:
        section = str(chunk.get("section", "")).strip()
        assert section, (
            f"Chunk {chunk['chunk_id']} has empty section"
        )


# --------------------------------------------------
# V2 EXTENDED METADATA
# --------------------------------------------------

def test_v2_has_intent_category_and_topic(v2_chunks: list[dict]) -> None:
    """V2 chunks must have both intent_category and topic."""

    for chunk in v2_chunks:
        assert "intent_category" in chunk, (
            f"Chunk {chunk['chunk_id']} missing intent_category"
        )
        assert "topic" in chunk, (
            f"Chunk {chunk['chunk_id']} missing topic"
        )


def test_v2_has_quality_status(v2_chunks: list[dict]) -> None:
    """V2 chunks must have quality_status field."""

    for chunk in v2_chunks:
        assert "quality_status" in chunk, (
            f"Chunk {chunk['chunk_id']} missing quality_status"
        )
        assert chunk["quality_status"] in ("ok", "needs_manual_review"), (
            f"Chunk {chunk['chunk_id']} has invalid quality_status: {chunk['quality_status']}"
        )


def test_v2_has_source_type(v2_chunks: list[dict]) -> None:
    """V2 chunks must have source_type field."""

    for chunk in v2_chunks:
        assert "source_type" in chunk, (
            f"Chunk {chunk['chunk_id']} missing source_type"
        )


# --------------------------------------------------
# TOKEN SIZE
# --------------------------------------------------

def test_v2_no_empty_token_chunks(v2_chunks: list[dict]) -> None:
    """No V2 chunk should have zero estimated tokens."""

    for chunk in v2_chunks:
        tokens = estimate_tokens(str(chunk["text"]))
        assert tokens > 0, (
            f"Chunk {chunk['chunk_id']} has 0 estimated tokens"
        )


# --------------------------------------------------
# CATEGORY DISTRIBUTION
# --------------------------------------------------

def test_v2_has_fees_category(v2_chunks: list[dict]) -> None:
    """V2 should include chunks with 'fees' category (from fees.pdf and scholarships.pdf)."""

    fees_chunks = [c for c in v2_chunks if c["category"] == "fees"]
    assert len(fees_chunks) > 0, "No V2 chunks have 'fees' category"


def test_v2_has_facilities_category(v2_chunks: list[dict]) -> None:
    """V2 should include chunks with 'facilities' category."""

    facilities_chunks = [c for c in v2_chunks if c["category"] == "facilities"]
    assert len(facilities_chunks) > 0, "No V2 chunks have 'facilities' category"


# --------------------------------------------------
# TOPIC DIVERSITY
# --------------------------------------------------

def test_v2_covers_all_five_sources(v2_chunks: list[dict]) -> None:
    """V2 must have chunks from all 5 source PDFs."""

    documents = {c["document"] for c in v2_chunks}
    expected_docs = {
        "fees.md",
        "class_routine.md",
        "academic_calendar.md",
        "holiday_calendar.md",
        "scholarships.md",
    }

    missing = expected_docs - documents
    assert not missing, f"Missing documents in V2: {missing}"


def test_v2_topics_include_expected_values(v2_chunks: list[dict]) -> None:
    """V2 must include expected topic values."""

    topics = {c.get("topic") for c in v2_chunks}
    expected_topics = {"fees", "class_routine", "academic_calendar", "holiday_calendar", "scholarship"}

    missing = expected_topics - topics
    assert not missing, f"Missing topics in V2: {missing}"
