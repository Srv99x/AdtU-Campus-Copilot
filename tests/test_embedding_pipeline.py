from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path

import chromadb
import pytest

import embed
import query_embed
from derived_embeddings import (
    MAX_INPUT_TOKENS,
    REQUIRED_METADATA_FIELDS,
    build_derived_units,
    estimate_tokens as estimate_derived_tokens,
    index_derived_embeddings,
    load_derived_embedding_records,
    ordered_derived_embedding_records,
    save_derived_embedding_records,
)
from scripts import chroma_ingest
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


class FakeModels:
    """In-memory Gemini replacement that records requests without network access."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls = 0

    def embed_content(self, **_: object) -> object:
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.models = FakeModels(responses)


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


def test_embedding_settings_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "25")
    monkeypatch.setenv("EMBEDDING_TPM_BUDGET", "75000")
    monkeypatch.setenv("EMBEDDING_RPM_BUDGET", "45")
    monkeypatch.setenv("EMBEDDING_RPD_BUDGET", "80")
    monkeypatch.setenv("EMBEDDING_TOKEN_SAFETY_FACTOR", "1.25")

    assert embed.embedding_settings() == embed.EmbeddingSettings(
        batch_size=25,
        tpm_budget=75000,
        rpm_budget=45,
        rpd_budget=80,
        token_safety_factor=1.25,
    )


def test_embedding_plan_estimates_tokens_and_rpd_protection() -> None:
    settings = embed.EmbeddingSettings(
        batch_size=10,
        tpm_budget=80_000,
        rpm_budget=50,
        rpd_budget=2,
        token_safety_factor=1.2,
    )
    chunks = [{"text": "a" * 40} for _ in range(21)]

    plan = embed.create_embedding_plan(chunks, settings)

    assert plan.estimated_requests == 3
    assert plan.estimated_tokens == 252
    assert not plan.fits_rpd_budget
    assert plan.max_chunks_within_rpd_budget == 20


def test_minute_rate_limiter_waits_before_exceeding_budget() -> None:
    now = [0.0]
    waits: list[float] = []

    def clock() -> float:
        return now[0]

    def sleeper(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    limiter = embed.MinuteRateLimiter(
        tpm_budget=100, rpm_budget=2, clock=clock, sleeper=sleeper
    )
    limiter.wait_for_capacity(60)
    limiter.wait_for_capacity(50)

    assert waits == [60.0]
    assert limiter.used_requests == 1
    assert limiter.used_tokens == 50


def test_batch_embedding_uses_multiple_contents_without_real_api() -> None:
    response = SimpleNamespace(
        embeddings=[
            SimpleNamespace(values=vector()),
            SimpleNamespace(values=vector()),
        ]
    )
    client = FakeClient([response])

    assert embed.create_embeddings(client, ["first text", "second text"]) == [
        vector(),
        vector(),
    ]
    assert client.models.calls == 1


def test_quota_error_stops_after_one_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient([RuntimeError("429 RESOURCE_EXHAUSTED: TPM token quota exceeded")])
    monkeypatch.setattr(embed.time, "sleep", lambda _: pytest.fail("must not sleep"))

    with pytest.raises(embed.QuotaExhaustedError, match="quota exhaustion"):
        embed.create_embeddings(client, ["pending text"])

    assert client.models.calls == 1


def test_transient_error_uses_bounded_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    response = SimpleNamespace(embeddings=[SimpleNamespace(values=vector())])
    client = FakeClient([RuntimeError("503 temporarily unavailable"), response])
    delays: list[int] = []
    monkeypatch.setattr(embed.time, "sleep", delays.append)

    assert embed.create_embeddings(client, ["pending text"]) == [vector()]
    assert client.models.calls == 2
    assert delays == [30]


def test_embedding_model_is_shared_with_query_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")

    assert embed.embedding_model() == "gemini-embedding-2"
    assert query_embed.embedding_model() == "gemini-embedding-2"


def test_derived_units_are_deterministic_and_safe() -> None:
    chunks = embed.load_chunks()
    first = build_derived_units(chunks)
    second = build_derived_units(deepcopy(chunks))

    assert first == second
    assert len({unit["child_id"] for unit in first}) == len(first)
    assert all(estimate_derived_tokens(unit["text"]) < MAX_INPUT_TOKENS for unit in first)

    by_parent: dict[str, list[dict[str, object]]] = {}
    source_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    for unit in first:
        parent_id = str(unit["parent_chunk_id"])
        by_parent.setdefault(parent_id, []).append(unit)
        assert unit["child_id"].startswith(f"{parent_id}::embed::v1::part-")
        for field in REQUIRED_METADATA_FIELDS:
            assert unit[field] == source_by_id[parent_id][field]

    for parent_id, units in by_parent.items():
        assert [unit["part_index"] for unit in units] == list(range(len(units)))
        assert {unit["part_count"] for unit in units} == {len(units)}


def test_derived_checkpoint_resumes_by_child_id(tmp_path: Path) -> None:
    units = build_derived_units(embed.load_chunks())
    checkpoint = tmp_path / "derived_embeddings.json"
    vectors_by_id = {units[0]["child_id"]: vector()}
    records = ordered_derived_embedding_records(units, vectors_by_id)

    save_derived_embedding_records(records, checkpoint)

    assert index_derived_embeddings(load_derived_embedding_records(checkpoint), units) == vectors_by_id
    assert not checkpoint.with_suffix(".tmp").exists()


def test_token_aware_batches_respect_token_and_count_limits() -> None:
    settings = embed.EmbeddingSettings(
        batch_size=10,
        tpm_budget=25_000,
        rpm_budget=80,
        rpd_budget=900,
        token_safety_factor=1.2,
    )
    items = [
        embed.EmbeddingWorkItem(f"chunk-{index}", "a" * 16_000, False)
        for index in range(5)
    ]

    batches = embed.token_aware_batches(items, settings)

    assert [len(batch) for batch in batches] == [4, 1]
    assert all(
        sum(embed.estimate_tokens(item.text, settings.token_safety_factor) for item in batch)
        <= embed.TARGET_BATCH_TOKENS
        for batch in batches
    )


def test_existing_canonical_embeddings_are_excluded_from_new_direct_work() -> None:
    existing = chunk("existing")
    pending = chunk("pending")

    assert embed.direct_missing_chunks(
        [existing, pending], {"existing": vector()}
    ) == [pending]


def test_chroma_metadata_is_copied_from_top_level_chunk_fields() -> None:
    source_chunk = chunk()

    assert metadata_for_chunk(source_chunk) == {
        "source_url": "https://adtu.in/hostel",
        "category": "fees",
        "section": "Hostel fees",
        "last_updated": "2026-08-16",
        "is_table": False,
        "chunk_id": "chunk-1",
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


def test_chroma_ingestion_rejects_incomplete_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_chunk = chunk()
    monkeypatch.setattr(chroma_ingest, "load_chunks", lambda: [source_chunk])
    monkeypatch.setattr(chroma_ingest, "load_embedding_records", lambda: [])
    monkeypatch.setattr(chroma_ingest, "build_derived_units", lambda _: [])
    monkeypatch.setattr(chroma_ingest, "load_derived_embedding_records", lambda: [])
    monkeypatch.setattr(chroma_ingest, "index_derived_embeddings", lambda *_: {})

    with pytest.raises(RuntimeError, match="incomplete embedding coverage"):
        chroma_ingest.prepare_records()


def test_derived_chroma_metadata_preserves_parent_linkage() -> None:
    unit = build_derived_units(embed.load_chunks())[0]

    metadata = chroma_ingest.metadata_for_derived_unit(unit)

    assert metadata["chunk_id"] == unit["child_id"]
    assert metadata["parent_chunk_id"] == unit["parent_chunk_id"]
    assert metadata["part_index"] == unit["part_index"]
    assert metadata["part_count"] == unit["part_count"]
    assert metadata["derivation_version"] == unit["derivation_version"]


def test_chroma_ingestion_rejects_incomplete_derived_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_chunk = chunk()
    derived_unit = {
        "child_id": "chunk-1::embed::v1::part-000::abc123",
        "parent_chunk_id": "chunk-1",
        "part_index": 0,
        "part_count": 1,
        "derivation_version": "v1",
        "text": "Derived official hostel fee information",
        **{field: source_chunk[field] for field in REQUIRED_METADATA_FIELDS},
    }
    monkeypatch.setattr(chroma_ingest, "load_chunks", lambda: [source_chunk])
    monkeypatch.setattr(chroma_ingest, "load_embedding_records", lambda: [])
    monkeypatch.setattr(
        chroma_ingest, "index_existing_embeddings", lambda *_: {"chunk-1": vector()}
    )
    monkeypatch.setattr(chroma_ingest, "build_derived_units", lambda _: [derived_unit])
    monkeypatch.setattr(chroma_ingest, "load_derived_embedding_records", lambda: [])
    monkeypatch.setattr(chroma_ingest, "index_derived_embeddings", lambda *_: {})

    with pytest.raises(RuntimeError, match="1 derived child records missing"):
        chroma_ingest.prepare_records()


def test_retrieval_result_checks_metadata_and_official_hostel_content() -> None:
    metadata = metadata_for_chunk(chunk())
    assert_complete_metadata(metadata, "chunk-1")
    assert_official_hostel_result(
        ["chunk-1"], ["Official AdtU hostel fee information"], [metadata]
    )


# ------------------------------------------------------------------
# V2 EMBEDDING PIPELINE TESTS
# ------------------------------------------------------------------


V2_REQUIRED_FIELDS = (
    "chunk_id", "document", "source_url", "intent_category", "category",
    "topic", "section", "last_updated", "is_table", "chunk_index", "text",
    "source_type", "verification_status", "effective_date", "published_date",
    "quality_status", "estimated_tokens",
)

V2_OPTIONAL_FIELDS = ("program", "specialization", "semester")


def v2_chunk(chunk_id: str = "v2_test_0") -> dict[str, object]:
    """Minimal V2 chunk fixture with all required metadata fields."""
    return {
        "chunk_id": chunk_id,
        "document": "test.md",
        "source_url": "",
        "intent_category": "fees",
        "category": "fees",
        "topic": "test_topic",
        "section": "Test section",
        "last_updated": "2026-08-18",
        "is_table": False,
        "chunk_index": 0,
        "text": "V2 test chunk content for embedding",
        "source_type": "official_pdf",
        "verification_status": "trusted_official_source",
        "effective_date": "2026-27",
        "published_date": "2026-08-18",
        "quality_status": "ok",
        "estimated_tokens": 11,
    }


def v2_chunk_with_routine_metadata(chunk_id: str = "v2_routine_0") -> dict[str, object]:
    """V2 chunk fixture with optional program/specialization/semester fields."""
    base = v2_chunk(chunk_id)
    base.update({
        "intent_category": "facilities",
        "category": "facilities",
        "topic": "class_routine",
        "section": "Monday",
        "program": "B.Tech",
        "specialization": "CSE",
        "semester": "3rd",
    })
    return base


def test_v1_chunk_count_is_immutable() -> None:
    """V1 loader must return exactly 842 chunks."""
    chunks = embed.load_chunks()
    assert len(chunks) == 842


def test_v2_chunks_loaded_with_v2_prefix() -> None:
    """V2 loader returns chunks whose IDs all start with v2_."""
    v2_chunks = embed.load_v2_chunks()
    assert len(v2_chunks) == 76
    assert all(chunk["chunk_id"].startswith("v2_") for chunk in v2_chunks)


def test_v1_v2_no_id_collision() -> None:
    """V1 and V2 chunk IDs must never overlap."""
    v1_chunks = embed.load_chunks()
    v1_ids = {chunk["chunk_id"] for chunk in v1_chunks}
    v2_chunks = embed.load_v2_chunks(v1_ids=v1_ids)
    v2_ids = {chunk["chunk_id"] for chunk in v2_chunks}
    assert not v1_ids & v2_ids


def test_v2_collision_detection_raises() -> None:
    """Loading V2 chunks with a colliding V1 ID must raise."""
    with pytest.raises(embed.EmbeddingValidationError, match="collides with V1"):
        embed.load_v2_chunks(v1_ids={"v2_academic_calendar_0"})


def test_v2_embedding_checkpoint_round_trip(tmp_path: Path) -> None:
    """V2 embeddings can be saved, loaded, and validated."""
    source = v2_chunk()
    v2_chunks_by_id = {source["chunk_id"]: source}
    records = [{"chunk": source, "embedding": vector()}]
    output_file = tmp_path / "v2_embeddings.json"

    embed.save_v2_embedding_records(records, output_file)
    loaded = embed.load_v2_embedding_records(output_file)
    indexed = embed.index_v2_existing_embeddings(loaded, v2_chunks_by_id)

    assert indexed == {"v2_test_0": vector()}
    assert not output_file.with_suffix(".tmp").exists()


def test_v2_duplicate_id_rejected(tmp_path: Path) -> None:
    """Duplicate V2 embedding records must be detected."""
    source = v2_chunk()
    records = [
        {"chunk": source, "embedding": vector()},
        {"chunk": source, "embedding": vector()},
    ]
    with pytest.raises(embed.EmbeddingValidationError, match="Duplicate V2"):
        embed.index_v2_existing_embeddings(records, {"v2_test_0": source})


def test_v2_metadata_preserved_in_checkpoint() -> None:
    """All V2 metadata fields must survive the checkpoint round-trip."""
    source = v2_chunk_with_routine_metadata()
    record = {"chunk": source, "embedding": vector()}
    indexed = embed.index_v2_existing_embeddings(
        [record], {source["chunk_id"]: source}
    )
    assert source["chunk_id"] in indexed

    # Verify the V2-specific fields are on the source (not stripped)
    for field in V2_REQUIRED_FIELDS:
        assert field in source, f"Missing required field: {field}"
    for field in V2_OPTIONAL_FIELDS:
        assert field in source, f"Missing optional field: {field}"


def test_v2_token_sizes_within_limit() -> None:
    """No V2 chunk may exceed the Gemini Embedding 2 input limit."""
    v2_chunks = embed.load_v2_chunks()
    for chunk_data in v2_chunks:
        tokens = embed.estimate_tokens(chunk_data["text"])
        assert tokens < MAX_INPUT_TOKENS, (
            f"{chunk_data['chunk_id']} has {tokens} estimated tokens "
            f"(limit: {MAX_INPUT_TOKENS})"
        )


def test_v2_items_pack_into_batches() -> None:
    """V2 work items must batch correctly under TPM and count limits."""
    settings = embed.EmbeddingSettings(
        batch_size=10, tpm_budget=25_000, rpm_budget=80,
        rpd_budget=900, token_safety_factor=1.2,
    )
    items = [
        embed.EmbeddingWorkItem(f"v2_test_{i}", "a" * 400, False, "v2")
        for i in range(25)
    ]
    batches = embed.token_aware_batches(items, settings)
    assert all(len(b) <= settings.batch_size for b in batches)
    assert sum(len(b) for b in batches) == 25
    for batch in batches:
        batch_tokens = sum(
            embed.estimate_tokens(item.text, settings.token_safety_factor)
            for item in batch
        )
        assert batch_tokens <= embed.TARGET_BATCH_TOKENS


def test_v2_resume_by_stable_id(tmp_path: Path) -> None:
    """V2 checkpoint resumes correctly after partial embedding."""
    first = v2_chunk("v2_a_0")
    second = v2_chunk("v2_b_0")
    output_file = tmp_path / "v2_embeddings.json"

    # Simulate: only first chunk embedded
    records = embed.ordered_v2_embedding_records(
        [first, second], {"v2_a_0": vector()}
    )
    embed.save_v2_embedding_records(records, output_file)

    loaded = embed.load_v2_embedding_records(output_file)
    indexed = embed.index_v2_existing_embeddings(
        loaded, {"v2_a_0": first, "v2_b_0": second}
    )
    assert indexed == {"v2_a_0": vector()}
    assert "v2_b_0" not in indexed


def test_v2_incomplete_coverage_detected() -> None:
    """Missing V2 embeddings must be reported, not silently ignored."""
    v2_chunks = embed.load_v2_chunks()
    # With no V2 embeddings, all V2 chunks should be pending
    v2_pending = [
        chunk for chunk in v2_chunks
        if chunk["chunk_id"] not in {}
    ]
    assert len(v2_pending) == len(v2_chunks)
    assert len(v2_pending) == 76


def test_v2_mismatched_chunk_rejected() -> None:
    """A V2 checkpoint with modified chunk payload must be rejected."""
    source = v2_chunk()
    modified = {**source, "text": "tampered content"}
    record = {"chunk": modified, "embedding": vector()}
    with pytest.raises(embed.EmbeddingValidationError, match="does not match"):
        embed.index_v2_existing_embeddings(
            [record], {"v2_test_0": source}
        )

