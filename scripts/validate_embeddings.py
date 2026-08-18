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
    load_v2_chunks,
    load_v2_embedding_records,
    index_v2_existing_embeddings,
)
from derived_embeddings import (  # noqa: E402
    DerivedEmbeddingValidationError,
    OVERSIZED_PARENT_IDS,
    build_derived_units,
    index_derived_embeddings,
    load_derived_embedding_records,
)


def main() -> None:
    """Print an embedding alignment report and exit nonzero for invalid progress."""

    try:
        chunks = load_chunks()
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        saved_records = load_embedding_records()
        embeddings_by_id = index_existing_embeddings(saved_records, chunks_by_id)
        derived_units = build_derived_units(chunks)
        saved_derived_records = load_derived_embedding_records()
        derived_vectors_by_id = index_derived_embeddings(
            saved_derived_records, derived_units
        )

        # V2
        v1_ids = set(chunks_by_id)
        v2_chunks = load_v2_chunks(v1_ids=v1_ids)
        v2_chunks_by_id = {chunk["chunk_id"]: chunk for chunk in v2_chunks}
        v2_saved_records = load_v2_embedding_records()
        v2_embeddings_by_id = index_v2_existing_embeddings(
            v2_saved_records, v2_chunks_by_id
        )
    except (EmbeddingValidationError, DerivedEmbeddingValidationError) as exc:
        print("EMBEDDING VALIDATION RESULT: FAIL")
        print(f"Reason: {exc}")
        raise SystemExit(1) from exc

    direct_missing = [
        chunk["chunk_id"]
        for chunk in chunks
        if chunk["chunk_id"] not in embeddings_by_id
        and chunk["chunk_id"] not in OVERSIZED_PARENT_IDS
    ]
    complete_derived_parents = {
        parent_id
        for parent_id in OVERSIZED_PARENT_IDS
        if all(
            unit["child_id"] in derived_vectors_by_id
            for unit in derived_units
            if unit["parent_chunk_id"] == parent_id
        )
    }
    canonical_coverage = len(embeddings_by_id) + len(complete_derived_parents)
    missing = len(chunks) - canonical_coverage

    v2_pending = [
        chunk["chunk_id"] for chunk in v2_chunks
        if chunk["chunk_id"] not in v2_embeddings_by_id
    ]

    print("ADT U EMBEDDING VALIDATION")
    print(f"Canonical chunks: {len(chunks)}")
    print(f"Direct canonical embeddings: {len(saved_records)}")
    print(f"Unique direct canonical chunk IDs: {len(embeddings_by_id)}")
    print(f"Canonical chunks pending direct embedding: {len(direct_missing)}")
    print(f"Oversized canonical parents: {len(OVERSIZED_PARENT_IDS)}")
    print(f"Derived child definitions: {len(derived_units)}")
    print(f"Derived child embeddings: {len(derived_vectors_by_id)}")
    print(f"Oversized parents with complete derived coverage: {len(complete_derived_parents)}")
    print(f"Canonical coverage: {canonical_coverage}/{len(chunks)}")
    print(f"Canonical chunks without complete coverage: {missing}")
    print("Invalid canonical or derived embeddings: 0")
    print(f"Dimension: {DIMENSION}")

    print()
    print(f"V2 canonical chunks: {len(v2_chunks)}")
    print(f"V2 direct embeddings: {len(v2_saved_records)}")
    print(f"V2 unique embedded chunk IDs: {len(v2_embeddings_by_id)}")
    print(f"V2 pending: {len(v2_pending)}")
    print(f"V2 coverage: {len(v2_embeddings_by_id)}/{len(v2_chunks)}")

    print()
    combined_total = len(chunks) + len(v2_chunks)
    combined_coverage = canonical_coverage + len(v2_embeddings_by_id)
    print(f"Combined chunks: {combined_total}")
    print(f"Combined coverage: {combined_coverage}/{combined_total}")

    print("\nEMBEDDING VALIDATION RESULT: PASS")


if __name__ == "__main__":
    main()
