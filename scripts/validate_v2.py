"""
Validate the V2 knowledge base chunks and cross-check against V1.

Checks:
    - V1 canonical chunks == 842 (read-only)
    - V2 canonical chunk count and metadata completeness
    - No chunk_id collisions between V1 and V2
    - All required fields present and valid
    - Category values in {admissions, fees, facilities}
    - is_table is boolean
    - No empty text
    - No duplicate chunk IDs within V2
    - Token size distribution
"""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

V1_CHUNKS = ROOT / "data" / "processed" / "chunks" / "chunks.jsonl"
V2_CHUNKS = ROOT / "data" / "processed" / "knowledge_v2" / "canonical_chunks.jsonl"

EXPECTED_V1_TOTAL = 842

ALLOWED_CATEGORIES = {"admissions", "fees", "facilities"}

REQUIRED_FIELDS = [
    "source_url",
    "category",
    "section",
    "last_updated",
    "is_table",
]


def estimate_tokens(text: str) -> int:
    """Match the project's 4 chars/token estimation."""
    return max(1, math.ceil(len(text) / 4))


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def is_non_empty(value: object) -> bool:
    """Check if a value is non-empty."""
    return value is not None and str(value).strip() != ""


def main() -> None:

    import sys
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    passed = True

    # --------------------------------------------------
    # V1 READ-ONLY CHECK
    # --------------------------------------------------

    print("=" * 60)
    print("V1 IMMUTABILITY CHECK")
    print("=" * 60)

    if not V1_CHUNKS.exists():
        print(f"ERROR: V1 chunks not found: {V1_CHUNKS}")
        passed = False
    else:
        v1_records = load_jsonl(V1_CHUNKS)
        v1_count = len(v1_records)
        print(f"V1 canonical chunks: {v1_count}")
        print(f"Expected: {EXPECTED_V1_TOTAL}")

        if v1_count == EXPECTED_V1_TOTAL:
            print("V1 count check: PASS")
        else:
            print("V1 count check: FAIL")
            passed = False

        v1_ids = {r.get("chunk_id") for r in v1_records}

    # --------------------------------------------------
    # V2 VALIDATION
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("V2 CHUNK VALIDATION")
    print("=" * 60)

    if not V2_CHUNKS.exists():
        print(f"ERROR: V2 chunks not found: {V2_CHUNKS}")
        return

    v2_records = load_jsonl(V2_CHUNKS)
    v2_count = len(v2_records)

    print(f"V2 canonical chunks: {v2_count}")

    # Field checks
    missing_metadata: list[dict] = []
    invalid_categories: list[dict] = []
    invalid_is_table: list[dict] = []
    empty_sections: list[str] = []
    empty_source_urls: list[str] = []
    empty_last_updated: list[str] = []
    empty_text: list[str] = []
    duplicate_v2_ids: list[str] = []
    v1_v2_collisions: list[str] = []
    needs_review: list[str] = []

    seen_v2_ids: set[str] = set()

    token_sizes: list[int] = []
    table_count = 0
    category_counts: dict[str, int] = {}
    topic_counts: dict[str, int] = {}

    for chunk in v2_records:
        chunk_id = chunk.get("chunk_id", "<missing>")

        # Required fields
        missing = [f for f in REQUIRED_FIELDS if f not in chunk]
        if missing:
            missing_metadata.append({"chunk_id": chunk_id, "fields": missing})

        # Category
        cat = chunk.get("category")
        if cat not in ALLOWED_CATEGORIES:
            invalid_categories.append({"chunk_id": chunk_id, "category": cat})

        category_counts[str(cat)] = category_counts.get(str(cat), 0) + 1

        # Topic
        topic = chunk.get("topic", "unknown")
        topic_counts[str(topic)] = topic_counts.get(str(topic), 0) + 1

        # is_table
        if not isinstance(chunk.get("is_table"), bool):
            invalid_is_table.append({"chunk_id": chunk_id, "value": chunk.get("is_table")})

        if chunk.get("is_table"):
            table_count += 1

        # Section
        if not is_non_empty(chunk.get("section")):
            empty_sections.append(chunk_id)

        # Source URL (allowed to be empty for PDFs without public URLs)
        # but we still track it
        if not is_non_empty(chunk.get("source_url")):
            empty_source_urls.append(chunk_id)

        # Last updated
        if not is_non_empty(chunk.get("last_updated")):
            empty_last_updated.append(chunk_id)

        # Text
        if not is_non_empty(chunk.get("text")):
            empty_text.append(chunk_id)

        # Token size
        text = str(chunk.get("text", ""))
        tokens = estimate_tokens(text)
        token_sizes.append(tokens)

        # Duplicate V2 IDs
        if chunk_id in seen_v2_ids:
            duplicate_v2_ids.append(chunk_id)
        seen_v2_ids.add(chunk_id)

        # V1-V2 collision
        if V1_CHUNKS.exists() and chunk_id in v1_ids:
            v1_v2_collisions.append(chunk_id)

        # Quality
        if chunk.get("quality_status") == "needs_manual_review":
            needs_review.append(chunk_id)

    # Report
    print(f"Category distribution: {category_counts}")
    print(f"Topic distribution: {topic_counts}")
    print(f"Table chunks: {table_count}")
    print(f"Needs manual review: {len(needs_review)}")
    print()

    print(f"Missing required metadata: {len(missing_metadata)}")
    print(f"Invalid categories: {len(invalid_categories)}")
    print(f"Invalid is_table: {len(invalid_is_table)}")
    print(f"Empty sections: {len(empty_sections)}")
    print(f"Empty source_url: {len(empty_source_urls)}")
    print(f"Empty last_updated: {len(empty_last_updated)}")
    print(f"Empty text: {len(empty_text)}")
    print(f"Duplicate V2 IDs: {len(duplicate_v2_ids)}")
    print(f"V1-V2 ID collisions: {len(v1_v2_collisions)}")

    if token_sizes:
        avg_tokens = sum(token_sizes) / len(token_sizes)
        min_tokens = min(token_sizes)
        max_tokens = max(token_sizes)
        print()
        print(f"Token size: min={min_tokens}, avg={avg_tokens:.0f}, max={max_tokens}")

    # Problems
    if missing_metadata:
        print("\nMissing metadata:")
        for item in missing_metadata[:10]:
            print(f"  {item['chunk_id']}: {item['fields']}")

    if invalid_categories:
        print("\nInvalid categories:")
        for item in invalid_categories[:10]:
            print(f"  {item['chunk_id']}: {item['category']}")

    if duplicate_v2_ids:
        print("\nDuplicate V2 IDs:")
        for cid in duplicate_v2_ids[:10]:
            print(f"  {cid}")

    if v1_v2_collisions:
        print("\nV1-V2 ID collisions:")
        for cid in v1_v2_collisions[:10]:
            print(f"  {cid}")

    if empty_text:
        print("\nEmpty text:")
        for cid in empty_text[:10]:
            print(f"  {cid}")

    if needs_review:
        print(f"\nChunks needing manual review:")
        for cid in needs_review:
            print(f"  {cid}")

    # Final verdict
    has_critical = (
        bool(missing_metadata)
        or bool(invalid_categories)
        or bool(invalid_is_table)
        or bool(empty_text)
        or bool(duplicate_v2_ids)
        or bool(v1_v2_collisions)
    )

    print()
    print("=" * 60)

    if has_critical:
        print("V2 VALIDATION RESULT: FAIL (critical issues found)")
        passed = False
    else:
        print("V2 VALIDATION RESULT: PASS")

    if needs_review:
        print(f"  (with {len(needs_review)} chunks flagged for manual review)")

    print("=" * 60)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
