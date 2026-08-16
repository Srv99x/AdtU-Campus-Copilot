from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CHUNKS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "chunks"
    / "chunks.jsonl"
)

EXPECTED_TOTAL = 842

REQUIRED_FIELDS = [
    "source_url",
    "category",
    "section",
    "last_updated",
    "is_table",
]

ALLOWED_CATEGORIES = {
    "admissions",
    "fees",
    "facilities",
}


def is_non_empty(value) -> bool:
    return value is not None and str(value).strip() != ""


def main() -> None:

    if not CHUNKS_FILE.exists():
        print(f"ERROR: File not found:\n{CHUNKS_FILE}")
        return

    chunks = []

    with CHUNKS_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            if not line.strip():
                continue

            try:
                chunks.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:
                print(
                    f"INVALID JSON on line {line_number}: {exc}"
                )

    total = len(chunks)

    complete_metadata = 0

    missing_metadata = []
    invalid_categories = []
    invalid_is_table = []
    empty_sections = []
    empty_source_urls = []
    empty_last_updated = []
    duplicate_chunk_ids = []
    empty_text = []

    seen_chunk_ids = set()

    for chunk in chunks:

        chunk_id = chunk.get(
            "chunk_id",
            "<missing chunk_id>"
        )

        # Required fields
        missing_fields = [
            field
            for field in REQUIRED_FIELDS
            if field not in chunk
        ]

        if missing_fields:
            missing_metadata.append({
                "chunk_id": chunk_id,
                "fields": missing_fields,
            })

        # Category
        category = chunk.get("category")

        if category not in ALLOWED_CATEGORIES:
            invalid_categories.append({
                "chunk_id": chunk_id,
                "category": category,
            })

        # is_table must be a real boolean
        if not isinstance(
            chunk.get("is_table"),
            bool
        ):
            invalid_is_table.append({
                "chunk_id": chunk_id,
                "value": chunk.get("is_table"),
            })

        # Section
        if not is_non_empty(
            chunk.get("section")
        ):
            empty_sections.append(
                chunk_id
            )

        # Source URL
        source_url = chunk.get(
            "source_url"
        )

        if not is_non_empty(source_url):
            empty_source_urls.append(
                chunk_id
            )

        # Last updated
        if not is_non_empty(
            chunk.get("last_updated")
        ):
            empty_last_updated.append(
                chunk_id
            )

        # Text
        if not is_non_empty(
            chunk.get("text")
        ):
            empty_text.append(
                chunk_id
            )

        # Duplicate chunk IDs
        if chunk_id in seen_chunk_ids:
            duplicate_chunk_ids.append(
                chunk_id
            )

        seen_chunk_ids.add(chunk_id)

        # Complete metadata
        if (
            not missing_fields
            and category in ALLOWED_CATEGORIES
            and isinstance(
                chunk.get("is_table"),
                bool
            )
            and is_non_empty(
                chunk.get("section")
            )
            and is_non_empty(
                chunk.get("source_url")
            )
            and is_non_empty(
                chunk.get("last_updated")
            )
        ):
            complete_metadata += 1

    print()
    print("=" * 60)
    print("ADT U CHUNK METADATA VALIDATION")
    print("=" * 60)

    print(
        f"Total chunks: {total}"
    )

    print(
        f"Expected chunks: {EXPECTED_TOTAL}"
    )

    if total == EXPECTED_TOTAL:
        print("Count check: PASS")
    else:
        print("Count check: FAIL")

    print(
        f"Complete metadata: {complete_metadata}"
    )

    print(
        f"Missing metadata: {len(missing_metadata)}"
    )

    print(
        f"Invalid categories: {len(invalid_categories)}"
    )

    print(
        f"Invalid is_table values: {len(invalid_is_table)}"
    )

    print(
        f"Empty section: {len(empty_sections)}"
    )

    print(
        f"Empty source_url: {len(empty_source_urls)}"
    )

    print(
        f"Empty last_updated: {len(empty_last_updated)}"
    )

    print(
        f"Empty text: {len(empty_text)}"
    )

    print(
        f"Duplicate chunk IDs: {len(duplicate_chunk_ids)}"
    )

    print()
    print("-" * 60)
    print("PROBLEM DETAILS")
    print("-" * 60)

    if missing_metadata:
        print("\nMissing metadata:")
        for item in missing_metadata:
            print(
                f"  {item['chunk_id']}: "
                f"{item['fields']}"
            )

    if invalid_categories:
        print("\nInvalid categories:")
        for item in invalid_categories:
            print(
                f"  {item['chunk_id']}: "
                f"{item['category']}"
            )

    if invalid_is_table:
        print("\nInvalid is_table values:")
        for item in invalid_is_table:
            print(
                f"  {item['chunk_id']}: "
                f"{item['value']}"
            )

    if empty_sections:
        print("\nEmpty sections:")
        for chunk_id in empty_sections:
            print(f"  {chunk_id}")

    if empty_source_urls:
        print("\nEmpty source URLs:")
        for chunk_id in empty_source_urls:
            print(f"  {chunk_id}")

    if empty_last_updated:
        print("\nEmpty last_updated:")
        for chunk_id in empty_last_updated:
            print(f"  {chunk_id}")

    if duplicate_chunk_ids:
        print("\nDuplicate chunk IDs:")
        for chunk_id in duplicate_chunk_ids:
            print(f"  {chunk_id}")

    print()
    print("=" * 60)

    passed = (
        total == EXPECTED_TOTAL
        and complete_metadata == total
        and not missing_metadata
        and not invalid_categories
        and not invalid_is_table
        and not empty_sections
        and not empty_source_urls
        and not empty_last_updated
        and not duplicate_chunk_ids
        and not empty_text
    )

    if passed:
        print("VALIDATION RESULT: PASS")
    else:
        print("VALIDATION RESULT: FAIL")

    print("=" * 60)


if __name__ == "__main__":
    main()