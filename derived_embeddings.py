"""Build and validate deterministic embedding-only children for oversized chunks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parent
DERIVED_EMBEDDINGS_FILE = ROOT / "data" / "processed" / "derived_embeddings.json"
DERIVATION_VERSION = "v1"
DIMENSION = 768
TOKEN_SAFETY_FACTOR = 1.2
TARGET_DERIVED_TOKENS = 6_000
MAX_INPUT_TOKENS = 8_192
OVERSIZED_PARENT_IDS = frozenset(
    {
        "iqac_225",
        "iqac_228",
        "iqac_239",
        "iqac_240",
        "iqac_242",
        "iqac_243",
    }
)
REQUIRED_METADATA_FIELDS = (
    "source_url",
    "category",
    "section",
    "last_updated",
    "is_table",
)


class DerivedEmbeddingValidationError(ValueError):
    """Raised when derived units or their checkpoint do not match their parents."""


def estimate_tokens(text: str) -> int:
    """Return the same conservative local estimate used by the embedding pipeline."""

    return max(1, math.ceil((len(text) / 4) * TOKEN_SAFETY_FACTOR))


def _is_notice_boundary(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and stripped.startswith("|") and stripped.endswith("|")


def _is_heading_boundary(line: str) -> bool:
    return line.lstrip().startswith("#")


def _structural_blocks(text: str) -> list[str]:
    """Split at notice rows and headings, retaining each row with its detail text."""

    blocks: list[str] = []
    current: list[str] = []

    for line in text.splitlines(keepends=True):
        starts_new_item = _is_notice_boundary(line) or _is_heading_boundary(line)
        if current and starts_new_item:
            block = "".join(current).strip()
            if block:
                blocks.append(block)
            current = []
        current.append(line)

    final_block = "".join(current).strip()
    if final_block:
        blocks.append(final_block)
    return blocks


def _split_long_text(text: str, maximum_characters: int) -> list[str]:
    """Split only when a structural block still exceeds the safe target size."""

    pieces: list[str] = []
    remaining = text.strip()
    while len(remaining) > maximum_characters:
        boundary = remaining.rfind("\n\n", 0, maximum_characters)
        if boundary < maximum_characters // 2:
            boundary = remaining.rfind("\n", 0, maximum_characters)
        if boundary < maximum_characters // 2:
            boundary = remaining.rfind(" ", 0, maximum_characters)
        if boundary < maximum_characters // 2:
            boundary = maximum_characters

        piece = remaining[:boundary].strip()
        if not piece:
            raise DerivedEmbeddingValidationError("Unable to split oversized derived text.")
        pieces.append(piece)
        remaining = remaining[boundary:].strip()

    if remaining:
        pieces.append(remaining)
    return pieces


def _context_prefix(parent: Mapping[str, Any]) -> str:
    section = parent.get("section")
    if not isinstance(section, str) or not section.strip():
        raise DerivedEmbeddingValidationError("Oversized parent has no valid section metadata.")
    return f"Source section: {section.strip()}\n\n"


def split_parent_for_embedding(parent: Mapping[str, Any]) -> list[str]:
    """Create context-preserving, structurally split children below the input limit."""

    parent_id = parent.get("chunk_id")
    text = parent.get("text")
    if not isinstance(parent_id, str) or parent_id not in OVERSIZED_PARENT_IDS:
        raise DerivedEmbeddingValidationError("Parent is not an approved oversized chunk.")
    if not isinstance(text, str) or not text.strip():
        raise DerivedEmbeddingValidationError(f"Parent {parent_id} has no text.")

    prefix = _context_prefix(parent)
    available_characters = int((TARGET_DERIVED_TOKENS * 4) / TOKEN_SAFETY_FACTOR) - len(prefix)
    if available_characters <= 0:
        raise DerivedEmbeddingValidationError(f"Parent {parent_id} section context is too large.")

    atomic_blocks: list[str] = []
    for block in _structural_blocks(text):
        atomic_blocks.extend(_split_long_text(block, available_characters))

    children: list[str] = []
    current = ""
    for block in atomic_blocks:
        candidate = block if not current else f"{current}\n\n{block}"
        if current and estimate_tokens(prefix + candidate) > TARGET_DERIVED_TOKENS:
            children.append(prefix + current)
            current = block
        else:
            current = candidate

    if current:
        children.append(prefix + current)

    if not children or any(estimate_tokens(child) > TARGET_DERIVED_TOKENS for child in children):
        raise DerivedEmbeddingValidationError(
            f"Derived children for {parent_id} exceed the configured target."
        )
    if any(estimate_tokens(child) >= MAX_INPUT_TOKENS for child in children):
        raise DerivedEmbeddingValidationError(
            f"Derived children for {parent_id} exceed Gemini's input limit."
        )
    return children


def build_derived_units(chunks: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create deterministic derived records for exactly the approved parent IDs."""

    chunks_by_id = {chunk.get("chunk_id"): chunk for chunk in chunks}
    missing_parents = OVERSIZED_PARENT_IDS - set(chunks_by_id)
    if missing_parents:
        raise DerivedEmbeddingValidationError(
            f"Missing approved oversized parent IDs: {', '.join(sorted(missing_parents))}"
        )

    units: list[dict[str, Any]] = []
    for parent_id in sorted(OVERSIZED_PARENT_IDS):
        parent = chunks_by_id[parent_id]
        for field in REQUIRED_METADATA_FIELDS:
            if field not in parent:
                raise DerivedEmbeddingValidationError(
                    f"Parent {parent_id} is missing metadata: {field}"
                )

        child_texts = split_parent_for_embedding(parent)
        part_count = len(child_texts)
        for part_index, text in enumerate(child_texts):
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            child_id = (
                f"{parent_id}::embed::{DERIVATION_VERSION}::part-{part_index:03d}::{content_hash}"
            )
            unit: dict[str, Any] = {
                "child_id": child_id,
                "parent_chunk_id": parent_id,
                "part_index": part_index,
                "part_count": part_count,
                "derivation_version": DERIVATION_VERSION,
                "text": text,
            }
            unit.update({field: parent[field] for field in REQUIRED_METADATA_FIELDS})
            units.append(unit)

    child_ids = [unit["child_id"] for unit in units]
    if len(child_ids) != len(set(child_ids)):
        raise DerivedEmbeddingValidationError("Derived child IDs are not unique.")
    return units


def _is_valid_vector(vector: object) -> bool:
    return (
        isinstance(vector, list)
        and len(vector) == DIMENSION
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in vector
        )
    )


def load_derived_embedding_records(
    embeddings_file: Path = DERIVED_EMBEDDINGS_FILE,
) -> list[dict[str, Any]]:
    """Load a derived checkpoint without replacing malformed saved progress."""

    if not embeddings_file.exists():
        return []
    try:
        records = json.loads(embeddings_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DerivedEmbeddingValidationError(
            f"Existing derived embeddings file is unreadable: {embeddings_file}"
        ) from exc
    if not isinstance(records, list):
        raise DerivedEmbeddingValidationError("Derived embeddings checkpoint must be a JSON list.")
    return records


def index_derived_embeddings(
    records: Iterable[object], expected_units: Iterable[Mapping[str, Any]]
) -> dict[str, list[float]]:
    """Validate a child-ID checkpoint against deterministic derived units."""

    expected_by_id = {unit["child_id"]: dict(unit) for unit in expected_units}
    vectors_by_id: dict[str, list[float]] = {}
    for record_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise DerivedEmbeddingValidationError(
                f"Derived embedding record {record_number} is not an object."
            )
        child_id = record.get("child_id")
        if not isinstance(child_id, str) or child_id not in expected_by_id:
            raise DerivedEmbeddingValidationError(
                f"Derived embedding record {record_number} has an unknown child_id."
            )
        if child_id in vectors_by_id:
            raise DerivedEmbeddingValidationError(
                f"Duplicate derived embedding child_id: {child_id}"
            )

        saved_unit = {key: value for key, value in record.items() if key != "embedding"}
        if saved_unit != expected_by_id[child_id]:
            raise DerivedEmbeddingValidationError(
                f"Derived checkpoint payload does not match parent chunk for child_id: {child_id}"
            )
        vector = record.get("embedding")
        if not _is_valid_vector(vector):
            raise DerivedEmbeddingValidationError(
                f"Invalid {DIMENSION}-dimension vector for derived child_id: {child_id}"
            )
        vectors_by_id[child_id] = [float(value) for value in vector]
    return vectors_by_id


def ordered_derived_embedding_records(
    units: Iterable[Mapping[str, Any]], vectors_by_id: Mapping[str, list[float]]
) -> list[dict[str, Any]]:
    """Return a deterministic derived checkpoint ordered by parent and part index."""

    return [
        {**dict(unit), "embedding": vectors_by_id[unit["child_id"]]}
        for unit in units
        if unit["child_id"] in vectors_by_id
    ]


def save_derived_embedding_records(
    records: list[dict[str, Any]],
    embeddings_file: Path = DERIVED_EMBEDDINGS_FILE,
) -> None:
    """Atomically checkpoint derived child vectors after a successful batch."""

    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = embeddings_file.with_suffix(".tmp")
    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_file, embeddings_file)
