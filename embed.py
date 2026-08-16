"""Create resumable Gemini embeddings for the validated AdtU chunk dataset."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


ROOT = Path(__file__).resolve().parent
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks" / "chunks.jsonl"
OUTPUT_FILE = ROOT / "data" / "processed" / "embeddings.json"

DIMENSION = 768
BATCH_SIZE = 1
DELAY_SECONDS = 60
MAX_TRANSIENT_RETRIES = 3
MAX_RATE_LIMIT_RETRIES = 2


class EmbeddingValidationError(ValueError):
    """Raised when saved embedding progress is malformed or inconsistent."""


class QuotaExhaustedError(RuntimeError):
    """Raised when Gemini reports a token or daily quota exhaustion."""


def embedding_model() -> str:
    """Return the configured Gemini embedding model after loading local settings."""

    return os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")


def load_chunks(chunks_file: Path = CHUNKS_FILE) -> list[dict[str, Any]]:
    """Load the immutable chunk source of truth and require unique IDs."""

    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with chunks_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EmbeddingValidationError(
                    f"Invalid JSON in {chunks_file} on line {line_number}: {exc}"
                ) from exc

            if not isinstance(chunk, dict):
                raise EmbeddingValidationError(
                    f"Chunk on line {line_number} is not an object."
                )

            chunk_id = chunk.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise EmbeddingValidationError(
                    f"Chunk on line {line_number} has no valid chunk_id."
                )

            if chunk_id in seen_ids:
                raise EmbeddingValidationError(f"Duplicate chunk_id: {chunk_id}")

            seen_ids.add(chunk_id)
            chunks.append(chunk)

    return chunks


def load_embedding_records(
    embeddings_file: Path = OUTPUT_FILE,
) -> list[dict[str, Any]]:
    """Load existing checkpoint data without ever replacing a bad checkpoint."""

    if not embeddings_file.exists():
        return []

    try:
        with embeddings_file.open("r", encoding="utf-8") as file:
            records = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingValidationError(
            f"Existing embeddings file is unreadable: {embeddings_file}. "
            "It was not modified. Repair or restore it before resuming."
        ) from exc

    if not isinstance(records, list):
        raise EmbeddingValidationError(
            f"Existing embeddings file must contain a JSON list: {embeddings_file}"
        )

    return records


def is_valid_vector(vector: object, dimension: int = DIMENSION) -> bool:
    """Return whether a vector is finite, numeric, and the configured dimension."""

    if not isinstance(vector, list) or len(vector) != dimension:
        return False

    return all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in vector
    )


def index_existing_embeddings(
    records: Iterable[object],
    chunks_by_id: Mapping[str, dict[str, Any]],
    dimension: int = DIMENSION,
) -> dict[str, list[float]]:
    """Validate saved records and return their vectors keyed by stable chunk ID."""

    embeddings_by_id: dict[str, list[float]] = {}

    for record_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise EmbeddingValidationError(
                f"Embedding record {record_number} is not an object."
            )

        saved_chunk = record.get("chunk")
        vector = record.get("embedding")

        if not isinstance(saved_chunk, dict):
            raise EmbeddingValidationError(
                f"Embedding record {record_number} has no chunk object."
            )

        chunk_id = saved_chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise EmbeddingValidationError(
                f"Embedding record {record_number} has no valid chunk_id."
            )

        source_chunk = chunks_by_id.get(chunk_id)
        if source_chunk is None:
            raise EmbeddingValidationError(
                f"Embedding exists for nonexistent chunk_id: {chunk_id}"
            )

        if saved_chunk != source_chunk:
            raise EmbeddingValidationError(
                f"Saved chunk payload does not match chunks.jsonl for chunk_id: {chunk_id}"
            )

        if chunk_id in embeddings_by_id:
            raise EmbeddingValidationError(
                f"Duplicate embedding record for chunk_id: {chunk_id}"
            )

        if not is_valid_vector(vector, dimension):
            raise EmbeddingValidationError(
                f"Invalid {dimension}-dimension vector for chunk_id: {chunk_id}"
            )

        embeddings_by_id[chunk_id] = [float(value) for value in vector]

    return embeddings_by_id


def ordered_embedding_records(
    chunks: Iterable[dict[str, Any]],
    embeddings_by_id: Mapping[str, list[float]],
) -> list[dict[str, Any]]:
    """Build a deterministic checkpoint using the canonical chunk records."""

    return [
        {"chunk": chunk, "embedding": embeddings_by_id[chunk["chunk_id"]]}
        for chunk in chunks
        if chunk["chunk_id"] in embeddings_by_id
    ]


def save_embedding_records(
    records: list[dict[str, Any]],
    embeddings_file: Path = OUTPUT_FILE,
) -> None:
    """Atomically replace the progress file only after a complete JSON write."""

    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = embeddings_file.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_file, embeddings_file)


def _error_text(error: Exception) -> str:
    return str(error).lower()


def is_daily_or_token_quota_error(error: Exception) -> bool:
    """Detect quota failures that cannot be fixed by immediate retries."""

    message = _error_text(error)
    quota_indicators = (
        "daily",
        "per day",
        "rpd",
        "tokens per minute",
        "token quota",
        "tpm",
        "quota exceeded",
        "limit: 0",
    )
    return "resource_exhausted" in message and any(
        indicator in message for indicator in quota_indicators
    )


def is_rate_limit_error(error: Exception) -> bool:
    message = _error_text(error)
    return "429" in message or "resource_exhausted" in message


def is_retryable_transient_error(error: Exception) -> bool:
    message = _error_text(error)
    transient_indicators = (
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "temporarily unavailable",
        "internal",
        "unavailable",
    )
    return any(indicator in message for indicator in transient_indicators)


def retry_delay_seconds(error: Exception, attempt: int) -> int:
    """Use a server-provided retry delay when present, otherwise bounded backoff."""

    retry_match = re.search(r"retry.{0,40}?(\d+)\s*s", str(error), re.IGNORECASE)
    if retry_match:
        return min(max(int(retry_match.group(1)), 1), 300)

    return min(30 * (2 ** (attempt - 1)), 300)


def create_embeddings(
    client: genai.Client,
    texts: list[str],
) -> list[list[float]]:
    """Request vectors with bounded retries and explicit quota handling."""

    contents = [types.Content(parts=[types.Part(text=text)]) for text in texts]
    rate_limit_attempts = 0
    transient_attempts = 0

    while True:
        try:
            result = client.models.embed_content(
                model=embedding_model(),
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=DIMENSION),
            )
            vectors = [list(embedding.values) for embedding in result.embeddings]

            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Gemini returned {len(vectors)} embeddings for {len(texts)} texts."
                )

            if not all(is_valid_vector(vector) for vector in vectors):
                raise RuntimeError("Gemini returned an invalid embedding vector.")

            return vectors

        except Exception as error:
            if is_daily_or_token_quota_error(error):
                raise QuotaExhaustedError(
                    "Gemini reported daily or token quota exhaustion. "
                    "Progress is preserved; resume after quota is available."
                ) from error

            if is_rate_limit_error(error):
                rate_limit_attempts += 1
                if rate_limit_attempts > MAX_RATE_LIMIT_RETRIES:
                    raise QuotaExhaustedError(
                        "Gemini rate limit persisted after bounded retries. "
                        "Progress is preserved; resume later."
                    ) from error
                attempt = rate_limit_attempts
            elif is_retryable_transient_error(error):
                transient_attempts += 1
                if transient_attempts > MAX_TRANSIENT_RETRIES:
                    raise RuntimeError(
                        "Gemini transient failure persisted after bounded retries."
                    ) from error
                attempt = transient_attempts
            else:
                raise

            delay = retry_delay_seconds(error, attempt)
            print(f"Embedding request failed; retrying in {delay} seconds: {error}")
            time.sleep(delay)


def main() -> None:
    """Embed only missing chunk IDs and checkpoint after every successful batch."""

    load_dotenv(ROOT / ".env")
    chunks = load_chunks()
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    existing_records = load_embedding_records()
    embeddings_by_id = index_existing_embeddings(existing_records, chunks_by_id)
    missing_chunks = [chunk for chunk in chunks if chunk["chunk_id"] not in embeddings_by_id]

    print(f"Total chunks: {len(chunks)}")
    print(f"Existing valid embeddings: {len(embeddings_by_id)}")
    print(f"Missing embeddings: {len(missing_chunks)}")

    if not missing_chunks:
        print("All chunk IDs already have valid embeddings.")
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    for offset in range(0, len(missing_chunks), BATCH_SIZE):
        batch = missing_chunks[offset:offset + BATCH_SIZE]
        chunk_ids = [chunk["chunk_id"] for chunk in batch]
        print(f"Embedding missing chunk IDs: {', '.join(chunk_ids)}")

        vectors = create_embeddings(client, [chunk["text"] for chunk in batch])
        for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
            embeddings_by_id[chunk_id] = vector

        checkpoint = ordered_embedding_records(chunks, embeddings_by_id)
        save_embedding_records(checkpoint)
        print(f"Progress safely saved: {len(checkpoint)}/{len(chunks)}")

        if offset + BATCH_SIZE < len(missing_chunks):
            print(f"Waiting {DELAY_SECONDS} seconds before the next request.")
            time.sleep(DELAY_SECONDS)


if __name__ == "__main__":
    main()
