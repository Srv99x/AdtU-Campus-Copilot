"""Create resumable Gemini embeddings for the validated AdtU chunk dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from dotenv import load_dotenv
from google import genai
from google.genai import types

from derived_embeddings import (
    DERIVED_EMBEDDINGS_FILE,
    MAX_INPUT_TOKENS,
    OVERSIZED_PARENT_IDS,
    build_derived_units,
    index_derived_embeddings,
    load_derived_embedding_records,
    ordered_derived_embedding_records,
    save_derived_embedding_records,
)


ROOT = Path(__file__).resolve().parent
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks" / "chunks.jsonl"
OUTPUT_FILE = ROOT / "data" / "processed" / "embeddings.json"
V2_CHUNKS_FILE = ROOT / "data" / "processed" / "knowledge_v2" / "canonical_chunks.jsonl"
V2_OUTPUT_FILE = ROOT / "data" / "processed" / "v2_embeddings.json"

DIMENSION = 768
MAX_TRANSIENT_RETRIES = 3
MAX_RATE_LIMIT_RETRIES = 2
DEFAULT_BATCH_SIZE = 10
DEFAULT_TPM_BUDGET = 25_000
DEFAULT_RPM_BUDGET = 80
DEFAULT_RPD_BUDGET = 90
DEFAULT_TOKEN_SAFETY_FACTOR = 1.2
RATE_LIMIT_WINDOW_SECONDS = 60
TARGET_BATCH_TOKENS = 20_000


class EmbeddingValidationError(ValueError):
    """Raised when saved embedding progress is malformed or inconsistent."""


class QuotaExhaustedError(RuntimeError):
    """Raised when Gemini reports a token or daily quota exhaustion."""


@dataclass(frozen=True)
class EmbeddingSettings:
    """Quota-safe embedding settings loaded from local environment variables."""

    batch_size: int
    tpm_budget: int
    rpm_budget: int
    rpd_budget: int
    token_safety_factor: float


@dataclass(frozen=True)
class EmbeddingPlan:
    """Preflight estimate for embedding all currently missing chunks."""

    remaining_chunks: int
    estimated_requests: int
    estimated_tokens: int
    fits_rpd_budget: bool
    max_chunks_within_rpd_budget: int


@dataclass(frozen=True)
class EmbeddingWorkItem:
    """One direct canonical or derived child input awaiting an embedding."""

    record_id: str
    text: str
    is_derived: bool
    source: str = "v1"


def embedding_model() -> str:
    """Return the configured Gemini embedding model after loading local settings."""

    return os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")


def _positive_int_environment_value(name: str, default: int) -> int:
    """Read a positive integer setting without silently accepting invalid values."""

    value = os.getenv(name)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc

    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _positive_float_environment_value(name: str, default: float) -> float:
    """Read a positive finite float setting without silently accepting invalid values."""

    value = os.getenv(name)
    if value is None:
        return default

    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number.") from exc

    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return parsed


def embedding_settings() -> EmbeddingSettings:
    """Return quota-safe settings configurable without editing application code."""

    return EmbeddingSettings(
        batch_size=_positive_int_environment_value(
            "EMBEDDING_BATCH_SIZE", DEFAULT_BATCH_SIZE
        ),
        tpm_budget=_positive_int_environment_value(
            "EMBEDDING_TPM_BUDGET", DEFAULT_TPM_BUDGET
        ),
        rpm_budget=_positive_int_environment_value(
            "EMBEDDING_RPM_BUDGET", DEFAULT_RPM_BUDGET
        ),
        rpd_budget=_positive_int_environment_value(
            "EMBEDDING_RPD_BUDGET", DEFAULT_RPD_BUDGET
        ),
        token_safety_factor=_positive_float_environment_value(
            "EMBEDDING_TOKEN_SAFETY_FACTOR", DEFAULT_TOKEN_SAFETY_FACTOR
        ),
    )


def estimate_tokens(text: str, safety_factor: float = DEFAULT_TOKEN_SAFETY_FACTOR) -> int:
    """Conservatively estimate input tokens without making an API request.

    The base estimate uses four characters per token.  A configurable safety factor
    keeps pacing below the nominal Gemini quota despite imperfect local estimation.
    """

    if not math.isfinite(safety_factor) or safety_factor <= 0:
        raise ValueError("Token safety factor must be a positive finite number.")
    return max(1, math.ceil((len(text) / 4) * safety_factor))


def estimate_batch_tokens(
    chunks: Iterable[Mapping[str, Any]], safety_factor: float
) -> int:
    """Estimate the conservative input-token cost of one pending embedding batch."""

    return sum(estimate_tokens(str(chunk["text"]), safety_factor) for chunk in chunks)


def planned_api_requests(remaining_chunks: int, batch_size: int) -> int:
    """Return the minimum number of embedding requests for the remaining chunks."""

    if remaining_chunks < 0:
        raise ValueError("Remaining chunk count cannot be negative.")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    return math.ceil(remaining_chunks / batch_size)


def create_embedding_plan(
    missing_chunks: Sequence[Mapping[str, Any]], settings: EmbeddingSettings
) -> EmbeddingPlan:
    """Calculate the quota-safe preflight plan without contacting Gemini."""

    requests = planned_api_requests(len(missing_chunks), settings.batch_size)
    return EmbeddingPlan(
        remaining_chunks=len(missing_chunks),
        estimated_requests=requests,
        estimated_tokens=estimate_batch_tokens(
            missing_chunks, settings.token_safety_factor
        ),
        fits_rpd_budget=requests <= settings.rpd_budget,
        max_chunks_within_rpd_budget=settings.rpd_budget * settings.batch_size,
    )


def token_aware_batches(
    items: Iterable[EmbeddingWorkItem], settings: EmbeddingSettings
) -> list[list[EmbeddingWorkItem]]:
    """Pack homogeneous work into safe batches without exceeding the token budget."""

    batch_limit = min(TARGET_BATCH_TOKENS, settings.tpm_budget)
    batches: list[list[EmbeddingWorkItem]] = []
    current_batch: list[EmbeddingWorkItem] = []
    current_tokens = 0

    for item in items:
        item_tokens = estimate_tokens(item.text, settings.token_safety_factor)
        if item_tokens >= MAX_INPUT_TOKENS:
            raise EmbeddingValidationError(
                f"Embedding input {item.record_id} exceeds Gemini's "
                f"{MAX_INPUT_TOKENS}-token input limit."
            )
        if item_tokens > batch_limit:
            raise EmbeddingValidationError(
                f"Embedding input {item.record_id} exceeds the safe batch budget "
                f"of {batch_limit} estimated tokens."
            )

        exceeds_count = len(current_batch) >= settings.batch_size
        exceeds_tokens = current_tokens + item_tokens > batch_limit
        if current_batch and (exceeds_count or exceeds_tokens):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(item)
        current_tokens += item_tokens

    if current_batch:
        batches.append(current_batch)
    return batches


def direct_missing_chunks(
    chunks: Iterable[dict[str, Any]], canonical_vectors_by_id: Mapping[str, list[float]]
) -> list[dict[str, Any]]:
    """Return only canonical chunks that still require direct embedding work."""

    return [
        chunk
        for chunk in chunks
        if chunk["chunk_id"] not in canonical_vectors_by_id
        and chunk["chunk_id"] not in OVERSIZED_PARENT_IDS
    ]


def print_derived_dry_run(
    units: Sequence[Mapping[str, Any]], derived_vectors_by_id: Mapping[str, list[float]]
) -> None:
    """Report the deterministic derived representation without calling Gemini."""

    by_parent: dict[str, list[Mapping[str, Any]]] = {}
    for unit in units:
        by_parent.setdefault(str(unit["parent_chunk_id"]), []).append(unit)

    print(f"Oversized canonical parents: {len(by_parent)}")
    for parent_id in sorted(by_parent):
        parent_units = by_parent[parent_id]
        print(f"Derived children for {parent_id}: {len(parent_units)}")
    estimates = [estimate_tokens(str(unit["text"])) for unit in units]
    over_limit = [unit["child_id"] for unit in units if estimate_tokens(str(unit["text"])) >= MAX_INPUT_TOKENS]
    print(f"Total derived children: {len(units)}")
    print(f"Existing derived embeddings: {len(derived_vectors_by_id)}")
    print(f"Derived children pending: {len(units) - len(derived_vectors_by_id)}")
    print(f"Largest estimated derived child tokens: {max(estimates, default=0)}")
    print(f"Smallest estimated derived child tokens: {min(estimates, default=0)}")
    print(f"Derived children at or above {MAX_INPUT_TOKENS} tokens: {len(over_limit)}")


def complete_derived_parent_ids(
    units: Iterable[Mapping[str, Any]], vectors_by_id: Mapping[str, list[float]]
) -> set[str]:
    """Return canonical parents whose deterministic child set is fully checkpointed."""

    by_parent: dict[str, list[Mapping[str, Any]]] = {}
    for unit in units:
        by_parent.setdefault(str(unit["parent_chunk_id"]), []).append(unit)
    return {
        parent_id
        for parent_id, parent_units in by_parent.items()
        if all(unit["child_id"] in vectors_by_id for unit in parent_units)
    }


def print_embedding_plan(plan: EmbeddingPlan, settings: EmbeddingSettings) -> None:
    """Print the plan required before the script is allowed to call Gemini."""

    print("ADT U EMBEDDING PREFLIGHT")
    print(f"Remaining embedding inputs: {plan.remaining_chunks}")
    print(f"Configured batch size: {settings.batch_size}")
    print(f"Estimated API requests: {plan.estimated_requests}")
    print(f"Estimated token volume (with safety factor): {plan.estimated_tokens}")
    print(f"Configured RPM budget: {settings.rpm_budget}")
    print(f"Configured TPM budget: {settings.tpm_budget}")
    print(f"Configured RPD budget: {settings.rpd_budget}")
    print(f"Maximum API requests permitted this run: {settings.rpd_budget}")
    print(f"Fits configured RPD budget: {'YES' if plan.fits_rpd_budget else 'NO'}")
    print(
        "Pacing limit: at most "
        f"{settings.rpm_budget} requests and {settings.tpm_budget} estimated tokens per minute."
    )
    if not plan.fits_rpd_budget:
        print(
            "RPD protection: this run is blocked before Gemini is called. "
            f"The configured daily request budget permits only {settings.rpd_budget} "
            "API request(s) and can cover at most "
            f"{plan.max_chunks_within_rpd_budget} chunks at this batch size."
        )


class MinuteRateLimiter:
    """Keep embedding traffic within configured request and token minute budgets."""

    def __init__(
        self,
        *,
        tpm_budget: int,
        rpm_budget: int,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.tpm_budget = tpm_budget
        self.rpm_budget = rpm_budget
        self.clock = clock
        self.sleeper = sleeper
        self.window_started_at = clock()
        self.used_tokens = 0
        self.used_requests = 0

    def wait_for_capacity(self, estimated_tokens: int) -> None:
        """Pause at a minute boundary when the next batch would exceed a budget."""

        if estimated_tokens <= 0:
            raise ValueError("Estimated batch tokens must be positive.")
        if estimated_tokens > self.tpm_budget:
            raise ValueError(
                "One embedding batch exceeds EMBEDDING_TPM_BUDGET. "
                "Reduce EMBEDDING_BATCH_SIZE or increase the configured budget."
            )

        elapsed = self.clock() - self.window_started_at
        if elapsed >= RATE_LIMIT_WINDOW_SECONDS:
            self.window_started_at = self.clock()
            self.used_tokens = 0
            self.used_requests = 0
        elif (
            self.used_requests + 1 > self.rpm_budget
            or self.used_tokens + estimated_tokens > self.tpm_budget
        ):
            delay = RATE_LIMIT_WINDOW_SECONDS - elapsed
            print(f"Quota pacing: waiting {math.ceil(delay)} seconds for the next minute window.")
            self.sleeper(delay)
            self.window_started_at = self.clock()
            self.used_tokens = 0
            self.used_requests = 0

        self.used_requests += 1
        self.used_tokens += estimated_tokens


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


# --------------------------------------------------
# V2 CHUNK AND EMBEDDING FUNCTIONS
# --------------------------------------------------


def load_v2_chunks(
    v2_chunks_file: Path = V2_CHUNKS_FILE,
    *,
    v1_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load V2 canonical chunks with unique ID and V1 collision validation."""

    if not v2_chunks_file.exists():
        return []

    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with v2_chunks_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EmbeddingValidationError(
                    f"Invalid JSON in {v2_chunks_file} on line {line_number}: {exc}"
                ) from exc

            if not isinstance(chunk, dict):
                raise EmbeddingValidationError(
                    f"V2 chunk on line {line_number} is not an object."
                )

            chunk_id = chunk.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise EmbeddingValidationError(
                    f"V2 chunk on line {line_number} has no valid chunk_id."
                )

            if chunk_id in seen_ids:
                raise EmbeddingValidationError(
                    f"Duplicate V2 chunk_id: {chunk_id}"
                )

            if v1_ids is not None and chunk_id in v1_ids:
                raise EmbeddingValidationError(
                    f"V2 chunk_id collides with V1: {chunk_id}"
                )

            seen_ids.add(chunk_id)
            chunks.append(chunk)

    return chunks


def load_v2_embedding_records(
    embeddings_file: Path = V2_OUTPUT_FILE,
) -> list[dict[str, Any]]:
    """Load V2 embedding checkpoint without replacing bad progress."""

    if not embeddings_file.exists():
        return []

    try:
        with embeddings_file.open("r", encoding="utf-8") as file:
            records = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingValidationError(
            f"Existing V2 embeddings file is unreadable: {embeddings_file}. "
            "Repair or restore it before resuming."
        ) from exc

    if not isinstance(records, list):
        raise EmbeddingValidationError(
            f"V2 embeddings file must contain a JSON list: {embeddings_file}"
        )

    return records


def index_v2_existing_embeddings(
    records: Iterable[object],
    v2_chunks_by_id: Mapping[str, dict[str, Any]],
    dimension: int = DIMENSION,
) -> dict[str, list[float]]:
    """Validate V2 saved records and return vectors keyed by chunk ID."""

    embeddings_by_id: dict[str, list[float]] = {}

    for record_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise EmbeddingValidationError(
                f"V2 embedding record {record_number} is not an object."
            )

        saved_chunk = record.get("chunk")
        vector = record.get("embedding")

        if not isinstance(saved_chunk, dict):
            raise EmbeddingValidationError(
                f"V2 embedding record {record_number} has no chunk object."
            )

        chunk_id = saved_chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise EmbeddingValidationError(
                f"V2 embedding record {record_number} has no valid chunk_id."
            )

        source_chunk = v2_chunks_by_id.get(chunk_id)
        if source_chunk is None:
            raise EmbeddingValidationError(
                f"V2 embedding exists for nonexistent V2 chunk_id: {chunk_id}"
            )

        if saved_chunk != source_chunk:
            raise EmbeddingValidationError(
                f"Saved V2 chunk payload does not match V2 source for chunk_id: "
                f"{chunk_id}"
            )

        if chunk_id in embeddings_by_id:
            raise EmbeddingValidationError(
                f"Duplicate V2 embedding record for chunk_id: {chunk_id}"
            )

        if not is_valid_vector(vector, dimension):
            raise EmbeddingValidationError(
                f"Invalid {dimension}-dimension vector for V2 chunk_id: {chunk_id}"
            )

        embeddings_by_id[chunk_id] = [float(value) for value in vector]

    return embeddings_by_id


def ordered_v2_embedding_records(
    v2_chunks: Iterable[dict[str, Any]],
    v2_embeddings_by_id: Mapping[str, list[float]],
) -> list[dict[str, Any]]:
    """Build a deterministic V2 checkpoint using V2 canonical chunk records."""

    return [
        {"chunk": chunk, "embedding": v2_embeddings_by_id[chunk["chunk_id"]]}
        for chunk in v2_chunks
        if chunk["chunk_id"] in v2_embeddings_by_id
    ]


def save_v2_embedding_records(
    records: list[dict[str, Any]],
    embeddings_file: Path = V2_OUTPUT_FILE,
) -> None:
    """Atomically replace the V2 progress file after a complete JSON write."""

    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = embeddings_file.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_file, embeddings_file)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse local-only command options for the resumable embedding job."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate the embedding plan without calling Gemini or changing progress.",
    )
    return parser.parse_args(argv)


def embed_work_batches(
    client: genai.Client,
    batches: Iterable[Sequence[EmbeddingWorkItem]],
    *,
    settings: EmbeddingSettings,
    rate_limiter: MinuteRateLimiter,
    canonical_chunks: Sequence[dict[str, Any]],
    canonical_vectors_by_id: dict[str, list[float]],
    derived_units: Sequence[Mapping[str, Any]],
    derived_vectors_by_id: dict[str, list[float]],
    v2_chunks: Sequence[dict[str, Any]] | None = None,
    v2_vectors_by_id: dict[str, list[float]] | None = None,
) -> None:
    """Embed and atomically checkpoint V1, V2, and derived records independently."""

    for batch in batches:
        batch_tokens = sum(
            estimate_tokens(item.text, settings.token_safety_factor) for item in batch
        )
        print(f"Embedding inputs: {', '.join(item.record_id for item in batch)}")
        print(f"Estimated batch tokens (with safety factor): {batch_tokens}")
        rate_limiter.wait_for_capacity(batch_tokens)
        vectors = create_embeddings(client, [item.text for item in batch])

        if batch[0].is_derived:
            for item, vector in zip(batch, vectors, strict=True):
                derived_vectors_by_id[item.record_id] = vector
            checkpoint = ordered_derived_embedding_records(
                derived_units, derived_vectors_by_id
            )
            save_derived_embedding_records(checkpoint)
            print(f"Derived progress safely saved: {len(checkpoint)}/{len(derived_units)}")
        elif batch[0].source == "v2":
            assert v2_chunks is not None and v2_vectors_by_id is not None
            for item, vector in zip(batch, vectors, strict=True):
                v2_vectors_by_id[item.record_id] = vector
            checkpoint = ordered_v2_embedding_records(
                v2_chunks, v2_vectors_by_id
            )
            save_v2_embedding_records(checkpoint)
            print(f"V2 progress safely saved: {len(checkpoint)}/{len(v2_chunks)}")
        else:
            for item, vector in zip(batch, vectors, strict=True):
                canonical_vectors_by_id[item.record_id] = vector
            checkpoint = ordered_embedding_records(
                canonical_chunks, canonical_vectors_by_id
            )
            save_embedding_records(checkpoint)
            print(f"V1 progress safely saved: {len(checkpoint)}/{len(canonical_chunks)}")


def main(argv: Sequence[str] | None = None) -> None:
    """Embed V1 canonical, V1 derived, and V2 canonical inputs safely."""

    arguments = parse_arguments(argv)
    load_dotenv(ROOT / ".env")
    settings = embedding_settings()

    # ── V1 canonical ──────────────────────────────────────────────
    chunks = load_chunks()
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    existing_records = load_embedding_records()
    canonical_vectors_by_id = index_existing_embeddings(existing_records, chunks_by_id)

    # ── V1 derived ────────────────────────────────────────────────
    derived_units = build_derived_units(chunks)
    existing_derived_records = load_derived_embedding_records()
    derived_vectors_by_id = index_derived_embeddings(
        existing_derived_records, derived_units
    )

    # ── V2 canonical ──────────────────────────────────────────────
    v1_ids = set(chunks_by_id)
    v2_chunks = load_v2_chunks(v1_ids=v1_ids)
    v2_chunks_by_id = {chunk["chunk_id"]: chunk for chunk in v2_chunks}
    v2_existing_records = load_v2_embedding_records()
    v2_vectors_by_id = index_v2_existing_embeddings(
        v2_existing_records, v2_chunks_by_id
    )

    # ── Pending work ──────────────────────────────────────────────
    pending_direct_chunks = direct_missing_chunks(chunks, canonical_vectors_by_id)
    derived_pending_units = [
        unit for unit in derived_units if unit["child_id"] not in derived_vectors_by_id
    ]
    v2_pending_chunks = [
        chunk for chunk in v2_chunks
        if chunk["chunk_id"] not in v2_vectors_by_id
    ]
    complete_derived_parents_set = complete_derived_parent_ids(
        derived_units, derived_vectors_by_id
    )

    # ── Work items ────────────────────────────────────────────────
    direct_work = [
        EmbeddingWorkItem(
            record_id=chunk["chunk_id"], text=chunk["text"],
            is_derived=False, source="v1",
        )
        for chunk in pending_direct_chunks
    ]
    derived_work = [
        EmbeddingWorkItem(
            record_id=unit["child_id"], text=unit["text"],
            is_derived=True, source="v1",
        )
        for unit in derived_pending_units
    ]
    v2_direct_work = [
        EmbeddingWorkItem(
            record_id=chunk["chunk_id"], text=chunk["text"],
            is_derived=False, source="v2",
        )
        for chunk in v2_pending_chunks
    ]

    # ── Batch planning (unified) ──────────────────────────────────
    direct_batches = token_aware_batches(direct_work, settings)
    derived_batches = token_aware_batches(derived_work, settings)
    v2_batches = token_aware_batches(v2_direct_work, settings)

    all_work = direct_work + derived_work + v2_direct_work
    estimated_tokens_total = sum(
        estimate_tokens(item.text, settings.token_safety_factor) for item in all_work
    )
    estimated_requests = len(direct_batches) + len(derived_batches) + len(v2_batches)

    all_token_estimates = [
        estimate_tokens(item.text, settings.token_safety_factor) for item in all_work
    ]
    largest_input = max(all_token_estimates, default=0)
    oversized_count = sum(1 for t in all_token_estimates if t >= MAX_INPUT_TOKENS)

    plan = EmbeddingPlan(
        remaining_chunks=len(all_work),
        estimated_requests=estimated_requests,
        estimated_tokens=estimated_tokens_total,
        fits_rpd_budget=estimated_requests <= settings.rpd_budget,
        max_chunks_within_rpd_budget=settings.rpd_budget * settings.batch_size,
    )
    v1_canonical_coverage = len(canonical_vectors_by_id) + len(complete_derived_parents_set)

    # ── Report ────────────────────────────────────────────────────
    print("=" * 60)
    print("ADTU EMBEDDING PIPELINE — COMBINED V1+V2 REPORT")
    print("=" * 60)
    print()
    print(f"V1 canonical chunks: {len(chunks)}")
    print(f"V1 existing valid embeddings: {len(canonical_vectors_by_id)}")
    print(f"V1 canonical pending direct: {len(pending_direct_chunks)}")
    print(f"V1 oversized parents: {len(OVERSIZED_PARENT_IDS)}")
    print(f"V1 derived children total: {len(derived_units)}")
    print(f"V1 derived children embedded: {len(derived_vectors_by_id)}")
    print(f"V1 derived pending: {len(derived_pending_units)}")
    print(f"V1 canonical coverage: {v1_canonical_coverage}/{len(chunks)}")
    print()
    print(f"V2 canonical chunks: {len(v2_chunks)}")
    print(f"V2 existing valid embeddings: {len(v2_vectors_by_id)}")
    print(f"V2 canonical pending: {len(v2_pending_chunks)}")
    print(f"V2 derived pending: 0")
    print(f"V2 oversized chunks: 0")
    print()
    print(f"Total pending embedding inputs: {len(all_work)}")
    print(f"Estimated API requests: {estimated_requests}")
    print(f"Estimated token volume (with safety factor): {estimated_tokens_total}")
    print(f"RPM budget: {settings.rpm_budget}")
    print(f"TPM budget: {settings.tpm_budget}")
    print(f"RPD budget: {settings.rpd_budget}")
    print(f"Fits RPD budget: {'YES' if plan.fits_rpd_budget else 'NO'}")
    print(f"Largest input (estimated tokens): {largest_input}")
    print(f"Oversized inputs (>= {MAX_INPUT_TOKENS} tokens): {oversized_count}")
    if estimated_requests > 0:
        rpm_minutes = math.ceil(estimated_requests / settings.rpm_budget)
        tpm_minutes = math.ceil(estimated_tokens_total / settings.tpm_budget)
        overhead_seconds = estimated_requests * 2  # ~2s per-request overhead
        pacing_minutes = max(rpm_minutes, tpm_minutes)
        est_minutes = pacing_minutes + math.ceil(overhead_seconds / 60)
        print(f"Estimated runtime: ~{est_minutes} minute(s)")
        print(f"  (RPM-paced: {rpm_minutes} min, TPM-paced: {tpm_minutes} min, overhead: ~{overhead_seconds}s)")
    else:
        print("Estimated runtime: 0 minutes (nothing to embed)")
    print()
    print_derived_dry_run(derived_units, derived_vectors_by_id)

    if arguments.dry_run:
        all_ids = [item.record_id for item in all_work]
        has_dupes = len(all_ids) != len(set(all_ids))
        print()
        print("V1 embeddings unchanged: YES")
        print("V1 chunks unchanged: YES")
        print(f"Duplicate IDs in work queue: {'YES — ERROR' if has_dupes else 'NO'}")
        print("API calls made: 0")
        print()
        print(
            "DRY RUN: no Gemini API calls were made and no embedding "
            "checkpoint was modified."
        )
        return

    if not all_work:
        print("All chunks have complete embedding coverage.")
        return

    if not plan.fits_rpd_budget:
        print(
            "Embedding run stopped before Gemini API calls because "
            "RPD protection blocked it."
        )
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    rate_limiter = MinuteRateLimiter(
        tpm_budget=settings.tpm_budget,
        rpm_budget=settings.rpm_budget,
    )
    try:
        embed_work_batches(
            client, direct_batches,
            settings=settings, rate_limiter=rate_limiter,
            canonical_chunks=chunks,
            canonical_vectors_by_id=canonical_vectors_by_id,
            derived_units=derived_units,
            derived_vectors_by_id=derived_vectors_by_id,
            v2_chunks=v2_chunks,
            v2_vectors_by_id=v2_vectors_by_id,
        )
        embed_work_batches(
            client, derived_batches,
            settings=settings, rate_limiter=rate_limiter,
            canonical_chunks=chunks,
            canonical_vectors_by_id=canonical_vectors_by_id,
            derived_units=derived_units,
            derived_vectors_by_id=derived_vectors_by_id,
            v2_chunks=v2_chunks,
            v2_vectors_by_id=v2_vectors_by_id,
        )
        embed_work_batches(
            client, v2_batches,
            settings=settings, rate_limiter=rate_limiter,
            canonical_chunks=chunks,
            canonical_vectors_by_id=canonical_vectors_by_id,
            derived_units=derived_units,
            derived_vectors_by_id=derived_vectors_by_id,
            v2_chunks=v2_chunks,
            v2_vectors_by_id=v2_vectors_by_id,
        )
    except QuotaExhaustedError as error:
        v1_complete = len(canonical_vectors_by_id) + len(
            complete_derived_parent_ids(derived_units, derived_vectors_by_id)
        )
        print(f"Embedding run stopped safely: {error}")
        print(f"V1 canonical coverage preserved: {v1_complete}/{len(chunks)}")
        print(f"V1 direct embeddings: {len(canonical_vectors_by_id)}")
        print(f"V1 derived embeddings: {len(derived_vectors_by_id)}/{len(derived_units)}")
        print(f"V2 embeddings: {len(v2_vectors_by_id)}/{len(v2_chunks)}")


if __name__ == "__main__":
    main()
