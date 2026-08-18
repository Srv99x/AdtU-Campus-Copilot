# AdtU Campus Copilot — Data Ingestion Pipeline

## Current status

The approved dataset has 12 official AdtU Markdown source files and 842 validated
chunks. All required metadata is stored directly on each chunk record:

- `source_url`
- `category`
- `section`
- `last_updated`
- `is_table`

The canonical corpus remains exactly **842** immutable chunks. Its direct checkpoint
currently contains **630** valid, 768-dimensional canonical embeddings. The final
ChromaDB database has not been created because embedding coverage is incomplete.

Six canonical IQAC chunks exceed Gemini Embedding 2's 8,192-token input limit. They
remain unchanged in `chunks.jsonl`, but are represented at embedding time by
deterministic derived children in `data/processed/derived_embeddings.json`. The final
representation is **836 direct canonical embeddings** plus the derived children for
those six parents; it is not accurate to describe the final vector count as 842.

Do not rebuild or modify `data/processed/chunks/chunks.jsonl` during normal
embedding or ingestion work.

## Pipeline

```text
data/processed/chunks/chunks.jsonl
        ↓
gemini-embedding-2
        ↓
data/processed/embeddings.json
        ↓
ChromaDB
        ↓
adtu_knowledge
        ↓
cosine retrieval
```

## Validate chunks and saved embeddings

Run these checks before resuming embedding work:

```powershell
python scripts/validate_chunks.py
python scripts/validate_embeddings.py
```

The embedding validation report confirms the total chunks, unique embedded chunk
IDs, missing embeddings, invalid records, and vector dimension. It treats
`chunks.jsonl` as the source of truth and rejects duplicate IDs, unknown IDs,
non-numeric vectors, incorrect dimensions, or saved chunk payloads that do not
match the immutable source record.

## Plan and resume Gemini embeddings

Set secrets locally only; never commit `.env` or an API key.

```powershell
$env:GEMINI_API_KEY = "your-key"
$env:GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
python embed.py --dry-run
```

The dry run loads the 842 chunks, direct checkpoint, and derived-child definition. It
reports direct pending chunks, the deterministic child count per oversized parent,
estimated batch tokens, request count, and pacing, then exits without calling Gemini
or changing either checkpoint.

The embedding job uses stable `chunk_id` values, not the position or count of
existing records. It validates every saved chunk payload and 768-dimensional vector,
skips only valid IDs already present, and atomically saves progress after each
successful batch. It never deletes or regenerates valid progress merely because a
run is interrupted. Derived children resume independently by deterministic
`child_id` in their separate checkpoint.

Derived IDs follow this stable form:

```text
<parent_chunk_id>::embed::v1::part-<zero-padded-index>::<sha256-prefix>
```

Each child copies `source_url`, `category`, `section`, `last_updated`, and `is_table`
from its immutable parent, plus `parent_chunk_id`, `part_index`, `part_count`, and
`derivation_version`. Children are structurally split around notice rows, headings,
and paragraphs, include their source-section context, target 6,000 estimated tokens,
and are rejected if they reach the 8,192-token input limit.

### Quota-safe configuration

The defaults leave safety margin below the previously observed free-tier limits.
They can be adjusted locally without editing Python or committing `.env`:

```powershell
$env:EMBEDDING_BATCH_SIZE = "10"
$env:EMBEDDING_TPM_BUDGET = "25000"
$env:EMBEDDING_RPM_BUDGET = "80"
$env:EMBEDDING_RPD_BUDGET = "900"
$env:EMBEDDING_TOKEN_SAFETY_FACTOR = "1.2"
```

Before a real run, the script prints remaining chunks, batch size, estimated API
requests, conservative estimated token volume, RPM/TPM/RPD budgets, and whether the
whole job fits its configured RPD budget. If it does not fit, it stops before
creating a Gemini client or making an API call. The script paces each batch against
both the configured rolling one-minute request and token budgets; it estimates token
cost locally with the safety factor before sending the batch.

After confirming the dry-run plan fits the available quota, start or resume the
actual job with:

```powershell
python embed.py
```

Gemini daily/token quota exhaustion stops the run with the existing checkpoint
preserved and prints completed and remaining embeddings. A `429 RESOURCE_EXHAUSTED`
quota response is not retried indefinitely. Short-lived rate-limit and transient
failures use bounded exponential backoff. Resume later by running the same command
after quota is available; it will again validate and reuse the successful checkpoint.

## Build ChromaDB after complete canonical coverage

Only after `validate_embeddings.py` reports zero missing embeddings, run:

```powershell
python scripts/chroma_ingest.py
```

The script refuses to build an incomplete collection. Completion means all 836 normal
canonical chunks have direct embeddings and all derived children for the six oversized
parents are present. It creates/updates `adtu_knowledge` in
`data/processed/chroma_db` using cosine distance.

Every Chroma record preserves `chunk_id`, text, embedding, and all five required
metadata fields. Derived records additionally include `parent_chunk_id`,
`part_index`, `part_count`, and `derivation_version`, enabling later retrieval code to
deduplicate citations by canonical parent. The collection record count is therefore
greater than 842. Missing or invalid direct or derived coverage causes ingestion to
fail rather than inserting defaults.

## Test retrieval after the complete collection exists

```powershell
python scripts/chroma_retrieval_test.py
```

The test embeds `What is the hostel fee?` using the configured
`gemini-embedding-2` model, queries `adtu_knowledge`, and displays each top result's
chunk ID, text, cosine distance, and five metadata values. It asserts complete
metadata and requires an official AdtU hostel-fee result. Do not treat this test as
passing until the full 842-vector Chroma collection has been built.
