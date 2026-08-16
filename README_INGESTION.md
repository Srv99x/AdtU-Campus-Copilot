# AdtU Campus Copilot — Data Ingestion Pipeline

## Current status

The approved dataset has 12 official AdtU Markdown source files and 842 validated
chunks. All required metadata is stored directly on each chunk record:

- `source_url`
- `category`
- `section`
- `last_updated`
- `is_table`

`data/processed/embeddings.json` currently contains 520 valid, 768-dimensional
embeddings. The remaining 322 chunk IDs are pending. The final Chroma database has
not been built because the corpus is incomplete.

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

## Resume Gemini embeddings

Set secrets locally only; never commit `.env` or an API key.

```powershell
$env:GEMINI_API_KEY = "your-key"
$env:GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
python embed.py
```

The embedding job uses stable `chunk_id` values, not the position or count of
existing records. It validates the existing checkpoint, skips only the IDs that
already have valid embeddings, and atomically saves progress after each successful
request. It never deletes or regenerates valid progress merely because a run is
interrupted.

Gemini daily/token quota exhaustion stops the run with the existing checkpoint
preserved. Short-lived rate-limit and transient failures use bounded exponential
backoff; they are not retried indefinitely. Resume later by running the same
command after quota is available.

## Build ChromaDB after all 842 embeddings exist

Only after `validate_embeddings.py` reports zero missing embeddings, run:

```powershell
python scripts/chroma_ingest.py
```

The script refuses to build an incomplete collection. It creates/updates the
`adtu_knowledge` collection in `data/processed/chroma_db` using cosine distance.
Each record uses its stable `chunk_id` as the Chroma ID and preserves the chunk
text, embedding, and all five required metadata fields. Missing or invalid metadata
causes ingestion to fail rather than inserting defaults.

## Test retrieval after the complete collection exists

```powershell
python scripts/chroma_retrieval_test.py
```

The test embeds `What is the hostel fee?` using the configured
`gemini-embedding-2` model, queries `adtu_knowledge`, and displays each top result's
chunk ID, text, cosine distance, and five metadata values. It asserts complete
metadata and requires an official AdtU hostel-fee result. Do not treat this test as
passing until the full 842-vector Chroma collection has been built.
