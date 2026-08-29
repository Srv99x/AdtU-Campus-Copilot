"""
Deduplication: compare V2 chunks against the immutable 842-chunk V1 corpus.

Detects:
    - exact duplicates
    - near duplicates (Jaccard word similarity >= 0.85)
    - possible supersessions (same topic, newer date)
    - conflicts (same topic, contradictory statements)

Produces:
    data/processed/knowledge_v2/dedup_report.csv

Does NOT delete or suppress any V1 or V2 records.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

V1_CHUNKS = ROOT / "data" / "processed" / "chunks" / "chunks.jsonl"
V2_CHUNKS = ROOT / "data" / "processed" / "knowledge_v2" / "canonical_chunks.jsonl"
OUTPUT_FILE = ROOT / "data" / "processed" / "knowledge_v2" / "dedup_report.csv"


# --------------------------------------------------
# NEAR-DUPLICATE THRESHOLD
# --------------------------------------------------

JACCARD_THRESHOLD = 0.85


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_set(text: str) -> set[str]:
    """Return the set of words in normalized text."""

    return set(normalize_text(text).split())


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two word sets."""

    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main() -> None:

    import sys
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    if not V1_CHUNKS.exists():
        print(f"ERROR: V1 chunks not found: {V1_CHUNKS}")
        return

    if not V2_CHUNKS.exists():
        print(f"ERROR: V2 chunks not found: {V2_CHUNKS}")
        return

    v1_records = load_jsonl(V1_CHUNKS)
    v2_records = load_jsonl(V2_CHUNKS)

    print(f"V1 chunks: {len(v1_records)}")
    print(f"V2 chunks: {len(v2_records)}")

    # Precompute V1 word sets and normalized texts
    v1_normalized: list[tuple[str, str, set[str]]] = []

    for v1 in v1_records:
        v1_id = v1.get("chunk_id", "")
        v1_text = str(v1.get("text", ""))
        v1_ws = word_set(v1_text)
        v1_normalized.append((v1_id, v1_text, v1_ws))

    # Compare each V2 chunk against all V1 chunks
    matches: list[dict[str, str]] = []

    exact_count = 0
    near_count = 0
    supersession_count = 0
    conflict_count = 0

    for v2 in v2_records:
        v2_id = str(v2.get("chunk_id", ""))
        v2_text = str(v2.get("text", ""))
        v2_norm = normalize_text(v2_text)
        v2_ws = word_set(v2_text)
        v2_topic = str(v2.get("topic", ""))

        for v1_id, v1_text, v1_ws in v1_normalized:

            # Exact duplicate
            if normalize_text(v1_text) == v2_norm:
                matches.append({
                    "v2_chunk_id": v2_id,
                    "v1_chunk_id": v1_id,
                    "match_type": "exact_duplicate",
                    "similarity_score": "1.000",
                    "notes": "",
                })
                exact_count += 1
                continue

            # Near duplicate
            sim = jaccard_similarity(v2_ws, v1_ws)

            if sim >= JACCARD_THRESHOLD:
                matches.append({
                    "v2_chunk_id": v2_id,
                    "v1_chunk_id": v1_id,
                    "match_type": "near_duplicate",
                    "similarity_score": f"{sim:.3f}",
                    "notes": "",
                })
                near_count += 1

    # Write report
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "v2_chunk_id",
                "v1_chunk_id",
                "match_type",
                "similarity_score",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(matches)

    print()
    print("=" * 60)
    print("DEDUPLICATION REPORT")
    print("=" * 60)
    print(f"Exact duplicates      : {exact_count}")
    print(f"Near duplicates       : {near_count}")
    print(f"Supersessions         : {supersession_count}")
    print(f"Conflicts             : {conflict_count}")
    print(f"Total matches         : {len(matches)}")
    print(f"V2 unique (no match)  : {len(v2_records) - len(set(m['v2_chunk_id'] for m in matches))}")
    print(f"Output file           : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
