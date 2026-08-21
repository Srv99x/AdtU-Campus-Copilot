"""
AdtU Campus Copilot — Gate Calibration Experiment Runner

Loads the frozen calibration dataset, runs the frozen classifier and Chroma
retrieval for each in-scope query, records all distance/metadata signals, and
saves results to data/rag/gate_calibration_results.csv.

Usage:
    python scripts/run_gate_calibration.py

DO NOT modify:
- classifier files
- embedding checkpoints
- ChromaDB data
- calibration dataset
"""

from __future__ import annotations

import sys
import csv
import math
from pathlib import Path
from typing import Any

import chromadb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.classifier.predict import predict_intent
from query_embed import create_query_embedding

CHROMA_DIR        = ROOT / "data" / "processed" / "chroma_db"
COLLECTION_NAME   = "adtu_knowledge"
CALIBRATION_CSV   = ROOT / "data" / "rag" / "gate_calibration_queries.csv"
RESULTS_CSV       = ROOT / "data" / "rag" / "gate_calibration_results.csv"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _distinct_parent_top_k(chunks: list[dict[str, Any]], k: int) -> tuple[list[str], float]:
    """
    Deduplicate by parent_chunk_id before computing mean distance of the top-k
    distinct-parent results.  Returns (deduplicated_chunk_ids_used, mean_distance).
    """
    seen_parents: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for c in chunks:
        parent = c["metadata"].get("parent_chunk_id") or c["chunk_id"]
        if parent not in seen_parents:
            seen_parents.add(parent)
            deduped.append(c)
        if len(deduped) == k:
            break
    distances = [c["distance"] for c in deduped]
    ids       = [c["chunk_id"]  for c in deduped]
    return ids, _mean(distances)


def _check_answer_found(
    expected_evidence: str,
    top3_docs: list[str],
) -> bool:
    """
    Check whether any of the expected evidence tokens are found in the top-3 retrieved text.
    Expected evidence is a semicolon-separated list of substrings.
    Returns True only if ALL evidence tokens are found in the combined top-3 text.
    """
    if not expected_evidence or str(expected_evidence).strip() in ("", "nan"):
        return False
    combined = " ".join(top3_docs).lower()
    tokens   = [t.strip().lower() for t in str(expected_evidence).split(";") if t.strip()]
    return all(tok in combined for tok in tokens)


def run(dry_run: bool = False) -> None:
    print("[calibration] Loading calibration dataset …")
    rows: list[dict[str, str]] = []
    with open(CALIBRATION_CSV, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    print(f"[calibration]   {len(rows)} queries loaded.")

    print("[calibration] Connecting to ChromaDB …")
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"[calibration]   Collection '{COLLECTION_NAME}' has {collection.count()} records.")

    results: list[dict[str, Any]] = []

    for i, row in enumerate(rows, 1):
        query             = row["query"].strip()
        cal_label         = row["calibration_label"].strip()
        expected_intent   = row["expected_intent"].strip()
        expected_evidence = row.get("expected_evidence", "").strip()
        known_chunk_id    = row.get("known_chunk_id", "").strip()

        print(f"\n[calibration] [{i:>2}/50] [{cal_label:<15}] {query[:60]}")

        # --- Classify ---
        predicted_intent = predict_intent(query)
        intent_correct   = (predicted_intent == expected_intent)

        if predicted_intent == "out_of_scope":
            print(f"             -> OUT_OF_SCOPE short-circuit")
            results.append({
                "query":                          query,
                "calibration_label":              cal_label,
                "expected_intent":                expected_intent,
                "predicted_intent":               predicted_intent,
                "intent_correct":                 intent_correct,
                "blocked_oos":                    True,
                "top1_distance":                  "",
                "top3_mean_distance":             "",
                "top5_mean_distance":             "",
                "distinct_parent_top3_mean_distance": "",
                "distinct_parent_top5_mean_distance": "",
                "top1_chunk_id":                  "",
                "top3_chunk_ids":                 "",
                "top5_chunk_ids":                 "",
                "top1_category_match":            "",
                "top3_category_match_rate":       "",
                "any_v2_chunk_in_top5":           "",
                "known_chunk_found":              "",
                "answer_found_in_text":           "",
            })
            continue

        # --- Embed ---
        embedding = create_query_embedding(query)

        # --- Retrieve ---
        chroma_result = collection.query(
            query_embeddings=[embedding],
            n_results=10,
            where={"category": predicted_intent},
            include=["documents", "metadatas", "distances"],
        )
        ids    = chroma_result.get("ids",       [[]])[0]
        docs   = chroma_result.get("documents", [[]])[0]
        metas  = chroma_result.get("metadatas", [[]])[0]
        dists  = chroma_result.get("distances", [[]])[0]

        chunks = [
            {"chunk_id": i_, "document": d, "metadata": m, "distance": dist}
            for i_, d, m, dist in zip(ids, docs, metas, dists)
        ]

        top1  = chunks[0]  if chunks else None
        top3  = chunks[:3]
        top5  = chunks[:5]

        top1_distance          = top1["distance"]               if top1  else float("nan")
        top3_mean_distance     = _mean([c["distance"] for c in top3])
        top5_mean_distance     = _mean([c["distance"] for c in top5])
        top1_chunk_id          = top1["chunk_id"]               if top1  else ""
        top3_chunk_ids         = "; ".join(c["chunk_id"]  for c in top3)
        top5_chunk_ids         = "; ".join(c["chunk_id"]  for c in top5)
        top1_category_match    = (
            top1["metadata"].get("category") == predicted_intent if top1 else False
        )
        top3_cat_matches       = sum(
            1 for c in top3 if c["metadata"].get("category") == predicted_intent
        )
        top3_category_match_rate = top3_cat_matches / len(top3) if top3 else 0.0
        any_v2_in_top5         = any(c["chunk_id"].startswith("v2_") for c in top5)
        known_chunk_found      = bool(known_chunk_id) and any(
            c["chunk_id"] == known_chunk_id for c in chunks
        )

        # Distinct-parent deduplication
        _, dp_top3_mean = _distinct_parent_top_k(chunks, 3)
        _, dp_top5_mean = _distinct_parent_top_k(chunks, 5)

        # Answer found in text
        top3_docs = [c["document"] for c in top3]
        answer_found = _check_answer_found(expected_evidence, top3_docs)

        print(
            f"             -> {predicted_intent:<15} d1={top1_distance:.4f}  "
            f"d3m={top3_mean_distance:.4f}  dp3m={dp_top3_mean:.4f}  "
            f"known={'Y' if known_chunk_found else 'N'}  "
            f"ans={'Y' if answer_found else 'N'}"
        )

        results.append({
            "query":                               query,
            "calibration_label":                   cal_label,
            "expected_intent":                     expected_intent,
            "predicted_intent":                    predicted_intent,
            "intent_correct":                      intent_correct,
            "blocked_oos":                         False,
            "top1_distance":                       round(top1_distance, 6),
            "top3_mean_distance":                  round(top3_mean_distance, 6),
            "top5_mean_distance":                  round(top5_mean_distance, 6),
            "distinct_parent_top3_mean_distance":  round(dp_top3_mean, 6),
            "distinct_parent_top5_mean_distance":  round(dp_top5_mean, 6),
            "top1_chunk_id":                       top1_chunk_id,
            "top3_chunk_ids":                      top3_chunk_ids,
            "top5_chunk_ids":                      top5_chunk_ids,
            "top1_category_match":                 top1_category_match,
            "top3_category_match_rate":            round(top3_category_match_rate, 4),
            "any_v2_chunk_in_top5":                any_v2_in_top5,
            "known_chunk_found":                   known_chunk_found,
            "answer_found_in_text":                answer_found,
        })

    # --- Write results ---
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query", "calibration_label", "expected_intent", "predicted_intent",
        "intent_correct", "blocked_oos",
        "top1_distance", "top3_mean_distance", "top5_mean_distance",
        "distinct_parent_top3_mean_distance", "distinct_parent_top5_mean_distance",
        "top1_chunk_id", "top3_chunk_ids", "top5_chunk_ids",
        "top1_category_match", "top3_category_match_rate",
        "any_v2_chunk_in_top5", "known_chunk_found", "answer_found_in_text",
    ]
    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[calibration] Saved {len(results)} rows -> {RESULTS_CSV}")


if __name__ == "__main__":
    run()
