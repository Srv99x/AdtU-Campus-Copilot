from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

# Ensure we can import from the root directory for scripts
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.classifier.predict import predict_intent
from query_embed import create_query_embedding


@dataclass
class RetrievedChunk:
    chunk_id: str
    document: str
    metadata: dict[str, Any]
    distance: float


@dataclass
class RagResult:
    query: str
    intent: str
    is_out_of_scope: bool
    retrieved_chunks: list[RetrievedChunk]
    confidence_status: str  # e.g., "high", "low", "unassessed", "not_applicable"
    reason: str


def assess_retrieval_confidence(chunks: list[RetrievedChunk]) -> tuple[str, str]:
    """
    Assess whether the retrieved chunks are likely to contain the answer.
    
    WARNING: The confidence threshold is NOT YET CALIBRATED.
    Do NOT invent a guessed threshold.
    """
    if not chunks:
        return "low", "No chunks retrieved."
    
    # Placeholder for actual calibration or LLM grounding logic.
    return "unassessed", "Confidence gate is not yet calibrated."


def run_rag_pipeline(query: str, collection: chromadb.Collection) -> RagResult:
    """
    Execute the first RAG orchestration layer.
    
    1. Classifies intent.
    2. Short-circuits out_of_scope queries without hitting embeddings/Chroma.
    3. Retrieves category-filtered context from ChromaDB for in-scope queries.
    4. Assesses retrieval confidence (uncalibrated placeholder).
    """
    # 1. Classify intent
    intent = predict_intent(query)
    
    # 2. Out of scope short-circuit
    if intent == "out_of_scope":
        return RagResult(
            query=query,
            intent=intent,
            is_out_of_scope=True,
            retrieved_chunks=[],
            confidence_status="not_applicable",
            reason="Query is out of scope. Short-circuited before retrieval."
        )

    # 3. Query embedding
    query_embedding = create_query_embedding(query)
    
    # 4. Category-filtered Chroma retrieval
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where={"category": intent},
        include=["documents", "metadatas", "distances"]
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    chunks = []
    for i, d, m, dist in zip(ids, docs, metas, dists, strict=True):
        chunks.append(RetrievedChunk(chunk_id=i, document=d, metadata=m, distance=dist))

    # 5. Retrieval confidence gate
    confidence_status, reason = assess_retrieval_confidence(chunks)

    return RagResult(
        query=query,
        intent=intent,
        is_out_of_scope=False,
        retrieved_chunks=chunks,
        confidence_status=confidence_status,
        reason=reason
    )
