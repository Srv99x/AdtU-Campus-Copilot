"""
AdtU Campus Copilot — FastAPI Backend
Provides a clean HTTP interface over the canonical RAG orchestration pipeline.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure root path is configured for module loading
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Repository-root .env, loaded before any environment-dependent runtime check.
# app/rag/generator.py and query_embed.py already load this file themselves, so
# /chat has always seen it; without this, /ready read only the raw process
# environment and reported "missing or blank" for a key that was present in
# .env and working -- i.e. it told operators the system was down while it ran.
ENV_PATH = ROOT / ".env"

from app.rag.pipeline import run_rag_pipeline, RagResult, Citation, GateMetrics, RetrievedChunk
from app.database.tickets import (
    initialize_database,
    get_ticket,
    list_tickets,
    update_ticket_status,
    Ticket,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application Initialization & State
# ---------------------------------------------------------------------------

app = FastAPI(title="AdtU Campus Copilot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"], # Local Streamlit only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency Injection Holders
class AppState:
    collection: chromadb.Collection | None = None
    db_path: Path | None = None

state = AppState()


def get_db_path() -> Path:
    if state.db_path is None:
        state.db_path = ROOT / "data" / "escalation.db"
    return state.db_path


def get_collection() -> chromadb.Collection:
    if state.collection is None:
        client = chromadb.PersistentClient(path=str(ROOT / "data" / "processed" / "chroma_db"))
        state.collection = client.get_collection("adtu_knowledge")
    return state.collection


@app.on_event("startup")
def startup_event() -> None:
    # Initialize SQLite schema if it doesn't exist
    # Do not execute heavy embeddings or Gemini calls here
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    initialize_database(db_path)


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled error handling {request.method} {request.url}: {exc}")
    # Never expose the raw Python traceback to the caller
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(..., max_length=1000, min_length=1)

class CitationModel(BaseModel):
    chunk_id: str
    parent_chunk_id: str | None
    source_url: str
    section: str
    source_type: str | None

    @classmethod
    def from_domain(cls, citation: Citation) -> CitationModel:
        return cls(
            chunk_id=citation.chunk_id,
            parent_chunk_id=citation.parent_chunk_id,
            source_url=citation.source_url,
            section=citation.section,
            source_type=citation.source_type
        )

class ChatResponse(BaseModel):
    status: str
    intent: str
    answer: str | None
    citations: list[CitationModel] | None
    confidence_status: str | None
    ticket_id: str | None
    reason: str

class TicketStatusUpdateRequest(BaseModel):
    """Body for PATCH /tickets/{ticket_id} (Phase 7C).

    Only the open -> resolved transition is exposed. `Literal["resolved"]`
    means any other value -- including "open", so a resolved ticket can never
    be re-opened through the API -- is rejected by FastAPI's own validation
    as a 422 before any database access happens.
    """
    status: Literal["resolved"]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check() -> dict:
    """Lightweight health check without touching Chroma or Gemini."""
    return {"status": "ok"}


@app.get("/ready")
def readiness_check() -> JSONResponse:
    """Dependency-aware readiness check.

    Verifies (without ever calling Gemini or generating an embedding):
      - GEMINI_API_KEY is present and non-blank in the environment, counting
        the repository-root .env that the rest of the runtime already reads.
      - The canonical runtime Chroma collection (via the existing
        get_collection() singleton/config) is reachable and non-empty.

    Never exposes the API key value itself, only presence/absence.
    """
    checks: dict[str, dict[str, Any]] = {}
    all_ok = True

    # Loaded per call (not just at import) so an operator who fixes .env can
    # re-check without restarting uvicorn. python-dotenv does not override
    # variables already exported in the real process environment.
    load_dotenv(ENV_PATH)

    api_key = os.getenv("GEMINI_API_KEY")
    key_present = bool(api_key and api_key.strip())
    checks["gemini_api_key"] = {
        "ok": key_present,
        "detail": "present" if key_present else "missing or blank",
    }
    if not key_present:
        all_ok = False

    try:
        collection = get_collection()
        count = collection.count()
        if count > 0:
            checks["chroma"] = {
                "ok": True,
                "detail": f"collection accessible with {count} records",
            }
        else:
            checks["chroma"] = {"ok": False, "detail": "collection is empty"}
            all_ok = False
    except Exception as e:
        logger.error(f"Readiness check: Chroma collection unreachable: {e}")
        checks["chroma"] = {"ok": False, "detail": "collection unreachable"}
        all_ok = False

    status_code = status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if all_ok else "not_ready", "checks": checks},
    )


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest,
    collection: chromadb.Collection = Depends(get_collection),
    db_path: Path = Depends(get_db_path)
) -> ChatResponse:
    """Process a user query through the RAG orchestrator."""
    if not request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be whitespace-only.")

    # Call canonical orchestrator
    try:
        rag_result: RagResult = run_rag_pipeline(
            query=request.query, 
            collection=collection, 
            db_path=db_path
        )
    except Exception as e:
        logger.error(f"Orchestrator failure: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Pipeline error")

    if rag_result.status == "error":
        # Controlled error return, preserving the high-level reason but masking the stacktrace
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=rag_result.reason)

    citations = [CitationModel.from_domain(c) for c in rag_result.citations] if rag_result.citations else None

    return ChatResponse(
        status=rag_result.status,
        intent=rag_result.intent,
        answer=rag_result.answer,
        citations=citations,
        confidence_status=rag_result.confidence_status,
        ticket_id=rag_result.ticket_id,
        reason=rag_result.reason
    )


@app.get("/tickets/{ticket_id}")
def get_ticket_endpoint(
    ticket_id: str,
    db_path: Path = Depends(get_db_path)
) -> dict:
    """Retrieve a single ticket by ID."""
    ticket = get_ticket(db_path, ticket_id)
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    return ticket.__dict__


@app.get("/tickets")
def list_tickets_endpoint(
    status_filter: str | None = None,
    db_path: Path = Depends(get_db_path)
) -> list[dict]:
    """List tickets, optionally filtered by status."""
    tickets = list_tickets(db_path, status=status_filter)
    return [t.__dict__ for t in tickets]


@app.patch("/tickets/{ticket_id}")
def update_ticket_endpoint(
    ticket_id: str,
    request: TicketStatusUpdateRequest,
    db_path: Path = Depends(get_db_path)
) -> dict:
    """Resolve an escalated ticket (staff action).

    Additive endpoint (Phase 7C); no existing endpoint or response contract
    is altered. Returns the updated ticket in the same shape as
    GET /tickets/{ticket_id}. Resolving an already-resolved ticket is
    idempotent and succeeds. Raw database errors are never surfaced.
    """
    try:
        ticket = update_ticket_status(db_path, ticket_id, request.status)
    except ValueError as e:
        # Invalid status transition (e.g. re-opening a resolved ticket).
        # Unreachable through this endpoint today -- the request model only
        # permits "resolved" -- but kept so the API stays correct if the
        # exposed transitions are ever widened.
        logger.warning(f"Rejected ticket status update for {ticket_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ticket status transition."
        )
    except Exception as e:
        logger.error(f"Ticket status update failed for {ticket_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not update ticket."
        )

    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")

    return ticket.__dict__
