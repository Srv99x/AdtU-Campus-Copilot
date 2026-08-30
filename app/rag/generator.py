"""
AdtU Campus Copilot — Stage 2: Evidence-Grounded Generation

This module is responsible for strictly generating an answer based ONLY on the
retrieved knowledge chunks passed to it. It does not perform retrieval itself.

IMPORTANT:
- No external knowledge is allowed.
- No inference, extrapolation, or hallucination.
- If the evidence is insufficient, it strictly returns INSUFFICIENT_EVIDENCE.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.rag.pipeline import RetrievedChunk


# Provider HTTP codes that mean "the generation service could not serve this
# request right now" rather than "this pipeline has a defect": 429 quota or
# rate limit, 503 overloaded, 500/502/504 provider-side faults, and 404 for a
# model that has been retired or is not available to this key.
_UNAVAILABLE_PROVIDER_CODES: frozenset[int] = frozenset({404, 429, 500, 502, 503, 504})


class GenerationUnavailableError(RuntimeError):
    """Stage 2 could not reach a working generation service.

    Raised ONLY when retrieval and the Stage 1 confidence gate have already
    succeeded -- so the knowledge base and the evidence are fine, and the
    identical query will usually succeed once the provider recovers. This is
    deliberately distinct from every other failure, which stays a generic
    internal error.

    Carries no provider text: callers must never surface quota figures, model
    ids, API keys, or raw provider errors to a user.
    """


@dataclass
class Citation:
    chunk_id: str
    parent_chunk_id: str | None
    source_url: str
    section: str
    source_type: str | None


@dataclass
class GenerationResult:
    status: str  # "answered" or "insufficient_evidence"
    answer: str
    citations: list[Citation]


SYSTEM_PROMPT = """You are the AdtU Campus Copilot.
Answer ONLY using the supplied evidence.
Do not use outside knowledge.
Do not infer, extrapolate, calculate, or invent facts.
If the supplied evidence does not explicitly support the answer, return exactly:
INSUFFICIENT_EVIDENCE
When answering, cite the exact source/document from the supplied metadata."""


def _generation_model() -> str:
    """Return the configured Gemini generation model.

    The fallback was "gemini-2.5-flash" (the model this pipeline was
    validated on). Google has since retired that model for new API keys --
    it now answers 404 NOT_FOUND with "no longer available to new users.
    Please update your code to use models/gemini-3.6-flash" -- so an
    environment that relied on this default failed *every* /chat call at
    Stage 2. Only the fallback model id changes here; the SYSTEM_PROMPT
    grounding contract, temperature, and INSUFFICIENT_EVIDENCE handling
    are untouched.
    """
    load_dotenv(ROOT / ".env")
    return os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.6-flash")


def format_evidence(chunks: list[RetrievedChunk]) -> str:
    """Format chunks into a structured context block for the prompt."""
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        # We supply the exact metadata fields for citation
        c_id = chunk.chunk_id
        parent = chunk.metadata.get("parent_chunk_id", "None")
        url = chunk.metadata.get("source_url", "Unknown")
        section = chunk.metadata.get("section", "Unknown")

        metadata_lines = [
            f"  chunk_id: {c_id}",
            f"  parent_chunk_id: {parent}",
            f"  source_url: {url}",
            f"  section: {section}",
        ]

        # Routine/table chunks (e.g. class_routine) carry program/
        # specialization/semester only in metadata -- their own text has no
        # such wording. Expose each only when actually present; never
        # fabricate a value for a chunk that lacks it.
        for label in ("program", "specialization", "semester"):
            value = chunk.metadata.get(label)
            if value:
                metadata_lines.append(f"  {label}: {value}")

        block = (
            f"--- EVIDENCE CHUNK {i} ---\n"
            f"Metadata:\n" + "\n".join(metadata_lines) + "\n"
            f"Content:\n{chunk.document}\n"
            f"-------------------------"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def extract_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Map retrieved chunks to strict citation records, resolving to parents when derived."""
    seen = set()
    citations = []
    for chunk in chunks:
        # Resolve to canonical parent if it's a derived child
        parent_id = chunk.metadata.get("parent_chunk_id")
        citation_id = parent_id if parent_id else chunk.chunk_id
        
        if citation_id not in seen:
            seen.add(citation_id)
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    parent_chunk_id=parent_id,
                    source_url=str(chunk.metadata.get("source_url", "")),
                    section=str(chunk.metadata.get("section", "")),
                    source_type=str(chunk.metadata.get("source_type", "")) if "source_type" in chunk.metadata else None,
                )
            )
    return citations


def format_citation_reference(
    *,
    chunk_id: str,
    parent_chunk_id: str | None,
    source_url: str,
    section: str,
    source_type: str | None = None,
) -> str:
    """Return a human-readable citation reference for display (Markdown-safe).

    - When *source_url* is present and non-blank, returns the existing
      clickable "[section](url)" form, unchanged.
    - When *source_url* is missing/blank (e.g. all current V2 chunks), NO
      URL is fabricated. Instead returns a clear, non-clickable identity
      built only from metadata already present on the citation (section,
      chunk_id, source_type).
    - Derived-child lineage (parent_chunk_id) is appended in both cases,
      when present, so citation provenance is never lost.

    Keyword arguments mirror the field names on both the domain `Citation`
    dataclass and the API's `CitationModel`/JSON response, so callers on
    either side of the FastAPI boundary can invoke this with `**citation`
    (a dict) or explicit fields from a `Citation` instance without any
    change to either contract.
    """
    section_label = section.strip() if section and section.strip() else chunk_id
    clean_url = source_url.strip() if source_url else ""

    if clean_url:
        line = f"[{section_label}]({clean_url})"
    else:
        if section_label != chunk_id:
            identity = f"{section_label} (chunk `{chunk_id}`)"
        else:
            identity = f"chunk `{chunk_id}`"
        if source_type:
            identity += f" — {source_type}"
        line = f"{identity} — source link unavailable"

    if parent_chunk_id:
        line += f" (via `{parent_chunk_id}`)"

    return line


def generate_grounded_answer(query: str, evidence: list[RetrievedChunk]) -> GenerationResult:
    """
    Generate an answer using ONLY the supplied evidence.
    
    If the evidence does not support the answer, this returns a status of
    'insufficient_evidence' and does NOT guess.
    
    Args:
        query: The user's original query.
        evidence: The chunks that passed the Stage 1 confidence gate.
        
    Returns:
        GenerationResult containing status, answer text, and mapped citations.
    """
    if not evidence:
        return GenerationResult(
            status="insufficient_evidence",
            answer="INSUFFICIENT_EVIDENCE",
            citations=[]
        )

    model = _generation_model()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    formatted_context = format_evidence(evidence)
    
    prompt = (
        f"USER QUERY:\n{query}\n\n"
        f"SUPPLIED EVIDENCE:\n{formatted_context}\n"
    )

    client = genai.Client(api_key=api_key)
    
    # We use generate_content with a strict system instruction
    try:
        result = client.models.generate_content(
            model=model,
            contents=[types.Content(parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,  # Zero temperature for strict grounding
            )
        )
    except genai_errors.APIError as exc:
        # Quota/overload/model-availability failures are a property of the
        # provider, not of the retrieved evidence. Translate them into one
        # typed error so the API can say "temporarily unavailable" instead of
        # reporting an internal fault that makes the KB look broken. Any other
        # provider error keeps propagating unchanged.
        if exc.code in _UNAVAILABLE_PROVIDER_CODES:
            raise GenerationUnavailableError(
                "The generation service is temporarily unavailable."
            ) from exc
        raise

    raw_text = result.text if result.text else ""
    cleaned_text = raw_text.strip()

    if cleaned_text == "INSUFFICIENT_EVIDENCE":
        return GenerationResult(
            status="insufficient_evidence",
            answer=cleaned_text,
            citations=[] # No citations for missing evidence
        )

    return GenerationResult(
        status="answered",
        answer=cleaned_text,
        citations=extract_citations(evidence)
    )
