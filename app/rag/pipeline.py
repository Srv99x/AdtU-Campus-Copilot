"""
AdtU Campus Copilot — RAG Orchestration Pipeline

Implements the full query-to-answer orchestration flow:
    Query
    -> Intent classifier (frozen)
    -> out_of_scope short-circuit  [Policy B: campus vocab gate may allow ONE recovery]
    -> Query embedding (gemini-embedding-2)
    -> Category-filtered Chroma retrieval
    -> Stage 1 confidence gate (empirically calibrated)
    -> Stage 2 LLM grounding (Gemini generation)
    -> SQLite escalation (fallback for unverified queries)

Policy B — OOS Recovery (Phase 1):
    If the frozen classifier predicts out_of_scope AND the query matches at least
    one strong campus-domain term (or two context-only campus terms), ONE recovery
    retrieval is attempted against the "facilities" category.  The existing Stage 1
    gate (theta=0.275, min 3 distinct parents) and Stage 2 Gemini grounding are
    applied without modification.  If Stage 1 or Stage 2 fails, the result is a
    SQLite escalation ticket — never a fabricated answer.

    Ticket sources used for recovery escalations:
        "oos_recovery_stage1_low"
        "oos_recovery_stage2_insufficient_evidence"

Holiday/Calendar Retrieval Normalization (Phase 2):
    Diagnostic data showed that the original query "When is the Bihu holiday?"
    produces dp-top3-mean=0.2834 (LOW) while "Bihu holiday academic calendar"
    produces dp-top3-mean=0.2178 (HIGH) and correctly surfaces the holiday chunk.

    Rule (applied ONLY inside _run_oos_recovery, ONLY for the embedding call):
        if matched vocabulary contains "holiday" OR "bihu":
            append "academic calendar" to the retrieval query
        otherwise:
            keep original query

    The user-visible query, classifier prediction, intent field, gate constants,
    and Stage 2 grounding are ALL unchanged.  This is a retrieval-only tweak.

Scholarship Monetary Retrieval Override (Phase 3):
    Audit found that "scholarship" queries asking about an AMOUNT/PERCENTAGE/
    DISCOUNT (e.g. "What scholarship is available for CBSE board students with
    95%?") are inconsistently classified across admissions/facilities/fees/
    out_of_scope by the frozen TF-IDF+LinearSVC model, yet the only chunks that
    contain the actual scholarship table (v2_scholarships_0, v2_scholarships_1)
    live under category="fees".  A single-category `where` filter tied 1:1 to
    the (unreliable, for this topic) classifier label means most such queries
    can never retrieve the answer, no matter how good Stage 1/Stage 2 are.

    Rule (applied ONLY to retrieval-category selection, never to `intent`):
        if the query mentions scholarship vocabulary
           AND a monetary/percentage/marks/fee-discount signal
           (e.g. "95%", "discount", "waiver", "amount", "concession",
           "marks", "kitna"):
            retrieval category = "fees"
        else:
            retrieval category = the classifier-selected category (unchanged)

    This intentionally does NOT catch every scholarship query — application/
    eligibility/portal-style scholarship questions ("NSP scholarship portal",
    "scholarship eligibility") are left on the classifier's own category,
    since they are not blocked by the same failure mode and forcing them to
    "fees" would just trade one false negative for another.

    Applies in two places, both retrieval-category-only, both still a single
    embedding call:
        - the normal in-scope path (step 3 of run_rag_pipeline), overriding
          the `where` filter category but never RagResult.intent
        - _run_oos_recovery, overriding the hard-coded "facilities" recovery
          category to "fees" for monetary scholarship queries only; all other
          OOS recovery queries keep targeting "facilities" exactly as before

Scholarship Retrieval-Query Normalization (Phase 4):
    A read-only diagnostic study found that v2_scholarships_0 is retrieved as
    rank 1 or 2 for every monetary scholarship query tested, yet Stage 1 often
    still fails because GATE_MIN_DISTINCT_PARENTS=3 pulls in an unrelated
    third "fees" chunk that dilutes the top-3-mean distance. Appending the
    literal phrase pattern used throughout the scholarship table itself
    ("...SCHOLARSHIP ON SEMESTER FEE...") to the retrieval embedding text
    pulled both scholarship chunks closer without ever regressing a query
    that already passed — unlike other candidate suffixes tested (e.g. "fee
    scholarship discount", which regressed a passing query and was rejected).

    Rule (applied ONLY to the embedding text, gated on the ORIGINAL query):
        if _is_monetary_scholarship_query(original_query) is True
           AND "semester fee" is not already present (case-insensitive):
            append " scholarship semester fee" to the retrieval text
        otherwise:
            keep the retrieval text unchanged

    The monetary check is evaluated exactly once, on the original query,
    by the caller — it is NEVER re-run on the normalized/appended text.
    RagResult.query, classifier intent, and retrieval category selection
    (Phase 3) are all unaffected; this is a retrieval-embedding-text-only
    tweak, applied in the same two call sites as Phase 3 (the normal
    in-scope path and _run_oos_recovery), always as a single embedding call.

V2 Class-Routine Sibling Expansion (Phase 5):
    A read-only audit found that class_routine.md (57 of 76 V2 chunks, 75%
    of the V2 corpus) is chunked into short "heading" chunks (program /
    specialization / semester / section identifiers, is_table=False) and
    separate "table" chunks (the actual day/time/faculty grid, is_table=True)
    whose raw text contains no program-identifying wording at all. Vector
    retrieval for a natural-language routine query reliably surfaces the
    heading (short, keyword-dense) but — confirmed empirically across 5
    tested query variants, 0/5 — never the paired table. Stage 1 can then
    pass "HIGH" on the strength of several unrelated sections' headings
    clustering together, while Stage 2 never actually receives the
    timetable content it needs to answer from.

    Correction to the original audit: it named "_9 <-> _17" as a known
    non-adjacent sibling pair. Full 4-field metadata verification (this
    phase) shows _9 is semester="1st year 1st semester" while _17 is
    semester="2nd year 1st semester" — they are NOT siblings (the audit had
    only compared the `program` field). _9 is in fact self-contained (its
    own text already includes the full heading line) and correctly forms no
    sibling pair under the rule below — a safe, deliberate non-match, not
    a bug. The chosen matching rule is verified against the live corpus,
    not against the original (partially incorrect) audit claim.

    Pairing rule (pure, deterministic, no Chroma/Gemini access — see
    _find_class_routine_sibling_ids):
        1. Metadata tier (primary): for every one of
           program/specialization/semester/section that is non-empty on the
           anchor chunk, the candidate must match EXACTLY; fields empty on
           the anchor are not required to match. At least one field must
           have been checked (chunks are never paired on topic alone).
        2. Adjacency tier (fallback, used only when the metadata tier finds
           zero matches — chiefly the ~32% of class_routine chunks whose
           program/specialization/semester extraction failed): candidates
           within +-1 chunk_index of the anchor, same document group
           (parsed from the "v2_class_routine_<N>" chunk_id, since Chroma's
           own document/chunk_index metadata fields are unpopulated for
           this corpus), and REJECTED if any populated field conflicts with
           the anchor (cross-section contamination guard).

    Architectural rule: expansion happens strictly AFTER the Stage 1 gate
    has already been computed from the original Chroma result set. Gate
    metrics (GATE_THETA_D, GATE_MIN_DISTINCT_PARENTS, dp-top3-mean,
    distinct_parent_count) are NEVER recomputed on the expanded set — a LOW
    original retrieval stays LOW and escalates exactly as before. Sibling
    lookup uses a single collection.get(where=...) metadata-only fetch (no
    embedding call), applied only when at least one retrieved chunk has
    topic=="class_routine". RagResult.retrieved_chunks and gate_metrics
    always reflect the original (unexpanded) set; only the evidence handed
    to generate_grounded_answer() is enriched.

    Phase 5.1 scope correction (Finding B): expansion is applied to the
    HIGHEST-RANKED routine anchor only. Expanding every routine anchor
    produced multi-anchor fan-out — a routine query returns ~10 headings
    from ~10 different timetables, each assembling its own siblings, which
    put 13-16 complete timetables in front of Gemini for a single-timetable
    question (evidence sets of 27-32 chunks). Every individual pairing was
    contamination-free, but the aggregate created an unnecessary
    disambiguation burden. Restricting to rank-1 assembles exactly the
    target timetable; lower-ranked routine chunks remain in evidence
    untouched. Pairing semantics themselves are unchanged.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.classifier.predict import predict_intent
from query_embed import create_query_embedding
from app.rag.generator import Citation, generate_grounded_answer
from app.database.tickets import create_ticket

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 1 gate configuration  (FROZEN — do not change)
# ---------------------------------------------------------------------------
GATE_THETA_D: float = 0.275
GATE_MIN_DISTINCT_PARENTS: int = 3

# ---------------------------------------------------------------------------
# OOS Recovery — Policy B vocabulary (Phase 1)
# ---------------------------------------------------------------------------
# Strong terms: a single match is sufficient to trigger a recovery attempt.
# These are specific campus-domain nouns that cannot plausibly appear in a
# truly out-of-scope query ("capital of France", "python sorting", etc.).
_OOS_RECOVERY_STRONG: frozenset[str] = frozenset({
    "bihu",
    "holiday", "vacation",
    "calendar",
    "hostel",
    "warden",
    "canteen", "cafeteria",
    "library",
    "auditorium",
    "classroom",
    "timetable", "routine",
    "scholarship", "stipend", "concession", "waiver",
    "fee", "fees",
    "admission", "application", "counselling",
    "a-connect", "aconnect",
    "wifi", "wi-fi",
    "bus", "shuttle", "transport",
    "attendance",
    "semester",
    "exam",
    "club", "clubs", "society", "societies",
    "mess",
    "capacity", "accommodation",
    "jalukbari",
    "nss", "ncc",
    "fest", "cultural",
})

# Context-only terms: ambiguous in isolation (e.g. "adtu alone" triggered
# suspicious HIGH retrieval in Phase 1B).  Two context-only terms together,
# or one context-only term co-occurring with a strong term, triggers recovery.
_OOS_RECOVERY_CONTEXT: frozenset[str] = frozenset({
    "adtu", "assam", "guwahati",
    "campus",
    "room", "block", "seat",
    "course", "programme",
    "professor", "faculty",
    "lab",
    "portal", "password",
    "payment",
    "class",
    "gym", "sports", "ground",
})

# Pre-compiled pattern — word-boundary, case-insensitive, longest-match order
_ALL_VOCAB: list[str] = sorted(
    _OOS_RECOVERY_STRONG | _OOS_RECOVERY_CONTEXT,
    key=len,
    reverse=True,  # longer terms matched first to avoid substring false-positives
)
_VOCAB_RE: re.Pattern[str] = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _ALL_VOCAB) + r")\b",
    re.IGNORECASE,
)

# Recovery retrieval is always against "facilities" only.
_OOS_RECOVERY_CATEGORY: str = "facilities"

# ---------------------------------------------------------------------------
# Holiday/Calendar retrieval-query normalization  (Phase 2)
# ---------------------------------------------------------------------------
# Terms whose presence in matched_terms triggers the "academic calendar" suffix.
_HOLIDAY_TRIGGER_TERMS: frozenset[str] = frozenset({"holiday", "bihu"})

# Suffix appended to the retrieval query — NOT to the user-visible query.
_HOLIDAY_RETRIEVAL_SUFFIX: str = "academic calendar"

# ---------------------------------------------------------------------------
# Scholarship monetary retrieval override  (Phase 3)
# ---------------------------------------------------------------------------
# Base gate: query must mention scholarship vocabulary at all.
_SCHOLARSHIP_TERMS: frozenset[str] = frozenset({"scholarship", "scholarships"})

# Monetary/percentage/marks/fee-discount signals. Presence of ANY of these
# alongside scholarship vocabulary indicates the user wants an amount, not
# an application/eligibility/portal answer.
_SCHOLARSHIP_MONETARY_SIGNAL_WORDS: frozenset[str] = frozenset({
    "amount", "percentage", "percent", "discount",
    "waiver", "concession", "marks", "kitna",
})

_SCHOLARSHIP_TERM_RE: re.Pattern[str] = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _SCHOLARSHIP_TERMS) + r")\b",
    re.IGNORECASE,
)
_SCHOLARSHIP_MONETARY_WORD_RE: re.Pattern[str] = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _SCHOLARSHIP_MONETARY_SIGNAL_WORDS) + r")\b",
    re.IGNORECASE,
)

# Retrieval category used for monetary scholarship queries, regardless of
# what the classifier / OOS recovery would otherwise select.
_SCHOLARSHIP_RETRIEVAL_CATEGORY: str = "fees"

# ---------------------------------------------------------------------------
# Scholarship retrieval-query normalization  (Phase 4)
# ---------------------------------------------------------------------------
# Suffix appended to the retrieval query — NOT to the user-visible query.
# Evidence-backed by the read-only normalization study: 0 regressions across
# the 2 baseline-passing monetary queries, 6/12 baseline-failing queries
# flipped from LOW to HIGH.
_SCHOLARSHIP_RETRIEVAL_SUFFIX: str = "scholarship semester fee"

# Duplicate-prevention check phrase (case-insensitive substring match).
_SCHOLARSHIP_SUFFIX_DUPLICATE_CHECK: str = "semester fee"

# ---------------------------------------------------------------------------
# V2 class-routine sibling expansion  (Phase 5)
# ---------------------------------------------------------------------------
_CLASS_ROUTINE_TOPIC: str = "class_routine"

# Metadata fields checked for the (primary) strong-match tier.
_ROUTINE_METADATA_FIELDS: tuple[str, ...] = ("program", "specialization", "semester", "section")

# chunk_id pattern for class_routine chunks: "v2_class_routine_<N>".
_ROUTINE_CHUNK_ID_RE: re.Pattern[str] = re.compile(r"^(v2_class_routine)_(\d+)$")

# Tightly bounded same-document chunk_index window for the fallback tier —
# used ONLY when the metadata tier finds zero matches. Kept at 1 (immediate
# neighbor only) per the audit's "tightly bounded" requirement; the corpus
# has legend/footnote fragments a few chunks away from real headings that a
# wider window would risk pulling in incorrectly.
_ROUTINE_ADJACENCY_WINDOW: int = 1

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    chunk_id: str
    document: str
    metadata: dict[str, Any]
    distance: float


@dataclass
class GateMetrics:
    distinct_parent_count: int
    dp_top3_mean_distance: float | None
    threshold: float = field(default=GATE_THETA_D)


@dataclass
class RagResult:
    """Structured output of the end-to-end orchestration pipeline."""
    status: str                 # "answered" | "out_of_scope" | "escalated" | "error"
    query: str
    intent: str
    answer: str | None
    citations: list[Citation] | None
    confidence_status: str | None
    ticket_id: str | None
    reason: str
    retrieved_chunks: list[RetrievedChunk]
    gate_metrics: GateMetrics | None
    
    @property
    def is_out_of_scope(self) -> bool:
        return self.status == "out_of_scope"


# ---------------------------------------------------------------------------
# Stage 1 confidence gate
# ---------------------------------------------------------------------------

def _distinct_parent_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen_parents: set[str] = set()
    deduped: list[RetrievedChunk] = []
    for chunk in chunks:
        parent_id: str = chunk.metadata.get("parent_chunk_id") or chunk.chunk_id
        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            deduped.append(chunk)
    return deduped


def assess_retrieval_confidence(
    chunks: list[RetrievedChunk],
) -> tuple[str, str, GateMetrics]:
    if not chunks:
        metrics = GateMetrics(distinct_parent_count=0, dp_top3_mean_distance=None)
        return "low", "No chunks retrieved.", metrics

    deduped = _distinct_parent_chunks(chunks)
    distinct_count = len(deduped)

    if distinct_count < GATE_MIN_DISTINCT_PARENTS:
        metrics = GateMetrics(distinct_parent_count=distinct_count, dp_top3_mean_distance=None)
        return (
            "low",
            f"Only {distinct_count} distinct parent chunk(s) retrieved.",
            metrics,
        )

    top3 = deduped[:GATE_MIN_DISTINCT_PARENTS]
    dp_top3_mean = sum(c.distance for c in top3) / len(top3)

    metrics = GateMetrics(
        distinct_parent_count=distinct_count,
        dp_top3_mean_distance=round(dp_top3_mean, 6),
    )

    if dp_top3_mean < GATE_THETA_D:
        return "high", f"dp-top3-mean={dp_top3_mean:.4f} < {GATE_THETA_D}", metrics
    else:
        return "low", f"dp-top3-mean={dp_top3_mean:.4f} >= {GATE_THETA_D}", metrics


# ---------------------------------------------------------------------------
# Main orchestration entry point
# ---------------------------------------------------------------------------

def run_rag_pipeline(
    query: str, 
    collection: chromadb.Collection,
    db_path: Path | str
) -> RagResult:
    """
    Execute the full end-to-end RAG pipeline.
    """
    if not query or not query.strip():
        return RagResult(
            status="error",
            query=query,
            intent="unknown",
            answer=None,
            citations=None,
            confidence_status=None,
            ticket_id=None,
            reason="Empty query provided.",
            retrieved_chunks=[],
            gate_metrics=None,
        )
    
    clean_query = query.strip()

    # 1. Classify intent
    try:
        intent: str = predict_intent(clean_query)
    except Exception as e:
        logger.error(f"Classifier error: {e}")
        return RagResult(
            status="error", query=clean_query, intent="unknown", answer=None, citations=None,
            confidence_status=None, ticket_id=None, reason="Internal error during classification.",
            retrieved_chunks=[], gate_metrics=None,
        )

    # 2. out_of_scope handling — Policy B recovery gate
    if intent == "out_of_scope":
        matched_terms, recovery_eligible = _campus_vocab_match(clean_query)
        if not recovery_eligible:
            # Hard OOS — no campus vocabulary, no embedding, no retrieval.
            return RagResult(
                status="out_of_scope",
                query=clean_query,
                intent=intent,
                answer="This query is outside the scope of AdtU Campus Copilot.",
                citations=[],
                confidence_status="not_applicable",
                ticket_id=None,
                reason="Query classified as out_of_scope.",
                retrieved_chunks=[],
                gate_metrics=None,
            )
        # Campus vocabulary matched — attempt ONE controlled recovery retrieval.
        logger.info(
            "OOS recovery triggered for query %r — matched terms: %s",
            clean_query, matched_terms,
        )
        return _run_oos_recovery(clean_query, matched_terms, collection, db_path)

    # 3. Query embedding & Retrieval
    # Retrieval category defaults to the classifier's intent. The scholarship
    # monetary override (Phase 3) may redirect it to "fees" — RagResult.intent
    # below still reports the frozen classifier's own `intent` unchanged.
    # is_monetary_scholarship is computed ONCE, from clean_query (the original
    # query), and reused for both the category override (Phase 3) and the
    # retrieval-text suffix (Phase 4) — never re-evaluated afterward.
    is_monetary_scholarship: bool = _is_monetary_scholarship_query(clean_query)
    retrieval_category: str = intent
    if is_monetary_scholarship:
        retrieval_category = _SCHOLARSHIP_RETRIEVAL_CATEGORY

    retrieval_query: str = _normalize_scholarship_retrieval_query(
        clean_query, is_monetary_scholarship
    )

    try:
        query_embedding: list[float] = create_query_embedding(retrieval_query)
        chroma_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            where={"category": retrieval_category},
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return RagResult(
            status="error", query=clean_query, intent=intent, answer=None, citations=None,
            confidence_status=None, ticket_id=None, reason="Internal error during retrieval.",
            retrieved_chunks=[], gate_metrics=None,
        )

    ids   = chroma_result.get("ids",       [[]])[0]
    docs  = chroma_result.get("documents", [[]])[0]
    metas = chroma_result.get("metadatas", [[]])[0]
    dists = chroma_result.get("distances", [[]])[0]

    chunks: list[RetrievedChunk] = [
        RetrievedChunk(chunk_id=i, document=d, metadata=m, distance=dist)
        for i, d, m, dist in zip(ids, docs, metas, dists, strict=True)
    ]

    # 4. Stage 1 confidence gate
    confidence_status, reason, gate_metrics = assess_retrieval_confidence(chunks)

    # 5. Handle LOW confidence (Escalate)
    if confidence_status == "low":
        try:
            ticket = create_ticket(db_path, clean_query, intent, source="stage_1_low")
            return RagResult(
                status="escalated", query=clean_query, intent=intent, answer=None, citations=None,
                confidence_status=confidence_status, ticket_id=ticket.ticket_id,
                reason="Escalated due to weak retrieval confidence.",
                retrieved_chunks=chunks, gate_metrics=gate_metrics,
            )
        except Exception as e:
            logger.error(f"Ticket creation error: {e}")
            return RagResult(
                status="error", query=clean_query, intent=intent, answer=None, citations=None,
                confidence_status=confidence_status, ticket_id=None,
                reason="Internal error during ticket creation.",
                retrieved_chunks=chunks, gate_metrics=gate_metrics,
            )

    # 6. Handle HIGH confidence (Stage 2 Grounding)
    # Sibling expansion (Phase 5) runs ONLY here — after the gate has
    # already decided HIGH from `chunks` — and only enriches the evidence
    # passed to Gemini. RagResult.retrieved_chunks/gate_metrics below still
    # report the original, unexpanded `chunks`.
    evidence_chunks = _expand_class_routine_evidence(chunks, collection)
    try:
        gen_result = generate_grounded_answer(clean_query, evidence_chunks)
    except Exception as e:
        logger.error(f"Gemini generation error: {e}")
        return RagResult(
            status="error", query=clean_query, intent=intent, answer=None, citations=None,
            confidence_status=confidence_status, ticket_id=None,
            reason="Internal error during grounded generation.",
            retrieved_chunks=chunks, gate_metrics=gate_metrics,
        )

    # 7. Check Stage 2 result
    if gen_result.status == "answered":
        return RagResult(
            status="answered", query=clean_query, intent=intent, answer=gen_result.answer,
            citations=gen_result.citations, confidence_status=confidence_status, ticket_id=None,
            reason="Successfully answered from evidence.",
            retrieved_chunks=chunks, gate_metrics=gate_metrics,
        )
    elif gen_result.status == "insufficient_evidence":
        try:
            ticket = create_ticket(db_path, clean_query, intent, source="stage_2_insufficient_evidence")
            return RagResult(
                status="escalated", query=clean_query, intent=intent, answer=None, citations=None,
                confidence_status=confidence_status, ticket_id=ticket.ticket_id,
                reason="Escalated due to insufficient evidence during generation.",
                retrieved_chunks=chunks, gate_metrics=gate_metrics,
            )
        except Exception as e:
            logger.error(f"Ticket creation error: {e}")
            return RagResult(
                status="error", query=clean_query, intent=intent, answer=None, citations=None,
                confidence_status=confidence_status, ticket_id=None,
                reason="Internal error during ticket creation.",
                retrieved_chunks=chunks, gate_metrics=gate_metrics,
            )
    else:
        # Fallback for unexpected status
        return RagResult(
            status="error", query=clean_query, intent=intent, answer=None, citations=None,
            confidence_status=confidence_status, ticket_id=None,
            reason=f"Unexpected generation status: {gen_result.status}",
            retrieved_chunks=chunks, gate_metrics=gate_metrics,
        )


# ---------------------------------------------------------------------------
# OOS Recovery helpers  (Policy B)
# ---------------------------------------------------------------------------

def _campus_vocab_match(query: str) -> tuple[list[str], bool]:
    """Check whether *query* matches the campus recovery vocabulary.

    Returns:
        (matched_terms, recovery_eligible)

        matched_terms     — sorted list of matched term strings (lower-cased),
                            useful for diagnostics / test assertions.
        recovery_eligible — True when the match set satisfies the trigger rule:
                            (a) ≥1 strong term, OR
                            (b) ≥2 context-only terms, OR
                            (c) ≥1 context-only term AND ≥1 strong term.
                            In practice (a) and (c) are the same condition;
                            they are spelled out separately for clarity.

    The function is deterministic, case-insensitive, and uses word-boundary
    matching so that e.g. "programme" does not match inside "programmer".
    """
    matches: set[str] = {
        m.group(0).lower() for m in _VOCAB_RE.finditer(query)
    }

    strong_hits  = matches & {t.lower() for t in _OOS_RECOVERY_STRONG}
    context_hits = matches & {t.lower() for t in _OOS_RECOVERY_CONTEXT}

    recovery_eligible: bool = (
        len(strong_hits) >= 1           # (a)/(c): any strong term is sufficient
        or len(context_hits) >= 2       # (b): two context-only terms together
    )

    return sorted(matches), recovery_eligible


def _normalize_oos_retrieval_query(query: str, matched_terms: list[str]) -> str:
    """Return the retrieval query to use for the OOS recovery embedding call.

    Rule (deterministic, pure, no I/O):
        If *matched_terms* contains "holiday" or "bihu" AND
        the phrase "academic calendar" is not already present in the query
        (case-insensitive), append " academic calendar".
        Otherwise return *query* unchanged.

    This function MUST NOT be called outside of _run_oos_recovery.
    The returned string is used only for create_query_embedding(); it is
    never shown to the user and never stored in RagResult.
    """
    matched_set = {t.lower() for t in matched_terms}
    if matched_set & _HOLIDAY_TRIGGER_TERMS:
        if "academic calendar" not in query.lower():
            return query + " academic calendar"
    return query


def _is_monetary_scholarship_query(query: str) -> bool:
    """Return True iff *query* asks about a scholarship AMOUNT/PERCENTAGE/
    DISCOUNT rather than the application process, eligibility, or portal.

    This decides RETRIEVAL CATEGORY only — it never touches classifier
    output, RagResult.intent, or the embedding text. Pure, deterministic,
    no I/O.

    Rule:
        query mentions scholarship vocabulary ("scholarship"/"scholarships")
        AND at least one monetary/percentage/marks/fee-discount signal:
            a monetary signal word (amount, percentage, percent, discount,
            waiver, concession, marks, kitna) OR a literal "%" character.

    Deliberately excluded (by omission — no monetary signal present):
        "NSP scholarship portal", "apply for scholarship",
        "scholarship application", "scholarship eligibility",
        "who can apply", "scholarship documents"
    These keep whatever category the classifier / OOS recovery already
    selected, since routing them to "fees" would not fix anything and
    risks masking a different (application-process) intent.
    """
    if not _SCHOLARSHIP_TERM_RE.search(query):
        return False
    return bool(_SCHOLARSHIP_MONETARY_WORD_RE.search(query)) or "%" in query


def _normalize_scholarship_retrieval_query(query: str, is_monetary: bool) -> str:
    """Return the retrieval text to embed for a (possibly) scholarship query.

    Rule (deterministic, pure, no I/O):
        If *is_monetary* is True AND the phrase "semester fee" is not already
        present in *query* (case-insensitive), append
        " scholarship semester fee". Otherwise return *query* unchanged.

    IMPORTANT: *is_monetary* MUST be computed exactly once, by the caller,
    from the ORIGINAL query via _is_monetary_scholarship_query() — BEFORE
    any normalization (this function's or any other, e.g. the Bihu/holiday
    one) has touched the text. This function itself never calls
    _is_monetary_scholarship_query and never re-evaluates it on the
    (possibly already-normalized) *query* argument, so a suffix appended
    upstream (or by this function) can never retroactively flip the
    decision that produced it.

    The returned string is used only for create_query_embedding(); it is
    never shown to the user and never stored in RagResult.query.
    """
    if is_monetary and _SCHOLARSHIP_SUFFIX_DUPLICATE_CHECK not in query.lower():
        return query + " " + _SCHOLARSHIP_RETRIEVAL_SUFFIX
    return query


def _parse_routine_chunk_id(chunk_id: str) -> tuple[str, int] | None:
    """Parse a class_routine chunk_id into (group_prefix, chunk_index).

    Returns None if *chunk_id* doesn't match "v2_class_routine_<N>".

    Chroma's own `document`/`chunk_index` metadata fields are not populated
    for this collection (verified empty at runtime), so chunk_id parsing is
    the only reliable source of adjacency information. Pure, no I/O.
    """
    match = _ROUTINE_CHUNK_ID_RE.match(chunk_id)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _normalized_routine_field(metadata: dict[str, Any], field_name: str) -> str:
    """Return metadata[field_name], stripped, treating the chunker's generic
    document-stem fallback as empty.

    chunk_v2_data.py falls back to `section_title = ... or file_path.stem`
    when no real H1/H2/H3 header was parsed, so many class_routine chunks
    carry `section == "class_routine"` -- a placeholder, not a real section
    label (A/B/C/D...). Treating it as a real value would let chunks match
    "solely because they are both class_routine" through the `section`
    field specifically, which is exactly what strong-tier matching must
    never do. Pure, no I/O.
    """
    value = (metadata.get(field_name) or "").strip()
    if field_name == "section" and value == _CLASS_ROUTINE_TOPIC:
        return ""
    return value


def _routine_metadata_overlap_matches(
    anchor_metadata: dict[str, Any], candidate_metadata: dict[str, Any]
) -> bool:
    """Strong (primary-tier) match: True iff every populated routine field on
    *anchor_metadata* matches *candidate_metadata* exactly, and at least one
    field was actually checked (so chunks are never paired on topic alone).
    Pure, deterministic, no I/O.
    """
    matched_any = False
    for field_name in _ROUTINE_METADATA_FIELDS:
        anchor_value = _normalized_routine_field(anchor_metadata, field_name)
        if not anchor_value:
            continue  # not required when empty on the anchor
        candidate_value = _normalized_routine_field(candidate_metadata, field_name)
        if anchor_value != candidate_value:
            return False
        matched_any = True
    return matched_any


def _routine_metadata_conflicts(
    anchor_metadata: dict[str, Any], candidate_metadata: dict[str, Any]
) -> bool:
    """True iff any routine field populated on BOTH sides disagrees.
    Used only by the adjacency fallback tier, as a cross-section/
    cross-semester contamination guard. Pure, deterministic, no I/O.
    """
    for field_name in _ROUTINE_METADATA_FIELDS:
        anchor_value = _normalized_routine_field(anchor_metadata, field_name)
        candidate_value = _normalized_routine_field(candidate_metadata, field_name)
        if anchor_value and candidate_value and anchor_value != candidate_value:
            return True
    return False


def _routine_has_any_metadata(metadata: dict[str, Any]) -> bool:
    """True iff at least one routine field is populated (normalized). Pure, no I/O."""
    return any(_normalized_routine_field(metadata, f) for f in _ROUTINE_METADATA_FIELDS)


def _find_class_routine_sibling_ids(
    anchor_chunk_id: str,
    anchor_metadata: dict[str, Any],
    candidates: list[tuple[str, dict[str, Any]]],
) -> list[str]:
    """Return the chunk_ids of *anchor*'s verified class_routine siblings.

    Pure, deterministic, no Chroma or Gemini access — *candidates* is a
    plain (chunk_id, metadata) list supplied by the caller (e.g. every other
    class_routine chunk in the corpus), so this is fully unit-testable with
    hand-built dicts.

    Rule:
        1. If anchor_metadata["topic"] != "class_routine", return [].
        2. Metadata tier: candidates sharing topic=="class_routine" where
           _routine_metadata_overlap_matches(anchor, candidate) is True.
           If this finds anything, return it — the fallback tier is never
           consulted (a strong metadata match is trusted exclusively).
        3. Adjacency fallback tier (only reached when tier 2 finds zero
           matches — typically because the anchor itself has no populated
           program/specialization/semester/section metadata): candidates
           parsed from the same "v2_class_routine_<N>" id group, within
           +-_ROUTINE_ADJACENCY_WINDOW of the anchor's index, with the
           OPPOSITE is_table value (a heading looks for its table and vice
           versa), without any conflicting populated field
           (_routine_metadata_conflicts) — AND, if the anchor itself has NO
           populated routine metadata at all, the candidate must ALSO have
           none. Without this last guard, an empty-metadata "orphan" chunk
           sitting immediately after a fully-identified, unrelated section
           (e.g. a generic legend fragment right after another section's
           table) would otherwise incorrectly inherit that unrelated
           section's real data — confirmed as a real failure mode against
           the live corpus (v2_class_routine_36, empty metadata, sits right
           after v2_class_routine_35, a fully-populated but unrelated
           section) and fixed by this guard.

    Never performs embedding-based or corpus-wide nearest-neighbor search —
    both tiers are bounded, exact-match operations over a caller-supplied
    candidate list.
    """
    if anchor_metadata.get("topic") != _CLASS_ROUTINE_TOPIC:
        return []

    metadata_matches: list[str] = []
    for candidate_id, candidate_metadata in candidates:
        if candidate_id == anchor_chunk_id:
            continue
        if candidate_metadata.get("topic") != _CLASS_ROUTINE_TOPIC:
            continue
        if _routine_metadata_overlap_matches(anchor_metadata, candidate_metadata):
            metadata_matches.append(candidate_id)

    if metadata_matches:
        return sorted(metadata_matches)

    anchor_parsed = _parse_routine_chunk_id(anchor_chunk_id)
    if anchor_parsed is None:
        return []
    anchor_group, anchor_index = anchor_parsed

    fallback_matches: list[str] = []
    for candidate_id, candidate_metadata in candidates:
        if candidate_id == anchor_chunk_id:
            continue
        if candidate_metadata.get("topic") != _CLASS_ROUTINE_TOPIC:
            continue
        candidate_parsed = _parse_routine_chunk_id(candidate_id)
        if candidate_parsed is None:
            continue
        candidate_group, candidate_index = candidate_parsed
        if candidate_group != anchor_group:
            continue
        if abs(candidate_index - anchor_index) > _ROUTINE_ADJACENCY_WINDOW:
            continue
        if candidate_metadata.get("is_table") == anchor_metadata.get("is_table"):
            continue
        if _routine_metadata_conflicts(anchor_metadata, candidate_metadata):
            continue
        if not _routine_has_any_metadata(anchor_metadata) and _routine_has_any_metadata(candidate_metadata):
            continue
        fallback_matches.append(candidate_id)

    return sorted(fallback_matches)


def _expand_class_routine_evidence(
    chunks: list[RetrievedChunk],
    collection: chromadb.Collection,
) -> list[RetrievedChunk]:
    """Return an evidence list for Stage 2, enriched with the verified
    class_routine siblings of the HIGHEST-RANKED routine anchor in *chunks*.

    Scope rule (Phase 5.1, Finding B): only the rank-1 routine anchor is
    expanded. Lower-ranked routine chunks stay in the evidence set exactly
    as retrieved, but contribute no siblings of their own — see the inline
    comment at the selection site for the fan-out rationale and the
    rank-order invariant this relies on.

    Architectural rule: this function MUST be called AFTER the Stage 1 gate
    has already been computed from *chunks* — it never mutates *chunks* and
    never feeds back into gate_metrics. It returns a NEW list; the original
    is left untouched so callers can still report RagResult.retrieved_chunks
    / gate_metrics from the original set.

    Performs at most one collection.get(where={"topic": "class_routine"})
    metadata-only fetch (no embedding call) — and only if at least one chunk
    in *chunks* is itself topic=="class_routine". Sibling chunks retain
    their real chunk_id/metadata/document text (never fabricated); already-
    present chunk_ids are never re-added (no duplicates).
    """
    routine_anchors = [c for c in chunks if c.metadata.get("topic") == _CLASS_ROUTINE_TOPIC]
    if not routine_anchors:
        return chunks

    # Phase 5.1 (Finding B): expand ONLY the highest-ranked routine anchor.
    # *chunks* is built by zipping Chroma's ids/documents/metadatas/distances
    # in result order, and Chroma returns results sorted by ascending distance;
    # neither assess_retrieval_confidence() nor _distinct_parent_chunks()
    # mutates or reorders it (all four properties verified against the live
    # collection). So the FIRST routine chunk in list order is provably the
    # nearest/rank-1 routine anchor.
    #
    # Expanding every routine anchor caused multi-anchor fan-out: a routine
    # query returns 10 headings belonging to 10 DIFFERENT timetables, each of
    # which then assembled its own siblings, putting 13-16 complete timetables
    # in front of Gemini for a question about one. Every individual pairing was
    # correct (zero contamination), but the aggregate forced the model to
    # disambiguate between many unrelated sections. Restricting expansion to
    # the rank-1 anchor assembles exactly the target timetable and leaves every
    # lower-ranked routine chunk in the evidence set untouched, exactly as it
    # appeared pre-Phase-5.
    primary_anchor = routine_anchors[0]

    routine_pool = collection.get(
        where={"topic": _CLASS_ROUTINE_TOPIC},
        include=["documents", "metadatas"],
    )
    pool_ids: list[str] = routine_pool.get("ids", [])
    pool_docs: list[str] = routine_pool.get("documents", [])
    pool_metas: list[dict[str, Any]] = routine_pool.get("metadatas", [])
    candidates: list[tuple[str, dict[str, Any]]] = list(zip(pool_ids, pool_metas))
    pool_lookup: dict[str, tuple[str, dict[str, Any]]] = {
        cid: (doc, meta) for cid, doc, meta in zip(pool_ids, pool_docs, pool_metas)
    }

    existing_ids = {c.chunk_id for c in chunks}
    expanded: list[RetrievedChunk] = list(chunks)
    added_ids: set[str] = set()

    sibling_ids = _find_class_routine_sibling_ids(
        primary_anchor.chunk_id, primary_anchor.metadata, candidates
    )
    for sibling_id in sibling_ids:
        if sibling_id in existing_ids or sibling_id in added_ids:
            continue
        if sibling_id not in pool_lookup:
            continue
        sibling_doc, sibling_meta = pool_lookup[sibling_id]
        expanded.append(
            RetrievedChunk(
                chunk_id=sibling_id,
                document=sibling_doc,
                metadata=sibling_meta,
                # distance is inherited from the anchor for dataclass
                # completeness only. It is NEVER used for gate
                # computation — the gate has already run, on `chunks`,
                # strictly before this function is ever called.
                distance=primary_anchor.distance,
            )
        )
        added_ids.add(sibling_id)

    return expanded


def _run_oos_recovery(
    query: str,
    matched_terms: list[str],
    collection: chromadb.Collection,
    db_path: Path | str,
) -> RagResult:
    """Execute the single controlled recovery retrieval for an OOS-classified query.

    This function mirrors the normal in-scope pipeline (steps 3-7) with six
    differences:
    - The retrieval category defaults to *facilities* (not the classifier
      intent, which is always "out_of_scope" here).
    - Ticket sources use the "oos_recovery_" prefix so operators can distinguish
      recovery escalations from normal-flow escalations.
    - For holiday/Bihu queries, _normalize_oos_retrieval_query() appends
      "academic calendar" to the embedding call only (Phase 2 normalization).
    - For monetary scholarship queries, the recovery category is redirected
      to "fees" instead of "facilities" (Phase 3 override). All other OOS
      recovery queries are unaffected and still target "facilities".
    - For the same monetary scholarship queries, _normalize_scholarship_
      retrieval_query() also appends "scholarship semester fee" to the
      embedding call only (Phase 4 normalization), stacked on top of any
      Phase 2 holiday normalization already applied.
    - If the retrieved set contains a class_routine chunk, the Stage 2
      evidence is enriched with its verified sibling(s) (Phase 5), after
      the gate has already decided on the unexpanded `chunks`.

    The frozen classifier output (intent == "out_of_scope") is preserved in
    the returned RagResult.intent field.  No new intent class is introduced.
    The RagResult.status follows the same contract as the normal pipeline:
    "answered" | "escalated" | "error".
    """
    # is_monetary_scholarship is computed ONCE, from the original `query`,
    # and reused for both the category override (Phase 3) and the retrieval
    # text suffix (Phase 4) — never re-evaluated on normalized text.
    is_monetary_scholarship = _is_monetary_scholarship_query(query)
    recovery_category = _OOS_RECOVERY_CATEGORY  # "facilities" (default)
    if is_monetary_scholarship:
        recovery_category = _SCHOLARSHIP_RETRIEVAL_CATEGORY  # "fees"

    # Step R1 — embedding + retrieval
    # Apply holiday/calendar normalization first (Phase 2), then scholarship
    # monetary normalization (Phase 4) on top. The original `query` is kept
    # intact for all user-visible fields.
    retrieval_query: str = _normalize_oos_retrieval_query(query, matched_terms)
    retrieval_query = _normalize_scholarship_retrieval_query(
        retrieval_query, is_monetary_scholarship
    )
    if retrieval_query != query:
        logger.info(
            "Retrieval-text normalization applied: %r → %r",
            query, retrieval_query,
        )
    try:
        query_embedding: list[float] = create_query_embedding(retrieval_query)
        chroma_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            where={"category": recovery_category},
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("OOS recovery retrieval error: %s", exc)
        return RagResult(
            status="error",
            query=query,
            intent="out_of_scope",
            answer=None,
            citations=None,
            confidence_status=None,
            ticket_id=None,
            reason="Internal error during OOS recovery retrieval.",
            retrieved_chunks=[],
            gate_metrics=None,
        )

    ids   = chroma_result.get("ids",       [[]])[0]
    docs  = chroma_result.get("documents", [[]])[0]
    metas = chroma_result.get("metadatas", [[]])[0]
    dists = chroma_result.get("distances", [[]])[0]

    chunks: list[RetrievedChunk] = [
        RetrievedChunk(chunk_id=i, document=d, metadata=m, distance=dist)
        for i, d, m, dist in zip(ids, docs, metas, dists, strict=True)
    ]

    # Step R2 — Stage 1 gate (identical logic, same constants)
    confidence_status, reason, gate_metrics = assess_retrieval_confidence(chunks)

    # Step R3 — LOW confidence → SQLite escalation
    if confidence_status == "low":
        try:
            ticket = create_ticket(
                db_path, query, "out_of_scope",
                source="oos_recovery_stage1_low",
            )
            return RagResult(
                status="escalated",
                query=query,
                intent="out_of_scope",
                answer=None,
                citations=None,
                confidence_status=confidence_status,
                ticket_id=ticket.ticket_id,
                reason=(
                    f"OOS recovery: facilities retrieval low confidence ({reason}). "
                    f"Matched terms: {matched_terms}."
                ),
                retrieved_chunks=chunks,
                gate_metrics=gate_metrics,
            )
        except Exception as exc:
            logger.error("OOS recovery ticket creation error: %s", exc)
            return RagResult(
                status="error",
                query=query,
                intent="out_of_scope",
                answer=None,
                citations=None,
                confidence_status=confidence_status,
                ticket_id=None,
                reason="Internal error during OOS recovery ticket creation.",
                retrieved_chunks=chunks,
                gate_metrics=gate_metrics,
            )

    # Step R4 — HIGH confidence → Stage 2 Gemini
    # Sibling expansion (Phase 5) enriches evidence only, after the gate has
    # already decided HIGH from `chunks`; gate_metrics above are unaffected.
    evidence_chunks = _expand_class_routine_evidence(chunks, collection)
    try:
        gen_result = generate_grounded_answer(query, evidence_chunks)
    except Exception as exc:
        logger.error("OOS recovery Gemini generation error: %s", exc)
        return RagResult(
            status="error",
            query=query,
            intent="out_of_scope",
            answer=None,
            citations=None,
            confidence_status=confidence_status,
            ticket_id=None,
            reason="Internal error during OOS recovery generation.",
            retrieved_chunks=chunks,
            gate_metrics=gate_metrics,
        )

    # Step R5 — Stage 2 result
    if gen_result.status == "answered":
        return RagResult(
            status="answered",
            query=query,
            intent="out_of_scope",
            answer=gen_result.answer,
            citations=gen_result.citations,
            confidence_status=confidence_status,
            ticket_id=None,
            reason=(
                f"OOS recovery answered via facilities retrieval. "
                f"Matched terms: {matched_terms}."
            ),
            retrieved_chunks=chunks,
            gate_metrics=gate_metrics,
        )

    # insufficient_evidence (or any unexpected status) → SQLite escalation
    try:
        ticket = create_ticket(
            db_path, query, "out_of_scope",
            source="oos_recovery_stage2_insufficient_evidence",
        )
        return RagResult(
            status="escalated",
            query=query,
            intent="out_of_scope",
            answer=None,
            citations=None,
            confidence_status=confidence_status,
            ticket_id=ticket.ticket_id,
            reason=(
                f"OOS recovery: Stage 2 returned insufficient evidence. "
                f"Matched terms: {matched_terms}."
            ),
            retrieved_chunks=chunks,
            gate_metrics=gate_metrics,
        )
    except Exception as exc:
        logger.error("OOS recovery Stage 2 ticket creation error: %s", exc)
        return RagResult(
            status="error",
            query=query,
            intent="out_of_scope",
            answer=None,
            citations=None,
            confidence_status=confidence_status,
            ticket_id=None,
            reason="Internal error during OOS recovery ticket creation.",
            retrieved_chunks=chunks,
            gate_metrics=gate_metrics,
        )
