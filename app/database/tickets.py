"""
AdtU Campus Copilot — Ticket Escalation Subsystem
Manages persistence for unanswered or unverified queries.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Bounded retry/backoff for transient SQLite lock contention on ticket
# creation (M4). Local single-file SQLite under WAL mode only ever blocks a
# concurrent writer for the few milliseconds another write transaction holds
# the lock, so a handful of short, fast-growing waits is enough to ride out
# real contention without making a failed escalation feel slow, while still
# failing fast (not hanging indefinitely) if the lock never clears.
# ---------------------------------------------------------------------------
_MAX_LOCK_RETRIES: int = 3
_LOCK_RETRY_BASE_DELAY_SECONDS: float = 0.05  # 50ms, 100ms, 200ms

# Substrings SQLite uses for transient lock/busy conditions (SQLITE_BUSY /
# SQLITE_LOCKED), as raised via sqlite3.OperationalError. Any other
# OperationalError (e.g. malformed SQL, missing table, disk corruption) does
# not match and is never retried.
_TRANSIENT_LOCK_MARKERS: tuple[str, ...] = ("database is locked", "database is busy")


def _is_transient_lock_error(exc: sqlite3.OperationalError) -> bool:
    """True iff *exc* is a transient SQLite lock/busy error that is safe to retry."""
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_LOCK_MARKERS)


def _lock_retry_delay_seconds(attempt: int) -> float:
    """Deterministic bounded backoff for retry *attempt* (1-indexed)."""
    return _LOCK_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))


@dataclass
class Ticket:
    ticket_id: str
    query: str
    predicted_intent: str
    status: str
    created_at: str
    source: str
    user_metadata: str | None


def _get_connection(db_path: Path | str) -> sqlite3.Connection:
    """Get a SQLite connection configured for our access pattern."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys and WAL for concurrency
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def initialize_database(db_path: Path | str) -> None:
    """Create the tickets schema if it does not exist."""
    schema = """
    CREATE TABLE IF NOT EXISTS tickets (
        ticket_id TEXT PRIMARY KEY,
        query TEXT NOT NULL,
        predicted_intent TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'open',
        created_at TEXT NOT NULL,
        source TEXT NOT NULL,
        user_metadata TEXT
    );
    """
    conn = _get_connection(db_path)
    try:
        with conn:
            conn.execute(schema)
    finally:
        conn.close()


def create_ticket(
    db_path: Path | str,
    query: str,
    predicted_intent: str,
    source: str,
    user_metadata: str | None = None
) -> Ticket:
    """
    Create a new escalation ticket.
    
    Args:
        db_path: Path to the SQLite database.
        query: The raw query text.
        predicted_intent: The classifier intent (e.g. 'admissions', 'out_of_scope').
        source: The reason for escalation (e.g., 'stage_1_low', 'insufficient_evidence').
        user_metadata: Optional JSON string of user context.

    A transient "database is locked"/"database is busy" error during the
    INSERT is retried up to _MAX_LOCK_RETRIES times with a short, bounded
    backoff. Any other error (malformed SQL, constraint violation, or any
    other exception) is raised immediately on the first attempt.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if not predicted_intent.strip():
        raise ValueError("Predicted intent cannot be empty.")
    if not source.strip():
        raise ValueError("Source cannot be empty.")

    ticket_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    status = "open"

    attempt = 0
    while True:
        conn: sqlite3.Connection | None = None
        try:
            conn = _get_connection(db_path)
            with conn:
                conn.execute(
                    """
                    INSERT INTO tickets (
                        ticket_id, query, predicted_intent, status, created_at, source, user_metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (ticket_id, query, predicted_intent, status, created_at, source, user_metadata)
                )
            break
        except sqlite3.OperationalError as exc:
            if attempt >= _MAX_LOCK_RETRIES or not _is_transient_lock_error(exc):
                raise
            attempt += 1
            time.sleep(_lock_retry_delay_seconds(attempt))
        finally:
            if conn is not None:
                conn.close()

    return Ticket(
        ticket_id=ticket_id,
        query=query,
        predicted_intent=predicted_intent,
        status=status,
        created_at=created_at,
        source=source,
        user_metadata=user_metadata
    )


def get_ticket(db_path: Path | str, ticket_id: str) -> Ticket | None:
    """Retrieve a ticket by its ID, or None if not found."""
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?",
            (ticket_id,)
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return None

    return Ticket(
        ticket_id=row["ticket_id"],
        query=row["query"],
        predicted_intent=row["predicted_intent"],
        status=row["status"],
        created_at=row["created_at"],
        source=row["source"],
        user_metadata=row["user_metadata"]
    )


def list_tickets(db_path: Path | str, status: str | None = None) -> list[Ticket]:
    """
    List all tickets, optionally filtered by status.
    Ordered by creation date descending (newest first).
    """
    conn = _get_connection(db_path)
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE status = ? ORDER BY created_at DESC",
                (status,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tickets ORDER BY created_at DESC"
            ).fetchall()
    finally:
        conn.close()

    return [
        Ticket(
            ticket_id=r["ticket_id"],
            query=r["query"],
            predicted_intent=r["predicted_intent"],
            status=r["status"],
            created_at=r["created_at"],
            source=r["source"],
            user_metadata=r["user_metadata"]
        )
        for r in rows
    ]
