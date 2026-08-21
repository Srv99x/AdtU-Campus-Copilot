"""
AdtU Campus Copilot — Ticket Escalation Subsystem
Manages persistence for unanswered or unverified queries.
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


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

    conn = _get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO tickets (
                    ticket_id, query, predicted_intent, status, created_at, source, user_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ticket_id, query, predicted_intent, status, created_at, source, user_metadata)
            )
    finally:
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
