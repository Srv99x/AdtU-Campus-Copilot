"""
Tests for app/database/tickets.py — SQLite ticket persistence subsystem.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.database.tickets import (
    Ticket,
    VALID_TICKET_STATUSES,
    create_ticket,
    get_ticket,
    initialize_database,
    list_tickets,
    update_ticket_status,
    _MAX_LOCK_RETRIES,
)


def _mock_connection(execute_effects: list) -> MagicMock:
    """Build a MagicMock standing in for a sqlite3.Connection.

    - `with conn:` behaves like the real context manager: __exit__ never
      suppresses an exception raised inside the block (a bare MagicMock's
      auto-generated __exit__ return value is truthy, which would silently
      swallow the OperationalError instead of letting it propagate).
    - conn.execute(...) raises/returns according to `execute_effects` in
      order (mirrors unittest.mock's own `side_effect` list semantics).
    """
    conn = MagicMock(spec=sqlite3.Connection)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    conn.execute = MagicMock(side_effect=execute_effects)
    return conn


class TestTicketsDatabase(unittest.TestCase):

    def setUp(self) -> None:
        # Create a fresh temporary directory for each test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_escalation.db"
        # Initialize schema
        initialize_database(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_initialize_database_idempotent(self) -> None:
        """Calling initialize_database multiple times should be safe."""
        initialize_database(self.db_path)
        initialize_database(self.db_path)
        # Verify table exists
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets'")
            self.assertIsNotNone(cursor.fetchone())
        conn.close()

    def test_create_and_get_ticket(self) -> None:
        """A created ticket can be retrieved perfectly."""
        ticket = create_ticket(
            db_path=self.db_path,
            query="When is the library open?",
            predicted_intent="facilities",
            source="stage_1_low",
            user_metadata='{"user_id": 123}'
        )

        self.assertIsInstance(ticket.ticket_id, str)
        self.assertEqual(ticket.query, "When is the library open?")
        self.assertEqual(ticket.predicted_intent, "facilities")
        self.assertEqual(ticket.status, "open")
        self.assertEqual(ticket.source, "stage_1_low")
        self.assertEqual(ticket.user_metadata, '{"user_id": 123}')

        retrieved = get_ticket(self.db_path, ticket.ticket_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.ticket_id, ticket.ticket_id)
        self.assertEqual(retrieved.query, ticket.query)

    def test_get_nonexistent_ticket_returns_none(self) -> None:
        retrieved = get_ticket(self.db_path, "does-not-exist")
        self.assertIsNone(retrieved)

    def test_list_tickets(self) -> None:
        """list_tickets should return all tickets ordered by creation date descending."""
        t1 = create_ticket(self.db_path, "Q1", "admissions", "source1")
        t2 = create_ticket(self.db_path, "Q2", "fees", "source2")
        
        # Manually backdate t1 to ensure ordering is tested
        # SQLite stores timestamps as text; we manipulate it to test order
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE tickets SET created_at = '2020-01-01T00:00:00Z' WHERE ticket_id = ?", (t1.ticket_id,))
            conn.commit()
        conn.close()

        tickets = list_tickets(self.db_path)
        self.assertEqual(len(tickets), 2)
        # Newest first
        self.assertEqual(tickets[0].ticket_id, t2.ticket_id)
        self.assertEqual(tickets[1].ticket_id, t1.ticket_id)

    def test_list_tickets_filtered_by_status(self) -> None:
        """list_tickets should correctly filter by status."""
        t1 = create_ticket(self.db_path, "Q1", "admissions", "source1")
        t2 = create_ticket(self.db_path, "Q2", "fees", "source2")
        
        # Change status of t1
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE tickets SET status = 'closed' WHERE ticket_id = ?", (t1.ticket_id,))
            conn.commit()
        conn.close()

        open_tickets = list_tickets(self.db_path, status="open")
        self.assertEqual(len(open_tickets), 1)
        self.assertEqual(open_tickets[0].ticket_id, t2.ticket_id)

        closed_tickets = list_tickets(self.db_path, status="closed")
        self.assertEqual(len(closed_tickets), 1)
        self.assertEqual(closed_tickets[0].ticket_id, t1.ticket_id)

    def test_malformed_input_raises_value_error(self) -> None:
        """Creating tickets with empty required fields should fail."""
        with self.assertRaisesRegex(ValueError, "Query cannot be empty"):
            create_ticket(self.db_path, "   ", "admissions", "source1")
            
        with self.assertRaisesRegex(ValueError, "intent cannot be empty"):
            create_ticket(self.db_path, "valid", "", "source1")
            
        with self.assertRaisesRegex(ValueError, "Source cannot be empty"):
            create_ticket(self.db_path, "valid", "admissions", "  ")

    def test_multiple_tickets_get_unique_ids(self) -> None:
        """Repeated create calls should generate unique ticket IDs."""
        t1 = create_ticket(self.db_path, "Q1", "admissions", "source1")
        t2 = create_ticket(self.db_path, "Q2", "fees", "source2")
        self.assertNotEqual(t1.ticket_id, t2.ticket_id)

    @patch("app.database.tickets.time.sleep")
    def test_normal_creation_succeeds_without_retry(self, mock_sleep: MagicMock) -> None:
        """Happy path against the real temp DB: retry machinery is a no-op."""
        ticket = create_ticket(self.db_path, "Q", "admissions", "source1")
        self.assertEqual(ticket.status, "open")
        mock_sleep.assert_not_called()

    @patch("app.database.tickets._get_connection")
    @patch("app.database.tickets.time.sleep")
    def test_database_locked_once_then_succeeds(
        self, mock_sleep: MagicMock, mock_get_connection: MagicMock
    ) -> None:
        locked_conn = _mock_connection([sqlite3.OperationalError("database is locked")])
        ok_conn = _mock_connection([None])
        mock_get_connection.side_effect = [locked_conn, ok_conn]

        ticket = create_ticket(self.db_path, "Q", "admissions", "source1")

        self.assertIsInstance(ticket, Ticket)
        self.assertEqual(ticket.status, "open")
        self.assertEqual(mock_get_connection.call_count, 2)
        mock_sleep.assert_called_once()
        locked_conn.close.assert_called_once()
        ok_conn.close.assert_called_once()

    @patch("app.database.tickets._get_connection")
    @patch("app.database.tickets.time.sleep")
    def test_database_busy_once_then_succeeds(
        self, mock_sleep: MagicMock, mock_get_connection: MagicMock
    ) -> None:
        busy_conn = _mock_connection([sqlite3.OperationalError("database is busy")])
        ok_conn = _mock_connection([None])
        mock_get_connection.side_effect = [busy_conn, ok_conn]

        ticket = create_ticket(self.db_path, "Q", "admissions", "source1")

        self.assertIsInstance(ticket, Ticket)
        self.assertEqual(mock_get_connection.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("app.database.tickets._get_connection")
    @patch("app.database.tickets.time.sleep")
    def test_retries_stop_after_max_and_raises_final_lock_error(
        self, mock_sleep: MagicMock, mock_get_connection: MagicMock
    ) -> None:
        # Provide more locked connections than could ever be consumed, so a
        # bug that retries unboundedly fails loudly (StopIteration) rather
        # than hanging or silently passing.
        conns = [
            _mock_connection([sqlite3.OperationalError("database is locked")])
            for _ in range(_MAX_LOCK_RETRIES + 5)
        ]
        mock_get_connection.side_effect = conns

        with self.assertRaises(sqlite3.OperationalError):
            create_ticket(self.db_path, "Q", "admissions", "source1")

        # 1 initial attempt + _MAX_LOCK_RETRIES retries
        self.assertEqual(mock_get_connection.call_count, _MAX_LOCK_RETRIES + 1)
        self.assertEqual(mock_sleep.call_count, _MAX_LOCK_RETRIES)
        for conn in conns[: _MAX_LOCK_RETRIES + 1]:
            conn.close.assert_called_once()

    @patch("app.database.tickets._get_connection")
    @patch("app.database.tickets.time.sleep")
    def test_non_transient_operational_error_is_not_retried(
        self, mock_sleep: MagicMock, mock_get_connection: MagicMock
    ) -> None:
        bad_conn = _mock_connection([sqlite3.OperationalError("no such table: tickets")])
        mock_get_connection.return_value = bad_conn

        with self.assertRaises(sqlite3.OperationalError):
            create_ticket(self.db_path, "Q", "admissions", "source1")

        mock_get_connection.assert_called_once()
        mock_sleep.assert_not_called()
        bad_conn.close.assert_called_once()

    @patch("app.database.tickets._get_connection")
    @patch("app.database.tickets.time.sleep")
    def test_constraint_violation_is_not_retried(
        self, mock_sleep: MagicMock, mock_get_connection: MagicMock
    ) -> None:
        """IntegrityError is a sibling of OperationalError, not a subclass —
        confirms it is never caught by the lock-retry except clause at all."""
        integrity_conn = _mock_connection([sqlite3.IntegrityError("UNIQUE constraint failed")])
        mock_get_connection.return_value = integrity_conn

        with self.assertRaises(sqlite3.IntegrityError):
            create_ticket(self.db_path, "Q", "admissions", "source1")

        mock_get_connection.assert_called_once()
        mock_sleep.assert_not_called()

    def test_persistence_across_connections(self) -> None:
        """Data persists when a new connection is established."""
        ticket = create_ticket(self.db_path, "Persistence test", "facilities", "source")
        
        # Re-initialize to mimic app restart (should not drop table)
        initialize_database(self.db_path)
        
        # Fetching with a completely new function call -> new connection
        retrieved = get_ticket(self.db_path, ticket.ticket_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.query, "Persistence test")


class TestUpdateTicketStatus(unittest.TestCase):
    """Phase 7C: the open -> resolved staff transition."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_escalation.db"
        initialize_database(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_valid_statuses_are_exactly_open_and_resolved(self) -> None:
        """The state machine stays deliberately minimal -- no in_progress,
        reopened, pending, archived, or deleted."""
        self.assertEqual(VALID_TICKET_STATUSES, frozenset({"open", "resolved"}))

    def test_open_ticket_becomes_resolved(self) -> None:
        ticket = create_ticket(self.db_path, "Q", "fees", "stage_1_low")
        self.assertEqual(ticket.status, "open")

        updated = update_ticket_status(self.db_path, ticket.ticket_id, "resolved")

        self.assertIsInstance(updated, Ticket)
        self.assertEqual(updated.ticket_id, ticket.ticket_id)
        self.assertEqual(updated.status, "resolved")

    def test_resolved_status_is_persisted_and_retrievable(self) -> None:
        ticket = create_ticket(self.db_path, "Q", "fees", "stage_1_low")
        update_ticket_status(self.db_path, ticket.ticket_id, "resolved")

        # Fresh connection via the existing read path.
        retrieved = get_ticket(self.db_path, ticket.ticket_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.status, "resolved")

        # And it moves between the existing status-filtered listings.
        self.assertEqual(list_tickets(self.db_path, status="open"), [])
        resolved_list = list_tickets(self.db_path, status="resolved")
        self.assertEqual(len(resolved_list), 1)
        self.assertEqual(resolved_list[0].ticket_id, ticket.ticket_id)

    def test_unknown_ticket_id_returns_none(self) -> None:
        """Mirrors get_ticket()'s None-for-not-found contract."""
        self.assertIsNone(update_ticket_status(self.db_path, "does-not-exist", "resolved"))

    def test_invalid_status_raises_value_error(self) -> None:
        ticket = create_ticket(self.db_path, "Q", "fees", "stage_1_low")

        for bad_status in ("closed", "in_progress", "RESOLVED", "", "archived"):
            with self.subTest(status=bad_status):
                with self.assertRaisesRegex(ValueError, "Invalid ticket status"):
                    update_ticket_status(self.db_path, ticket.ticket_id, bad_status)

        # The rejected calls left the ticket untouched.
        self.assertEqual(get_ticket(self.db_path, ticket.ticket_id).status, "open")

    def test_invalid_status_is_rejected_before_any_database_access(self) -> None:
        with patch("app.database.tickets._get_connection") as mock_get_connection:
            with self.assertRaises(ValueError):
                update_ticket_status(self.db_path, "any-id", "bogus")
        mock_get_connection.assert_not_called()

    def test_resolving_an_already_resolved_ticket_is_idempotent(self) -> None:
        ticket = create_ticket(self.db_path, "Q", "fees", "stage_1_low")
        first = update_ticket_status(self.db_path, ticket.ticket_id, "resolved")
        second = update_ticket_status(self.db_path, ticket.ticket_id, "resolved")

        self.assertEqual(first.status, "resolved")
        self.assertEqual(second.status, "resolved")
        self.assertEqual(second.ticket_id, ticket.ticket_id)
        # No duplicate row was created by the repeat call.
        self.assertEqual(len(list_tickets(self.db_path)), 1)

    def test_resolved_ticket_cannot_be_silently_reopened(self) -> None:
        ticket = create_ticket(self.db_path, "Q", "fees", "stage_1_low")
        update_ticket_status(self.db_path, ticket.ticket_id, "resolved")

        with self.assertRaisesRegex(ValueError, "already resolved"):
            update_ticket_status(self.db_path, ticket.ticket_id, "open")

        # Still resolved -- the refused transition changed nothing.
        self.assertEqual(get_ticket(self.db_path, ticket.ticket_id).status, "resolved")

    def test_setting_open_ticket_to_open_is_idempotent(self) -> None:
        ticket = create_ticket(self.db_path, "Q", "fees", "stage_1_low")
        updated = update_ticket_status(self.db_path, ticket.ticket_id, "open")
        self.assertEqual(updated.status, "open")
        self.assertEqual(len(list_tickets(self.db_path)), 1)

    def test_unrelated_ticket_fields_are_unchanged(self) -> None:
        ticket = create_ticket(
            self.db_path,
            "What is the WiFi password?",
            "facilities",
            "stage_2_insufficient_evidence",
            user_metadata='{"user_id": 7}',
        )

        updated = update_ticket_status(self.db_path, ticket.ticket_id, "resolved")

        for field in ("ticket_id", "query", "predicted_intent", "created_at", "source", "user_metadata"):
            with self.subTest(field=field):
                self.assertEqual(getattr(updated, field), getattr(ticket, field))

        # ...and the same holds for what is actually stored on disk.
        stored = get_ticket(self.db_path, ticket.ticket_id)
        for field in ("ticket_id", "query", "predicted_intent", "created_at", "source", "user_metadata"):
            with self.subTest(field=field, layer="db"):
                self.assertEqual(getattr(stored, field), getattr(ticket, field))

    def test_update_does_not_affect_other_tickets(self) -> None:
        t1 = create_ticket(self.db_path, "Q1", "admissions", "source1")
        t2 = create_ticket(self.db_path, "Q2", "fees", "source2")

        update_ticket_status(self.db_path, t1.ticket_id, "resolved")

        self.assertEqual(get_ticket(self.db_path, t1.ticket_id).status, "resolved")
        self.assertEqual(get_ticket(self.db_path, t2.ticket_id).status, "open")
        self.assertEqual(len(list_tickets(self.db_path)), 2)

    @patch("app.database.tickets.time.sleep")
    def test_normal_update_succeeds_without_retry(self, mock_sleep: MagicMock) -> None:
        ticket = create_ticket(self.db_path, "Q", "fees", "stage_1_low")
        mock_sleep.reset_mock()
        update_ticket_status(self.db_path, ticket.ticket_id, "resolved")
        mock_sleep.assert_not_called()

    @patch("app.database.tickets._get_connection")
    @patch("app.database.tickets.time.sleep")
    def test_update_retries_transient_lock_then_succeeds(
        self, mock_sleep: MagicMock, mock_get_connection: MagicMock
    ) -> None:
        """M4's bounded lock retry also covers the new update path."""
        stored_row = {
            "ticket_id": "tk-1",
            "query": "Q",
            "predicted_intent": "fees",
            "status": "open",
            "created_at": "2026-01-01T00:00:00Z",
            "source": "stage_1_low",
            "user_metadata": None,
        }

        locked_conn = _mock_connection([sqlite3.OperationalError("database is locked")])

        select_cursor = MagicMock()
        select_cursor.fetchone.return_value = stored_row
        ok_conn = _mock_connection([select_cursor, None])  # SELECT then UPDATE

        mock_get_connection.side_effect = [locked_conn, ok_conn]

        updated = update_ticket_status(self.db_path, "tk-1", "resolved")

        self.assertEqual(updated.status, "resolved")
        self.assertEqual(mock_get_connection.call_count, 2)
        mock_sleep.assert_called_once()
        locked_conn.close.assert_called_once()
        ok_conn.close.assert_called_once()

    @patch("app.database.tickets._get_connection")
    @patch("app.database.tickets.time.sleep")
    def test_update_non_transient_error_is_not_retried(
        self, mock_sleep: MagicMock, mock_get_connection: MagicMock
    ) -> None:
        bad_conn = _mock_connection([sqlite3.OperationalError("no such table: tickets")])
        mock_get_connection.return_value = bad_conn

        with self.assertRaises(sqlite3.OperationalError):
            update_ticket_status(self.db_path, "tk-1", "resolved")

        mock_get_connection.assert_called_once()
        mock_sleep.assert_not_called()
        bad_conn.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
