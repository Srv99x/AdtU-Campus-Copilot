"""
Tests for app/database/tickets.py — SQLite ticket persistence subsystem.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.database.tickets import (
    Ticket,
    create_ticket,
    get_ticket,
    initialize_database,
    list_tickets,
)


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

    def test_persistence_across_connections(self) -> None:
        """Data persists when a new connection is established."""
        ticket = create_ticket(self.db_path, "Persistence test", "facilities", "source")
        
        # Re-initialize to mimic app restart (should not drop table)
        initialize_database(self.db_path)
        
        # Fetching with a completely new function call -> new connection
        retrieved = get_ticket(self.db_path, ticket.ticket_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.query, "Persistence test")


if __name__ == "__main__":
    unittest.main()
