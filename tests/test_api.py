"""
Tests for app/api/main.py — FastAPI wrapper
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.main import app, get_collection, get_db_path
from app.rag.pipeline import RagResult, Citation, GateMetrics, RetrievedChunk
from app.database.tickets import initialize_database, create_ticket


class TestFastAPIWrapper(unittest.TestCase):
    def setUp(self) -> None:
        # Override dependencies
        self.mock_collection = MagicMock()
        
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_api.db"
        initialize_database(self.db_path)

        app.dependency_overrides[get_collection] = lambda: self.mock_collection
        app.dependency_overrides[get_db_path] = lambda: self.db_path

        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        self.temp_dir.cleanup()

    def test_health_check(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        # Assert Chroma was not called
        self.mock_collection.query.assert_not_called()

    def test_health_check_unchanged_after_ready_endpoint_added(self) -> None:
        """/health must remain byte-for-byte identical after /ready is introduced."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertEqual(set(response.json().keys()), {"status"})

    @patch("app.api.main.get_collection")
    def test_ready_all_checks_pass(self, mock_get_collection: MagicMock) -> None:
        mock_ready_collection = MagicMock()
        mock_ready_collection.count.return_value = 957
        mock_get_collection.return_value = mock_ready_collection

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-value"}):
            response = self.client.get("/ready")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ready")
        self.assertTrue(data["checks"]["gemini_api_key"]["ok"])
        self.assertTrue(data["checks"]["chroma"]["ok"])
        # Never expose the actual key value
        self.assertNotIn("test-key-value", response.text)

    @patch("app.api.main.get_collection")
    def test_ready_missing_api_key_returns_503(self, mock_get_collection: MagicMock) -> None:
        mock_ready_collection = MagicMock()
        mock_ready_collection.count.return_value = 957
        mock_get_collection.return_value = mock_ready_collection

        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            response = self.client.get("/ready")

        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["status"], "not_ready")
        self.assertFalse(data["checks"]["gemini_api_key"]["ok"])
        self.assertTrue(data["checks"]["chroma"]["ok"])

    @patch("app.api.main.get_collection")
    def test_ready_chroma_unreachable_returns_503(self, mock_get_collection: MagicMock) -> None:
        mock_get_collection.side_effect = RuntimeError("Could not connect to Chroma")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-value"}):
            response = self.client.get("/ready")

        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["status"], "not_ready")
        self.assertTrue(data["checks"]["gemini_api_key"]["ok"])
        self.assertFalse(data["checks"]["chroma"]["ok"])
        # No raw exception text leaked
        self.assertNotIn("Could not connect to Chroma", response.text)

    @patch("app.api.main.get_collection")
    def test_ready_chroma_empty_collection_returns_503(self, mock_get_collection: MagicMock) -> None:
        mock_empty_collection = MagicMock()
        mock_empty_collection.count.return_value = 0
        mock_get_collection.return_value = mock_empty_collection

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-value"}):
            response = self.client.get("/ready")

        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["status"], "not_ready")
        self.assertTrue(data["checks"]["gemini_api_key"]["ok"])
        self.assertFalse(data["checks"]["chroma"]["ok"])

    @patch("google.genai.Client")
    @patch("app.api.main.get_collection")
    def test_ready_never_calls_gemini(
        self, mock_get_collection: MagicMock, mock_genai_client: MagicMock
    ) -> None:
        mock_ready_collection = MagicMock()
        mock_ready_collection.count.return_value = 957
        mock_get_collection.return_value = mock_ready_collection

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-value"}):
            # Exercise both the ready and not-ready branches.
            self.client.get("/ready")
            with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
                self.client.get("/ready")

        mock_genai_client.assert_not_called()

    def test_oversized_query_rejected(self) -> None:
        response = self.client.post("/chat", json={"query": "A" * 1500})
        self.assertEqual(response.status_code, 422)  # Pydantic validation error

    def test_empty_query_rejected(self) -> None:
        response = self.client.post("/chat", json={"query": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("cannot be whitespace", response.json()["detail"])

    @patch("app.api.main.run_rag_pipeline")
    def test_valid_chat_answer(self, mock_run) -> None:
        mock_run.return_value = RagResult(
            status="answered",
            query="test",
            intent="admissions",
            answer="Yes",
            citations=[Citation("c1", None, "url", "sec", None)],
            confidence_status="high",
            ticket_id=None,
            reason="success",
            retrieved_chunks=[],
            gate_metrics=None
        )
        
        response = self.client.post("/chat", json={"query": "test"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "answered")
        self.assertEqual(data["answer"], "Yes")
        self.assertEqual(len(data["citations"]), 1)
        self.assertEqual(data["citations"][0]["chunk_id"], "c1")

    @patch("app.api.main.run_rag_pipeline")
    def test_out_of_scope_chat(self, mock_run) -> None:
        mock_run.return_value = RagResult(
            status="out_of_scope",
            query="test",
            intent="out_of_scope",
            answer="OOS",
            citations=[],
            confidence_status="not_applicable",
            ticket_id=None,
            reason="oos",
            retrieved_chunks=[],
            gate_metrics=None
        )
        
        response = self.client.post("/chat", json={"query": "test"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "out_of_scope")

    @patch("app.api.main.run_rag_pipeline")
    def test_escalated_chat(self, mock_run) -> None:
        mock_run.return_value = RagResult(
            status="escalated",
            query="test",
            intent="fees",
            answer=None,
            citations=None,
            confidence_status="low",
            ticket_id="tk-999",
            reason="weak",
            retrieved_chunks=[],
            gate_metrics=None
        )
        
        response = self.client.post("/chat", json={"query": "test"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "escalated")
        self.assertEqual(data["ticket_id"], "tk-999")

    @patch("app.api.main.run_rag_pipeline")
    def test_internal_pipeline_failure_returns_500(self, mock_run) -> None:
        mock_run.return_value = RagResult(
            status="error",
            query="test",
            intent="unknown",
            answer=None,
            citations=None,
            confidence_status=None,
            ticket_id=None,
            reason="Pipeline blew up",
            retrieved_chunks=[],
            gate_metrics=None
        )
        
        response = self.client.post("/chat", json={"query": "test"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Pipeline blew up")

    @patch("app.api.main.run_rag_pipeline")
    def test_unhandled_exception_returns_500(self, mock_run) -> None:
        mock_run.side_effect = RuntimeError("Database crashed entirely")
        
        response = self.client.post("/chat", json={"query": "test"})
        self.assertEqual(response.status_code, 500)
        # Verify traceback is not leaked
        self.assertNotIn("Database crashed entirely", response.json()["detail"])
        self.assertEqual(response.json()["detail"], "Pipeline error")

    def test_ticket_listing_and_retrieval(self) -> None:
        # Create directly in DB
        ticket = create_ticket(self.db_path, "Q", "fees", "source")
        
        # Test GET /tickets
        list_response = self.client.get("/tickets")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(len(list_response.json()), 1)
        
        # Test GET /tickets/{id}
        get_response = self.client.get(f"/tickets/{ticket.ticket_id}")
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(get_response.json()["ticket_id"], ticket.ticket_id)
        
        # Test 404
        bad_response = self.client.get("/tickets/not-a-ticket")
        self.assertEqual(bad_response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
