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
    def test_ready_recognizes_api_key_from_dotenv_file(self, mock_get_collection: MagicMock) -> None:
        """Phase 7D-2: a key that exists only in the repo-root .env (not
        exported to the shell) must count as present. Before this fix /ready
        read the raw process environment only, so it reported 'missing or
        blank' -- and a 503 -- for a key that /chat was successfully using."""
        mock_ready_collection = MagicMock()
        mock_ready_collection.count.return_value = 957
        mock_get_collection.return_value = mock_ready_collection

        env_file = Path(self.temp_dir.name) / "dotenv_only.env"
        env_file.write_text(
            "GEMINI_API_KEY=key-from-dotenv-file\n"
            "GEMINI_GENERATION_MODEL=gemini-2.5-flash\n",
            encoding="utf-8",
        )

        # patch.dict snapshots os.environ and restores it on exit, so the key
        # load_dotenv performs here cannot leak into any other test.
        with patch.dict(os.environ), patch("app.api.main.ENV_PATH", env_file):
            os.environ.pop("GEMINI_API_KEY", None)
            self.assertNotIn("GEMINI_API_KEY", os.environ)  # precondition
            response = self.client.get("/ready")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ready")
        self.assertTrue(data["checks"]["gemini_api_key"]["ok"])
        self.assertEqual(data["checks"]["gemini_api_key"]["detail"], "present")
        # Presence only -- the value from .env is never echoed back.
        self.assertNotIn("key-from-dotenv-file", response.text)

    @patch("app.api.main.get_collection")
    def test_ready_reports_missing_key_when_absent_from_env_and_dotenv(
        self, mock_get_collection: MagicMock
    ) -> None:
        """With no key in the process environment and a .env that does not
        define one, /ready must still correctly report it missing."""
        mock_ready_collection = MagicMock()
        mock_ready_collection.count.return_value = 957
        mock_get_collection.return_value = mock_ready_collection

        env_file = Path(self.temp_dir.name) / "no_key.env"
        env_file.write_text("GEMINI_GENERATION_MODEL=gemini-2.5-flash\n", encoding="utf-8")

        with patch.dict(os.environ), patch("app.api.main.ENV_PATH", env_file):
            os.environ.pop("GEMINI_API_KEY", None)
            response = self.client.get("/ready")

        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["status"], "not_ready")
        self.assertFalse(data["checks"]["gemini_api_key"]["ok"])
        self.assertEqual(data["checks"]["gemini_api_key"]["detail"], "missing or blank")

    @patch("app.api.main.get_collection")
    def test_ready_tolerates_absent_dotenv_file(self, mock_get_collection: MagicMock) -> None:
        """A missing .env must not raise -- an exported env var alone suffices."""
        mock_ready_collection = MagicMock()
        mock_ready_collection.count.return_value = 957
        mock_get_collection.return_value = mock_ready_collection

        missing_env = Path(self.temp_dir.name) / "does_not_exist.env"
        self.assertFalse(missing_env.exists())

        with patch.dict(os.environ, {"GEMINI_API_KEY": "exported-key"}), \
             patch("app.api.main.ENV_PATH", missing_env):
            response = self.client.get("/ready")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")

    @patch("app.api.main.get_collection")
    def test_exported_env_var_takes_precedence_over_dotenv(
        self, mock_get_collection: MagicMock
    ) -> None:
        """python-dotenv must not clobber a real exported variable."""
        mock_ready_collection = MagicMock()
        mock_ready_collection.count.return_value = 957
        mock_get_collection.return_value = mock_ready_collection

        env_file = Path(self.temp_dir.name) / "conflicting.env"
        env_file.write_text("GEMINI_API_KEY=value-from-file\n", encoding="utf-8")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "value-from-shell"}), \
             patch("app.api.main.ENV_PATH", env_file):
            response = self.client.get("/ready")
            self.assertEqual(os.environ["GEMINI_API_KEY"], "value-from-shell")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("value-from-shell", response.text)
        self.assertNotIn("value-from-file", response.text)

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

    # -----------------------------------------------------------------
    # PATCH /tickets/{ticket_id} — Phase 7C (additive endpoint)
    # -----------------------------------------------------------------

    def test_patch_ticket_resolves_open_ticket(self) -> None:
        ticket = create_ticket(self.db_path, "Q", "fees", "stage_1_low")

        response = self.client.patch(
            f"/tickets/{ticket.ticket_id}", json={"status": "resolved"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["ticket_id"], ticket.ticket_id)
        self.assertEqual(data["status"], "resolved")
        # Unrelated fields survive the update.
        self.assertEqual(data["query"], "Q")
        self.assertEqual(data["predicted_intent"], "fees")
        self.assertEqual(data["created_at"], ticket.created_at)
        self.assertEqual(data["source"], "stage_1_low")

    def test_patch_ticket_is_visible_through_existing_get_endpoints(self) -> None:
        ticket = create_ticket(self.db_path, "Q", "fees", "stage_1_low")
        self.client.patch(f"/tickets/{ticket.ticket_id}", json={"status": "resolved"})

        get_response = self.client.get(f"/tickets/{ticket.ticket_id}")
        self.assertEqual(get_response.json()["status"], "resolved")

        open_list = self.client.get("/tickets", params={"status_filter": "open"})
        self.assertEqual(open_list.json(), [])

        resolved_list = self.client.get("/tickets", params={"status_filter": "resolved"})
        self.assertEqual(len(resolved_list.json()), 1)

    def test_patch_unknown_ticket_returns_404(self) -> None:
        response = self.client.patch("/tickets/not-a-ticket", json={"status": "resolved"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Ticket not found")

    def test_patch_invalid_status_is_rejected(self) -> None:
        ticket = create_ticket(self.db_path, "Q", "fees", "stage_1_low")

        # "open" is deliberately NOT exposed -- a resolved ticket must not be
        # re-openable through the API -- alongside statuses outside the model.
        for bad_status in ("open", "closed", "in_progress", "", None, 123):
            with self.subTest(status=bad_status):
                response = self.client.patch(
                    f"/tickets/{ticket.ticket_id}", json={"status": bad_status}
                )
                self.assertEqual(response.status_code, 422)

        # Nothing was mutated by any rejected call.
        self.assertEqual(
            self.client.get(f"/tickets/{ticket.ticket_id}").json()["status"], "open"
        )

    def test_patch_malformed_payload_is_rejected(self) -> None:
        ticket = create_ticket(self.db_path, "Q", "fees", "stage_1_low")

        for payload in ({}, {"stat": "resolved"}, {"status": ["resolved"]}):
            with self.subTest(payload=payload):
                response = self.client.patch(f"/tickets/{ticket.ticket_id}", json=payload)
                self.assertEqual(response.status_code, 422)

    def test_patch_already_resolved_ticket_is_idempotent(self) -> None:
        ticket = create_ticket(self.db_path, "Q", "fees", "stage_1_low")

        first = self.client.patch(f"/tickets/{ticket.ticket_id}", json={"status": "resolved"})
        second = self.client.patch(f"/tickets/{ticket.ticket_id}", json={"status": "resolved"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["status"], "resolved")
        # No duplicate ticket rows were produced.
        self.assertEqual(len(self.client.get("/tickets").json()), 1)

    @patch("app.api.main.update_ticket_status")
    def test_patch_database_failure_does_not_leak_details(self, mock_update) -> None:
        mock_update.side_effect = RuntimeError("disk I/O error: /secret/path/escalation.db")

        response = self.client.patch("/tickets/tk-1", json={"status": "resolved"})

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("disk I/O error", response.text)
        self.assertNotIn("/secret/path", response.text)
        self.assertEqual(response.json()["detail"], "Could not update ticket.")

    @patch("app.api.main.update_ticket_status")
    def test_patch_invalid_transition_returns_400(self, mock_update) -> None:
        """Not reachable through the current request model, but the endpoint
        still translates a data-layer ValueError into a clean 400."""
        mock_update.side_effect = ValueError("Ticket tk-1 is already resolved and cannot be changed to 'open'.")

        response = self.client.patch("/tickets/tk-1", json={"status": "resolved"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid ticket status transition.")

    def test_existing_endpoints_unchanged_by_patch_addition(self) -> None:
        """Backward-compatibility guard for the pre-Phase-7C contracts."""
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})

        ticket = create_ticket(self.db_path, "Q", "fees", "source")

        listing = self.client.get("/tickets")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(
            set(listing.json()[0].keys()),
            {"ticket_id", "query", "predicted_intent", "status", "created_at", "source", "user_metadata"},
        )

        single = self.client.get(f"/tickets/{ticket.ticket_id}")
        self.assertEqual(single.status_code, 200)
        self.assertEqual(set(single.json().keys()), set(listing.json()[0].keys()))

        # GET on an id still 404s; PATCH did not shadow the GET route.
        self.assertEqual(self.client.get("/tickets/nope").status_code, 404)


if __name__ == "__main__":
    unittest.main()
