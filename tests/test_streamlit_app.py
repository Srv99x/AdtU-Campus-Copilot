"""
Tests for app/ui/streamlit_app.py — Guided Demo Mode (Phase 7A) and the
Trust & Evidence Panel (Phase 7B).

Uses Streamlit's own AppTest harness to run the real script and drive it
like a user would, with the backend (`requests`) mocked so no network call
or Gemini/Chroma access ever happens. Verifies the demo scenario buttons
are pure data plus a click that submits through the SAME chat workflow as
manually typed input -- never a second implementation, never a canned
answer. Phase 7B tests further verify the evidence/escalation panels only
ever render fields actually present in the mocked ChatResponse -- never a
fabricated confidence value, citation, or ticket detail.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
from streamlit.testing.v1 import AppTest

from app.ui.streamlit_app import DEMO_SCENARIOS

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "ui" / "streamlit_app.py")
RUN_TIMEOUT = 20  # AppTest's default 3s is too tight for a cold script run


def _mock_response(json_body, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.raise_for_status.return_value = None
    return resp


# Shape returned by a healthy GET /ready (see app/api/main.py readiness_check).
_READY_PAYLOAD = {
    "status": "ready",
    "checks": {
        "gemini_api_key": {"ok": True, "detail": "present"},
        "chroma": {"ok": True, "detail": "collection accessible with 957 records"},
    },
}


class TestDemoScenarioDefinitions(unittest.TestCase):
    """Pure data-shape checks on the curated example list -- no app run needed."""

    def test_exactly_five_scenarios(self) -> None:
        self.assertEqual(len(DEMO_SCENARIOS), 5)

    def test_every_scenario_has_a_label_and_a_nonblank_query(self) -> None:
        for scenario in DEMO_SCENARIOS:
            self.assertIn("label", scenario)
            self.assertIn("query", scenario)
            self.assertTrue(scenario["label"].strip())
            self.assertTrue(scenario["query"].strip())

    def test_labels_are_unique(self) -> None:
        labels = [s["label"] for s in DEMO_SCENARIOS]
        self.assertEqual(len(labels), len(set(labels)))

    def test_covers_the_five_required_scenario_themes(self) -> None:
        """Loose vocabulary checks, not exact-string checks, so the example
        queries stay easy to edit without this test becoming brittle."""
        queries = [s["query"].lower() for s in DEMO_SCENARIOS]

        self.assertTrue(
            any("admission" in q or "fee" in q or "facilit" in q for q in queries),
            "expected a normal campus Q&A scenario",
        )
        self.assertTrue(
            any("scholarship" in q and ("%" in q or "percent" in q or "amount" in q) for q in queries),
            "expected a monetary scholarship routing scenario",
        )
        self.assertTrue(
            any("routine" in q or "schedule" in q or "timetable" in q for q in queries),
            "expected a class-routine retrieval scenario",
        )
        self.assertTrue(
            any("capital" in q or "france" in q for q in queries),
            "expected an out-of-scope rejection scenario",
        )
        self.assertTrue(
            any("wifi" in q or "password" in q for q in queries),
            "expected a likely-safe-escalation scenario",
        )


class TestGuidedDemoModeIntegration(unittest.TestCase):
    """End-to-end (backend-mocked) checks that a scenario button drives the
    real chat workflow -- not a second, parallel UI implementation."""

    def setUp(self) -> None:
        self.get_patcher = patch("requests.get")
        self.post_patcher = patch("requests.post")
        self.mock_get = self.get_patcher.start()
        self.mock_post = self.post_patcher.start()
        self.addCleanup(self.get_patcher.stop)
        self.addCleanup(self.post_patcher.stop)

        self.mock_get.return_value = _mock_response({"status": "ok"})

    def test_all_scenario_buttons_render(self) -> None:
        self.mock_post.return_value = _mock_response({"status": "out_of_scope"})

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)

        self.assertFalse(at.exception)
        rendered_labels = {b.label for b in at.button}
        for scenario in DEMO_SCENARIOS:
            self.assertIn(scenario["label"], rendered_labels)

    def test_clicking_scenario_submits_its_own_query_to_the_chat_endpoint(self) -> None:
        """Clicking a scenario must send exactly its own query text to the
        real /chat call -- confirming no answer is hardcoded and no second
        chat path exists; it goes through send_chat_query() like any typed
        message would."""
        self.mock_post.return_value = _mock_response(
            {
                "status": "answered",
                "intent": "admissions",
                "answer": "Grounded answer from the real pipeline.",
                "citations": [],
                "confidence_status": "high",
                "ticket_id": None,
                "reason": "ok",
            }
        )

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)

        target_scenario = DEMO_SCENARIOS[0]
        buttons = [b for b in at.button if b.label == target_scenario["label"]]
        self.assertEqual(len(buttons), 1)

        buttons[0].click().run(timeout=RUN_TIMEOUT)

        self.assertFalse(at.exception)
        self.mock_post.assert_called_once()
        sent_query = self.mock_post.call_args.kwargs["json"]["query"]
        self.assertEqual(sent_query, target_scenario["query"])

        # Rendered through the existing chat transcript (st.chat_message),
        # not a separate surface.
        user_texts = [
            m.value
            for cm in at.chat_message
            if cm.name == "user"
            for m in cm.markdown
        ]
        assistant_texts = [
            m.value
            for cm in at.chat_message
            if cm.name == "assistant"
            for m in cm.markdown
        ]
        self.assertIn(target_scenario["query"], user_texts)
        self.assertIn("Grounded answer from the real pipeline.", assistant_texts)

    def test_clicking_out_of_scope_scenario_shows_rejection_not_a_fabricated_answer(self) -> None:
        """The out-of-scope scenario must go through the same status-handling
        branch as a manually typed out-of-scope query -- no shortcut."""
        self.mock_post.return_value = _mock_response(
            {
                "status": "out_of_scope",
                "intent": "out_of_scope",
                "answer": "This query is outside the scope of AdtU Campus Copilot.",
                "citations": [],
                "confidence_status": "not_applicable",
                "ticket_id": None,
                "reason": "Query classified as out_of_scope.",
            }
        )

        oos_scenario = next(s for s in DEMO_SCENARIOS if "capital" in s["query"].lower())

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        button = next(b for b in at.button if b.label == oos_scenario["label"])
        button.click().run(timeout=RUN_TIMEOUT)

        self.assertFalse(at.exception)
        sent_query = self.mock_post.call_args.kwargs["json"]["query"]
        self.assertEqual(sent_query, oos_scenario["query"])

        warning_texts = [w.value for w in at.warning]
        self.assertTrue(
            any("only handles AdtU-related queries" in w for w in warning_texts)
        )

    def test_manual_chat_input_still_works_unchanged(self) -> None:
        """Guided Demo Mode must not alter the pre-existing manual chat path."""
        self.mock_post.return_value = _mock_response(
            {
                "status": "answered",
                "intent": "fees",
                "answer": "Manually typed answer.",
                "citations": [],
                "confidence_status": "high",
                "ticket_id": None,
                "reason": "ok",
            }
        )

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.chat_input[0].set_value("What is the hostel fee?").run(timeout=RUN_TIMEOUT)

        self.assertFalse(at.exception)
        sent_query = self.mock_post.call_args.kwargs["json"]["query"]
        self.assertEqual(sent_query, "What is the hostel fee?")
        assistant_texts = [
            m.value
            for cm in at.chat_message
            if cm.name == "assistant"
            for m in cm.markdown
        ]
        self.assertIn("Manually typed answer.", assistant_texts)


class TestTrustAndEvidencePanel(unittest.TestCase):
    """Phase 7B: the '🔎 Why this answer?' / '🟡 Not enough verified evidence'
    panels must render only fields already present in the mocked
    ChatResponse -- never a computed, guessed, or fabricated value."""

    def setUp(self) -> None:
        self.get_patcher = patch("requests.get")
        self.post_patcher = patch("requests.post")
        self.mock_get = self.get_patcher.start()
        self.mock_post = self.post_patcher.start()
        self.addCleanup(self.get_patcher.stop)
        self.addCleanup(self.post_patcher.stop)

        self.mock_get.return_value = _mock_response({"status": "ok"})

    def _ask(self, query: str) -> AppTest:
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.chat_input[0].set_value(query).run(timeout=RUN_TIMEOUT)
        self.assertFalse(at.exception)
        return at

    def test_grounded_answer_displays_evidence_panel(self) -> None:
        self.mock_post.return_value = _mock_response(
            {
                "status": "answered",
                "intent": "fees",
                "answer": "The BCA fee is X per semester.",
                "citations": [
                    {
                        "chunk_id": "v2_fees_6",
                        "parent_chunk_id": None,
                        "source_url": "https://adtu.in/fees",
                        "section": "BCA Fee Structure",
                        "source_type": "html",
                    }
                ],
                "confidence_status": "high",
                "ticket_id": None,
                "reason": "dp-top3-mean=0.1900 < 0.275",
            }
        )

        at = self._ask("Total fee structure for BCA with IBM collaboration")

        panel_labels = [e.label for e in at.expander]
        self.assertIn("🔎 Why this answer?", panel_labels)
        self.assertNotIn("🟡 Not enough verified evidence", panel_labels)

    def test_real_confidence_and_evidence_values_are_displayed(self) -> None:
        """The panel must show the ACTUAL backend-provided confidence tier,
        intent, citation count, and reasoning text -- not placeholder text."""
        self.mock_post.return_value = _mock_response(
            {
                "status": "answered",
                "intent": "admissions",
                "answer": "Documents required: ...",
                "citations": [
                    {
                        "chunk_id": "c1",
                        "parent_chunk_id": None,
                        "source_url": "https://adtu.in/admissions",
                        "section": "Required Documents",
                        "source_type": "html",
                    },
                    {
                        "chunk_id": "c2",
                        "parent_chunk_id": None,
                        "source_url": "https://adtu.in/admissions#docs",
                        "section": "Required Documents",
                        "source_type": "html",
                    },
                ],
                "confidence_status": "high",
                "ticket_id": None,
                "reason": "dp-top3-mean=0.2100 < 0.275",
            }
        )

        at = self._ask("What documents are required for BTech admission at AdtU?")

        panel = next(e for e in at.expander if e.label == "🔎 Why this answer?")
        captions = [c.value for c in panel.caption]

        self.assertTrue(any("admissions" in c for c in captions))
        self.assertTrue(any("high" in c for c in captions))
        self.assertTrue(any("Sources cited: 2" in c for c in captions))
        self.assertTrue(any("dp-top3-mean=0.2100 < 0.275" in c for c in captions))

    def test_citations_rendered_with_existing_citation_formatter(self) -> None:
        """Both the clickable-URL form and the non-fabricated blank-URL
        fallback (Phase 6A's format_citation_reference) must appear
        verbatim inside the new panel -- no separate/duplicate formatting
        logic was introduced for Phase 7B."""
        self.mock_post.return_value = _mock_response(
            {
                "status": "answered",
                "intent": "facilities",
                "answer": "Some grounded facilities answer.",
                "citations": [
                    {
                        "chunk_id": "c1",
                        "parent_chunk_id": None,
                        "source_url": "https://adtu.in/hostel",
                        "section": "Hostel Fees",
                        "source_type": "pdf",
                    },
                    {
                        "chunk_id": "v2_scholarships_0",
                        "parent_chunk_id": "v2_scholarships_parent",
                        "source_url": "",
                        "section": "Scholarship Table",
                        "source_type": "pdf",
                    },
                ],
                "confidence_status": "high",
                "ticket_id": None,
                "reason": "ok",
            }
        )

        at = self._ask("Warden name for A Block girls hostel")

        panel = next(e for e in at.expander if e.label == "🔎 Why this answer?")
        markdown_lines = [m.value for m in panel.markdown]

        # Clickable form, produced by format_citation_reference() unchanged.
        self.assertIn("- [Hostel Fees](https://adtu.in/hostel)", markdown_lines)
        # Non-fabricated fallback identity for a blank source_url, including
        # the real parent lineage -- exact same string the formatter has
        # always produced (Phase 6A), never a fabricated URL.
        self.assertIn(
            "- Scholarship Table (chunk `v2_scholarships_0`) — pdf — source link unavailable (via `v2_scholarships_parent`)",
            markdown_lines,
        )

    def test_missing_optional_metadata_does_not_fabricate_values(self) -> None:
        """No citations, no reason, no confidence_status returned by the
        backend -- the panel must show only what IS present (intent) and
        must never invent a confidence tier, citation, or reasoning string."""
        self.mock_post.return_value = _mock_response(
            {
                "status": "answered",
                "intent": "facilities",
                "answer": "A minimal grounded answer.",
                "citations": None,
                "confidence_status": None,
                "ticket_id": None,
                "reason": "",
            }
        )

        at = self._ask("Some minimal-metadata facilities question")

        panel = next(e for e in at.expander if e.label == "🔎 Why this answer?")
        captions = [c.value for c in panel.caption]
        markdown_lines = [m.value for m in panel.markdown]

        self.assertTrue(any("facilities" in c for c in captions))
        self.assertTrue(any("No citation metadata was returned" in c for c in captions))
        # Nothing fabricated: no confidence/sources-cited/reasoning captions,
        # no cited-evidence markdown block at all.
        self.assertFalse(any("Confidence:" in c for c in captions))
        self.assertFalse(any("Sources cited" in c for c in captions))
        self.assertFalse(any("Backend reasoning" in c for c in captions))
        self.assertEqual(markdown_lines, [])

    def test_escalated_response_displays_safe_escalation_explanation(self) -> None:
        self.mock_post.return_value = _mock_response(
            {
                "status": "escalated",
                "intent": "fees",
                "answer": None,
                "citations": None,
                "confidence_status": "low",
                "ticket_id": "tk-abc123",
                "reason": "Escalated due to weak retrieval confidence.",
            }
        )

        at = self._ask("Exam schedule for first semester 2024")

        panel_labels = [e.label for e in at.expander]
        self.assertIn("🟡 Not enough verified evidence", panel_labels)
        self.assertNotIn("🔎 Why this answer?", panel_labels)

        panel = next(e for e in at.expander if e.label == "🟡 Not enough verified evidence")
        captions = [c.value for c in panel.caption]
        self.assertTrue(any("low" in c for c in captions))
        self.assertTrue(any("Escalated due to weak retrieval confidence." in c for c in captions))
        self.assertTrue(any("tk-abc123" in c for c in captions))

    def test_guided_demo_mode_still_works_with_evidence_panel(self) -> None:
        """Phase 7A's scenario buttons must still drive the real workflow,
        now including the new Phase 7B panel, unchanged in mechanism."""
        self.mock_post.return_value = _mock_response(
            {
                "status": "answered",
                "intent": "admissions",
                "answer": "Answer via guided demo.",
                "citations": [
                    {
                        "chunk_id": "c1",
                        "parent_chunk_id": None,
                        "source_url": "https://adtu.in/admissions",
                        "section": "Admissions",
                        "source_type": "html",
                    }
                ],
                "confidence_status": "high",
                "ticket_id": None,
                "reason": "ok",
            }
        )

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        button = next(b for b in at.button if b.label == DEMO_SCENARIOS[0]["label"])
        button.click().run(timeout=RUN_TIMEOUT)

        self.assertFalse(at.exception)
        sent_query = self.mock_post.call_args.kwargs["json"]["query"]
        self.assertEqual(sent_query, DEMO_SCENARIOS[0]["query"])
        panel_labels = [e.label for e in at.expander]
        self.assertIn("🔎 Why this answer?", panel_labels)

    def test_manual_chat_input_still_works_with_evidence_panel(self) -> None:
        self.mock_post.return_value = _mock_response(
            {
                "status": "answered",
                "intent": "fees",
                "answer": "Manually typed answer with evidence.",
                "citations": [],
                "confidence_status": "high",
                "ticket_id": None,
                "reason": "ok",
            }
        )

        at = self._ask("What is the hostel fee?")

        assistant_texts = [
            m.value
            for cm in at.chat_message
            if cm.name == "assistant"
            for m in cm.markdown
        ]
        self.assertIn("Manually typed answer with evidence.", assistant_texts)
        panel_labels = [e.label for e in at.expander]
        self.assertIn("🔎 Why this answer?", panel_labels)


class TestStaffAdminView(unittest.TestCase):
    """Phase 7C: the sidebar staff/admin ticket queue and Resolve action.

    Every ticket field asserted here comes from the mocked ticket API
    payload -- the panel must never invent staff names, resolution
    timestamps, priority, or SLA data, none of which exist in the model.
    """

    OPEN_TICKET = {
        "ticket_id": "tk-open-1",
        "query": "What is the WiFi password for the boys hostel?",
        "predicted_intent": "facilities",
        "status": "open",
        "created_at": "2026-08-21T10:00:00Z",
        "source": "stage_2_insufficient_evidence",
        "user_metadata": None,
    }

    def setUp(self) -> None:
        self.get_patcher = patch("requests.get")
        self.post_patcher = patch("requests.post")
        self.patch_patcher = patch("requests.patch")
        self.mock_get = self.get_patcher.start()
        self.mock_post = self.post_patcher.start()
        self.mock_patch = self.patch_patcher.start()
        self.addCleanup(self.get_patcher.stop)
        self.addCleanup(self.post_patcher.stop)
        self.addCleanup(self.patch_patcher.stop)

        # Ticket list served to the admin panel; individual tests mutate it.
        self.tickets = [dict(self.OPEN_TICKET)]

        def _get_side_effect(url: str, **kwargs):
            if url.endswith("/ready"):
                return _mock_response(_READY_PAYLOAD)
            if url.endswith("/tickets"):
                return _mock_response(self.tickets)
            return _mock_response({})

        self.mock_get.side_effect = _get_side_effect
        self.mock_patch.return_value = _mock_response(
            {**self.OPEN_TICKET, "status": "resolved"}
        )

    def _load_admin(self) -> AppTest:
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        self.assertFalse(at.exception)
        load_button = next(b for b in at.button if b.key == "admin_load_tickets")
        load_button.click().run(timeout=RUN_TIMEOUT)
        self.assertFalse(at.exception)
        return at

    def test_admin_section_renders(self) -> None:
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)

        self.assertFalse(at.exception)
        sidebar_markdown = [m.value for m in at.sidebar.markdown]
        self.assertTrue(any("Staff / Admin" in m for m in sidebar_markdown))
        self.assertTrue(any(b.key == "admin_load_tickets" for b in at.sidebar.button))

    def test_open_ticket_is_displayed_with_real_fields(self) -> None:
        at = self._load_admin()

        sidebar_markdown = [m.value for m in at.sidebar.markdown]
        sidebar_captions = [c.value for c in at.sidebar.caption]

        # Status is clearly distinguished as OPEN.
        self.assertTrue(any("OPEN" in m for m in sidebar_markdown))
        # Real ticket fields from the API payload.
        self.assertTrue(any(self.OPEN_TICKET["ticket_id"] in c for c in sidebar_captions))
        self.assertTrue(any(self.OPEN_TICKET["query"] in c for c in sidebar_captions))
        self.assertTrue(any(self.OPEN_TICKET["predicted_intent"] in c for c in sidebar_captions))
        self.assertTrue(any(self.OPEN_TICKET["created_at"] in c for c in sidebar_captions))

    def test_open_ticket_offers_resolve_action(self) -> None:
        at = self._load_admin()
        self.assertTrue(any(b.key == f"admin_resolve_{self.OPEN_TICKET['ticket_id']}" for b in at.button))

    def test_resolved_ticket_is_shown_without_resolve_action(self) -> None:
        self.tickets = [{**self.OPEN_TICKET, "status": "resolved"}]

        at = self._load_admin()

        sidebar_markdown = [m.value for m in at.sidebar.markdown]
        self.assertTrue(any("RESOLVED" in m for m in sidebar_markdown))
        # A resolved ticket cannot be re-opened or re-resolved from the UI.
        self.assertFalse(any(b.key.startswith("admin_resolve_") for b in at.button))

    def test_resolve_action_sends_correct_patch_request(self) -> None:
        at = self._load_admin()
        resolve_button = next(b for b in at.button if b.key.startswith("admin_resolve_"))

        resolve_button.click().run(timeout=RUN_TIMEOUT)

        self.assertFalse(at.exception)
        self.mock_patch.assert_called_once()
        called_url = self.mock_patch.call_args.args[0]
        self.assertTrue(called_url.endswith(f"/tickets/{self.OPEN_TICKET['ticket_id']}"))
        self.assertEqual(self.mock_patch.call_args.kwargs["json"], {"status": "resolved"})

    def test_successful_resolution_refreshes_displayed_status(self) -> None:
        at = self._load_admin()
        resolve_button = next(b for b in at.button if b.key.startswith("admin_resolve_"))

        # Backend now reports the ticket as resolved on the refresh GET.
        self.tickets = [{**self.OPEN_TICKET, "status": "resolved"}]
        resolve_button.click().run(timeout=RUN_TIMEOUT)

        self.assertFalse(at.exception)
        sidebar_markdown = [m.value for m in at.sidebar.markdown]
        self.assertTrue(any("RESOLVED" in m for m in sidebar_markdown))
        self.assertFalse(any("OPEN" in m for m in sidebar_markdown))
        # Success feedback is shown to the operator.
        self.assertTrue(any("resolved" in s.value.lower() for s in at.sidebar.success))

    def test_resolve_api_failure_is_displayed_safely(self) -> None:
        at = self._load_admin()
        resolve_button = next(b for b in at.button if b.key.startswith("admin_resolve_"))

        self.mock_patch.return_value = _mock_response(
            {"detail": "Could not update ticket."}, status_code=500
        )
        resolve_button.click().run(timeout=RUN_TIMEOUT)

        self.assertFalse(at.exception)
        sidebar_errors = [e.value for e in at.sidebar.error]
        self.assertTrue(any("Backend internal error" in e for e in sidebar_errors))

    def test_ticket_load_failure_is_displayed_safely(self) -> None:
        import requests as _requests

        def _failing_get(url: str, **kwargs):
            if url.endswith("/ready"):
                return _mock_response(_READY_PAYLOAD)
            raise _requests.exceptions.ConnectionError("connection refused")

        self.mock_get.side_effect = _failing_get

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        next(b for b in at.button if b.key == "admin_load_tickets").click().run(timeout=RUN_TIMEOUT)

        self.assertFalse(at.exception)
        sidebar_errors = [e.value for e in at.sidebar.error]
        self.assertTrue(any("unreachable" in e.lower() for e in sidebar_errors))
        # The raw exception text is never surfaced.
        self.assertFalse(any("connection refused" in e for e in sidebar_errors))

    def test_empty_ticket_queue_is_not_fabricated(self) -> None:
        self.tickets = []

        at = self._load_admin()

        sidebar_captions = [c.value for c in at.sidebar.caption]
        self.assertTrue(any("No escalation tickets yet" in c for c in sidebar_captions))
        self.assertFalse(any(b.key.startswith("admin_resolve_") for b in at.button))

    def test_student_chat_still_works_with_admin_panel(self) -> None:
        self.mock_post.return_value = _mock_response(
            {
                "status": "answered",
                "intent": "fees",
                "answer": "Student-facing answer.",
                "citations": [],
                "confidence_status": "high",
                "ticket_id": None,
                "reason": "ok",
            }
        )

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.chat_input[0].set_value("What is the hostel fee?").run(timeout=RUN_TIMEOUT)

        self.assertFalse(at.exception)
        self.assertEqual(
            self.mock_post.call_args.kwargs["json"]["query"], "What is the hostel fee?"
        )
        assistant_texts = [
            m.value for cm in at.chat_message if cm.name == "assistant" for m in cm.markdown
        ]
        self.assertIn("Student-facing answer.", assistant_texts)

    def test_guided_demo_mode_still_works_with_admin_panel(self) -> None:
        self.mock_post.return_value = _mock_response(
            {
                "status": "answered",
                "intent": "admissions",
                "answer": "Demo answer.",
                "citations": [],
                "confidence_status": "high",
                "ticket_id": None,
                "reason": "ok",
            }
        )

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        next(b for b in at.button if b.label == DEMO_SCENARIOS[0]["label"]).click().run(
            timeout=RUN_TIMEOUT
        )

        self.assertFalse(at.exception)
        self.assertEqual(
            self.mock_post.call_args.kwargs["json"]["query"], DEMO_SCENARIOS[0]["query"]
        )

    def test_trust_evidence_panel_still_works_with_admin_panel(self) -> None:
        self.mock_post.return_value = _mock_response(
            {
                "status": "answered",
                "intent": "fees",
                "answer": "Grounded answer.",
                "citations": [
                    {
                        "chunk_id": "c1",
                        "parent_chunk_id": None,
                        "source_url": "https://adtu.in/fees",
                        "section": "Fees",
                        "source_type": "html",
                    }
                ],
                "confidence_status": "high",
                "ticket_id": None,
                "reason": "dp-top3-mean=0.2000 < 0.275",
            }
        )

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.chat_input[0].set_value("Total fee structure for BCA").run(timeout=RUN_TIMEOUT)

        self.assertFalse(at.exception)
        panel = next(e for e in at.expander if e.label == "🔎 Why this answer?")
        self.assertIn("- [Fees](https://adtu.in/fees)", [m.value for m in panel.markdown])


class TestBackendStatusIndicator(unittest.TestCase):
    """Phase 7D-2: the banner must distinguish offline / alive-but-not-ready /
    ready, driven by /ready rather than the always-green /health ping."""

    def setUp(self) -> None:
        self.get_patcher = patch("requests.get")
        self.post_patcher = patch("requests.post")
        self.mock_get = self.get_patcher.start()
        self.mock_post = self.post_patcher.start()
        self.addCleanup(self.get_patcher.stop)
        self.addCleanup(self.post_patcher.stop)

    def _run(self) -> AppTest:
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        self.assertFalse(at.exception)
        return at

    def test_ready_backend_shows_ready(self) -> None:
        self.mock_get.return_value = _mock_response(_READY_PAYLOAD)

        at = self._run()

        self.assertTrue(any("Backend: Ready" in s.value for s in at.success))
        self.assertFalse(any("Backend" in w.value for w in at.warning))
        self.assertFalse(any("Backend" in e.value for e in at.error))

    def test_alive_but_not_ready_shows_warning_with_failed_checks(self) -> None:
        self.mock_get.return_value = _mock_response(
            {
                "status": "not_ready",
                "checks": {
                    "gemini_api_key": {"ok": False, "detail": "missing or blank"},
                    "chroma": {"ok": True, "detail": "collection accessible with 957 records"},
                },
            },
            status_code=503,
        )

        at = self._run()

        warnings = [w.value for w in at.warning]
        self.assertTrue(any("not ready" in w.lower() for w in warnings))
        # Names the failing dependency so the operator knows what to fix...
        self.assertTrue(any("gemini_api_key" in w for w in warnings))
        # ...but never says the system is fine, and never leaks a key value.
        self.assertFalse(any("Backend: Ready" in s.value for s in at.success))
        self.assertFalse(any("chroma" in w for w in warnings))

    def test_offline_backend_shows_error(self) -> None:
        self.mock_get.side_effect = requests.exceptions.ConnectionError("connection refused")

        at = self._run()

        errors = [e.value for e in at.error]
        self.assertTrue(any("Offline or Unreachable" in e for e in errors))
        self.assertFalse(any("Backend: Ready" in s.value for s in at.success))
        # Raw transport detail is not surfaced to the user.
        self.assertFalse(any("connection refused" in e for e in errors))

    def test_timeout_is_treated_as_offline(self) -> None:
        self.mock_get.side_effect = requests.exceptions.Timeout("timed out")

        at = self._run()

        self.assertTrue(any("Offline or Unreachable" in e.value for e in at.error))

    def test_indicator_queries_the_ready_endpoint(self) -> None:
        self.mock_get.return_value = _mock_response(_READY_PAYLOAD)

        self._run()

        called_urls = [c.args[0] for c in self.mock_get.call_args_list if c.args]
        self.assertTrue(any(u.endswith("/ready") for u in called_urls))

    def test_chat_still_works_with_ready_indicator(self) -> None:
        """The status indicator change must not disturb the chat path."""
        self.mock_get.return_value = _mock_response(_READY_PAYLOAD)
        self.mock_post.return_value = _mock_response(
            {
                "status": "answered",
                "intent": "fees",
                "answer": "Chat unaffected by the readiness indicator.",
                "citations": [],
                "confidence_status": "high",
                "ticket_id": None,
                "reason": "ok",
            }
        )

        at = self._run()
        at.chat_input[0].set_value("What is the hostel fee?").run(timeout=RUN_TIMEOUT)

        self.assertFalse(at.exception)
        self.assertEqual(
            self.mock_post.call_args.kwargs["json"]["query"], "What is the hostel fee?"
        )
        assistant_texts = [
            m.value for cm in at.chat_message if cm.name == "assistant" for m in cm.markdown
        ]
        self.assertIn("Chat unaffected by the readiness indicator.", assistant_texts)

    def test_chat_still_usable_when_backend_not_ready(self) -> None:
        """A not_ready banner must not block or alter the chat workflow."""
        self.mock_get.return_value = _mock_response(
            {
                "status": "not_ready",
                "checks": {"gemini_api_key": {"ok": False, "detail": "missing or blank"}},
            },
            status_code=503,
        )
        self.mock_post.return_value = _mock_response(
            {
                "status": "answered",
                "intent": "fees",
                "answer": "Still answering.",
                "citations": [],
                "confidence_status": "high",
                "ticket_id": None,
                "reason": "ok",
            }
        )

        at = self._run()
        at.chat_input[0].set_value("Fee question").run(timeout=RUN_TIMEOUT)

        self.assertFalse(at.exception)
        assistant_texts = [
            m.value for cm in at.chat_message if cm.name == "assistant" for m in cm.markdown
        ]
        self.assertIn("Still answering.", assistant_texts)


if __name__ == "__main__":
    unittest.main()
