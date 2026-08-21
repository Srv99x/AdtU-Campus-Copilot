"""
Tests for app/rag/pipeline.py — End-to-End Orchestrator Pipeline
"""
from __future__ import annotations

import sqlite3
import unittest
from unittest.mock import MagicMock, patch

from app.rag.pipeline import (
    GATE_MIN_DISTINCT_PARENTS,
    GATE_THETA_D,
    GateMetrics,
    RetrievedChunk,
    RagResult,
    _distinct_parent_chunks,
    _campus_vocab_match,
    _normalize_oos_retrieval_query,
    _is_monetary_scholarship_query,
    _normalize_scholarship_retrieval_query,
    _SCHOLARSHIP_RETRIEVAL_SUFFIX,
    _OOS_RECOVERY_CATEGORY,
    _SCHOLARSHIP_RETRIEVAL_CATEGORY,
    _find_class_routine_sibling_ids,
    _parse_routine_chunk_id,
    _expand_class_routine_evidence,
    assess_retrieval_confidence,
    run_rag_pipeline,
)
from app.rag.generator import GenerationResult, Citation
from app.database.tickets import Ticket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(
    chunk_id: str,
    distance: float,
    parent_chunk_id: str | None = None,
) -> RetrievedChunk:
    meta: dict = {}
    if parent_chunk_id is not None:
        meta["parent_chunk_id"] = parent_chunk_id
    return RetrievedChunk(chunk_id=chunk_id, document="doc", metadata=meta, distance=distance)


def _chroma_response(
    ids: list[str],
    distances: list[float],
    parent_ids: list[str | None] | None = None,
) -> dict:
    """Build a minimal Chroma query response fixture."""
    if parent_ids is None:
        parent_ids = [None] * len(ids)
    metas = [
        {"parent_chunk_id": p} if p else {"category": "admissions"}
        for p in parent_ids
    ]
    return {
        "ids": [ids],
        "documents": [["doc"] * len(ids)],
        "metadatas": [metas],
        "distances": [distances],
    }


# ---------------------------------------------------------------------------
# Test: threshold constant
# ---------------------------------------------------------------------------

class TestGateConstants(unittest.TestCase):
    def test_theta_is_calibrated_value(self) -> None:
        self.assertAlmostEqual(GATE_THETA_D, 0.275, places=6)

    def test_min_distinct_parents_is_three(self) -> None:
        self.assertEqual(GATE_MIN_DISTINCT_PARENTS, 3)


# ---------------------------------------------------------------------------
# Test: _distinct_parent_chunks deduplication
# ---------------------------------------------------------------------------

class TestDistinctParentChunks(unittest.TestCase):
    def test_no_parents_returns_all(self) -> None:
        chunks = [make_chunk("a", 0.1), make_chunk("b", 0.2), make_chunk("c", 0.3)]
        result = _distinct_parent_chunks(chunks)
        self.assertEqual([c.chunk_id for c in result], ["a", "b", "c"])

    def test_derived_children_deduplicated(self) -> None:
        chunks = [
            make_chunk("parent_0", 0.10, parent_chunk_id="parent"),
            make_chunk("parent_1", 0.11, parent_chunk_id="parent"),
            make_chunk("other",    0.20),
        ]
        result = _distinct_parent_chunks(chunks)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].chunk_id, "parent_0")
        self.assertEqual(result[1].chunk_id, "other")

    def test_three_different_parents_all_kept(self) -> None:
        chunks = [
            make_chunk("c1", 0.10, parent_chunk_id="p1"),
            make_chunk("c2", 0.15, parent_chunk_id="p2"),
            make_chunk("c3", 0.20, parent_chunk_id="p3"),
        ]
        result = _distinct_parent_chunks(chunks)
        self.assertEqual(len(result), 3)

    def test_dedup_does_not_distort_mean(self) -> None:
        chunks = [
            make_chunk("p1_c0", 0.10, parent_chunk_id="p1"),
            make_chunk("p1_c1", 0.10, parent_chunk_id="p1"),
            make_chunk("p2",    0.20),
            make_chunk("p3",    0.30),
        ]
        deduped = _distinct_parent_chunks(chunks)[:3]
        self.assertEqual(len(deduped), 3)
        distances = [c.distance for c in deduped]
        self.assertAlmostEqual(sum(distances) / 3, (0.10 + 0.20 + 0.30) / 3, places=6)


# ---------------------------------------------------------------------------
# Test: assess_retrieval_confidence
# ---------------------------------------------------------------------------

class TestAssessRetrievalConfidence(unittest.TestCase):
    def test_empty_chunks_returns_low(self) -> None:
        status, reason, metrics = assess_retrieval_confidence([])
        self.assertEqual(status, "low")
        self.assertEqual(metrics.distinct_parent_count, 0)
        self.assertIsNone(metrics.dp_top3_mean_distance)

    def test_fewer_than_3_distinct_parents_returns_low(self) -> None:
        chunks = [
            make_chunk("p1_c0", 0.10, parent_chunk_id="p1"),
            make_chunk("p1_c1", 0.11, parent_chunk_id="p1"),
            make_chunk("p2",    0.12),
        ]
        status, reason, metrics = assess_retrieval_confidence(chunks)
        self.assertEqual(status, "low")
        self.assertEqual(metrics.distinct_parent_count, 2)
        self.assertIsNone(metrics.dp_top3_mean_distance)

    def test_exactly_3_distinct_parents_below_threshold_returns_high(self) -> None:
        chunks = [
            make_chunk("a", 0.20),
            make_chunk("b", 0.21),
            make_chunk("c", 0.22),
        ]
        status, _, metrics = assess_retrieval_confidence(chunks)
        self.assertEqual(status, "high")
        self.assertAlmostEqual(metrics.dp_top3_mean_distance, (0.20 + 0.21 + 0.22) / 3, places=4)
        self.assertEqual(metrics.distinct_parent_count, 3)
        self.assertAlmostEqual(metrics.threshold, GATE_THETA_D, places=6)

    def test_strong_retrieval_returns_high(self) -> None:
        chunks = [make_chunk(f"c{i}", 0.15 + i * 0.01) for i in range(5)]
        status, _, metrics = assess_retrieval_confidence(chunks)
        self.assertEqual(status, "high")
        self.assertLess(metrics.dp_top3_mean_distance, GATE_THETA_D)

    def test_weak_retrieval_returns_low(self) -> None:
        chunks = [make_chunk(f"c{i}", 0.35 + i * 0.01) for i in range(5)]
        status, _, metrics = assess_retrieval_confidence(chunks)
        self.assertEqual(status, "low")
        self.assertGreaterEqual(metrics.dp_top3_mean_distance, GATE_THETA_D)

    def test_exactly_at_threshold_returns_low(self) -> None:
        chunks = [make_chunk(f"c{i}", GATE_THETA_D) for i in range(3)]
        status, _, _ = assess_retrieval_confidence(chunks)
        self.assertEqual(status, "low")

    def test_derived_children_deduplicated_before_gate(self) -> None:
        chunks = [
            make_chunk("A_c0", 0.15, parent_chunk_id="A"),
            make_chunk("A_c1", 0.15, parent_chunk_id="A"),
            make_chunk("A_c2", 0.15, parent_chunk_id="A"),
            make_chunk("B",    0.30),
            make_chunk("C",    0.35),
        ]
        status, _, metrics = assess_retrieval_confidence(chunks)
        expected_mean = (0.15 + 0.30 + 0.35) / 3  # 0.2667
        self.assertAlmostEqual(metrics.dp_top3_mean_distance, expected_mean, places=4)
        self.assertEqual(status, "high")
        self.assertEqual(metrics.distinct_parent_count, 3)

    def test_dedup_pushes_result_to_low(self) -> None:
        chunks = [
            make_chunk("A_c0", 0.10, parent_chunk_id="A"),
            make_chunk("A_c1", 0.10, parent_chunk_id="A"),
            make_chunk("A_c2", 0.10, parent_chunk_id="A"),
            make_chunk("B",    0.38),
            make_chunk("C",    0.40),
        ]
        status, _, metrics = assess_retrieval_confidence(chunks)
        expected_mean = (0.10 + 0.38 + 0.40) / 3  # 0.2933
        self.assertAlmostEqual(metrics.dp_top3_mean_distance, expected_mean, places=4)
        self.assertEqual(status, "low")


# ---------------------------------------------------------------------------
# Test: run_rag_pipeline integration (Orchestrator End-to-End)
# ---------------------------------------------------------------------------

class TestRunRagPipelineOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_collection = MagicMock()
        self.db_path = "memory:dummy"

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_normal_answerable_query(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        mock_predict.return_value = "admissions"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(["c1", "c2", "c3"], [0.15, 0.18, 0.20])
        
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="B.Tech costs 150k.",
            citations=[Citation("c1", None, "url", "sec", None)]
        )

        result = run_rag_pipeline("how much is btech?", self.mock_collection, self.db_path)

        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer, "B.Tech costs 150k.")
        self.assertEqual(result.confidence_status, "high")
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].chunk_id, "c1")
        self.assertIsNone(result.ticket_id)
        mock_create_ticket.assert_not_called()

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_out_of_scope_short_circuit(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        mock_predict.return_value = "out_of_scope"

        result = run_rag_pipeline("virat kohli centuries", self.mock_collection, self.db_path)

        self.assertTrue(result.is_out_of_scope)
        self.assertEqual(result.status, "out_of_scope")
        self.assertEqual(result.confidence_status, "not_applicable")
        self.assertIsNone(result.ticket_id)
        mock_embed.assert_not_called()
        self.mock_collection.query.assert_not_called()
        mock_generate.assert_not_called()
        mock_create_ticket.assert_not_called()

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_weak_retrieval_creates_ticket(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        mock_predict.return_value = "fees"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(["c1", "c2", "c3"], [0.35, 0.38, 0.40])
        
        mock_create_ticket.return_value = Ticket("tk-123", "q", "fees", "open", "time", "stage_1_low", None)

        result = run_rag_pipeline("very obscure question", self.mock_collection, self.db_path)

        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.confidence_status, "low")
        self.assertEqual(result.ticket_id, "tk-123")
        mock_generate.assert_not_called()
        mock_create_ticket.assert_called_once()
        self.assertEqual(mock_create_ticket.call_args.kwargs["source"], "stage_1_low")

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_strong_retrieval_insufficient_evidence_creates_ticket(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        mock_predict.return_value = "facilities"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(["c1", "c2", "c3"], [0.15, 0.18, 0.20])
        
        mock_generate.return_value = GenerationResult(
            status="insufficient_evidence",
            answer="INSUFFICIENT_EVIDENCE",
            citations=[]
        )
        mock_create_ticket.return_value = Ticket("tk-456", "q", "facilities", "open", "time", "stage_2_insufficient_evidence", None)

        result = run_rag_pipeline("does the library have harry potter?", self.mock_collection, self.db_path)

        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.confidence_status, "high")
        self.assertEqual(result.ticket_id, "tk-456")
        mock_generate.assert_called_once()
        mock_create_ticket.assert_called_once()
        self.assertEqual(mock_create_ticket.call_args.kwargs["source"], "stage_2_insufficient_evidence")

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_derived_child_lineage_preserved(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        mock_predict.return_value = "fees"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20], parent_ids=["p1", "p2", "p3"]
        )
        
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="Answer.",
            citations=[Citation("c1", "p1", "url", "sec", None)]
        )

        result = run_rag_pipeline("question", self.mock_collection, self.db_path)

        self.assertEqual(result.status, "answered")
        self.assertEqual(result.citations[0].parent_chunk_id, "p1")

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_empty_retrieval_escalates(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        mock_predict.return_value = "fees"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response([], [])
        
        mock_create_ticket.return_value = Ticket("tk-789", "q", "fees", "open", "time", "stage_1_low", None)

        result = run_rag_pipeline("question", self.mock_collection, self.db_path)

        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.confidence_status, "low")
        mock_generate.assert_not_called()

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_gemini_failure_returns_error(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        mock_predict.return_value = "admissions"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(["c1", "c2", "c3"], [0.15, 0.18, 0.20])
        
        mock_generate.side_effect = RuntimeError("API down")

        result = run_rag_pipeline("question", self.mock_collection, self.db_path)

        self.assertEqual(result.status, "error")
        self.assertIn("Internal error during grounded generation", result.reason)
        # Raw traceback should NOT be in reason
        self.assertNotIn("API down", result.reason)
        mock_create_ticket.assert_not_called()

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_ticket_creation_failure_returns_error(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        mock_predict.return_value = "admissions"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(["c1", "c2", "c3"], [0.35, 0.38, 0.40])
        
        mock_create_ticket.side_effect = sqlite3.OperationalError("DB locked")

        result = run_rag_pipeline("question", self.mock_collection, self.db_path)

        self.assertEqual(result.status, "error")
        self.assertIn("ticket creation", result.reason)
        self.assertNotIn("DB locked", result.reason)

if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# Policy B — OOS Recovery Tests
# ===========================================================================


class TestCampusVocabMatchPositives(unittest.TestCase):
    """Test 1-3: strong campus terms that must trigger recovery."""

    def test_bihu_holiday_triggers(self) -> None:
        """Test 1: 'Bihu' is a strong term; 'holiday' is also strong."""
        matched, eligible = _campus_vocab_match("When is the Bihu holiday?")
        self.assertTrue(eligible)
        self.assertIn("bihu", matched)
        self.assertIn("holiday", matched)

    def test_hostel_h_block_triggers(self) -> None:
        """Test 2: 'hostel' is strong; 'block' is context-only (co-occurs → eligible)."""
        matched, eligible = _campus_vocab_match("H Block boys hostel seats")
        self.assertTrue(eligible)
        self.assertIn("hostel", matched)  # strong term alone is sufficient

    def test_clubs_societies_triggers(self) -> None:
        """Test 3: 'clubs' and 'societies' are both strong terms."""
        matched, eligible = _campus_vocab_match("What clubs and societies can I join at AdtU?")
        self.assertTrue(eligible)
        # At least one of the strong terms must be present
        strong_in_match = {"club", "clubs", "society", "societies"} & set(matched)
        self.assertTrue(strong_in_match)


class TestCampusVocabMatchNegatives(unittest.TestCase):
    """Tests 4-6: non-campus queries that must NOT trigger recovery."""

    def test_capital_of_france_does_not_trigger(self) -> None:
        """Test 4: Pure geography — no campus vocab."""
        matched, eligible = _campus_vocab_match("Capital of France?")
        self.assertFalse(eligible)

    def test_python_sorting_does_not_trigger(self) -> None:
        """Test 5: Programming question — no campus vocab."""
        matched, eligible = _campus_vocab_match("Python sorting code?")
        self.assertFalse(eligible)

    def test_cricket_world_cup_does_not_trigger(self) -> None:
        """Test 6: Sports news — no campus vocab."""
        matched, eligible = _campus_vocab_match("Cricket world cup result?")
        self.assertFalse(eligible)


class TestCampusVocabContextOnlyRules(unittest.TestCase):
    """Tests 7-8: context-only terms must not trigger alone."""

    def test_adtu_alone_does_not_trigger(self) -> None:
        """Test 7: 'adtu' is context-only; one context-only term is NOT enough."""
        matched, eligible = _campus_vocab_match("What is AdtU?")
        self.assertFalse(eligible)
        self.assertIn("adtu", matched)

    def test_single_context_only_term_does_not_trigger(self) -> None:
        """Test 8: One context-only term without any strong term must not trigger."""
        # 'assam' is context-only; by itself it must NOT trigger recovery
        matched, eligible = _campus_vocab_match("Tell me about Assam.")
        self.assertFalse(eligible)
        self.assertIn("assam", matched)


class TestOosRecoveryPipelineStage1Low(unittest.TestCase):
    """Test 9: OOS + vocab match + Stage 1 LOW → SQLite escalation with oos_recovery source."""

    def setUp(self) -> None:
        self.mock_collection = MagicMock()
        self.db_path = "memory:dummy"

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_oos_recovery_low_confidence_escalates(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """OOS + hostel vocab + weak retrieval → escalated with oos_recovery_stage1_low."""
        mock_predict.return_value = "out_of_scope"
        mock_embed.return_value = [0.1] * 768
        # Weak distances → Stage 1 LOW
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.35, 0.38, 0.40]
        )
        mock_create_ticket.return_value = Ticket(
            "tk-oos-1", "hostel query", "out_of_scope", "open", "now",
            "oos_recovery_stage1_low", None
        )

        result = run_rag_pipeline("hostel fee per semester", self.mock_collection, self.db_path)

        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.intent, "out_of_scope")  # intent preserved
        self.assertEqual(result.ticket_id, "tk-oos-1")
        mock_generate.assert_not_called()  # Stage 2 must NOT be called on LOW
        mock_create_ticket.assert_called_once()
        self.assertEqual(
            mock_create_ticket.call_args.kwargs["source"],
            "oos_recovery_stage1_low",
        )


class TestOosRecoveryPipelineStage2Called(unittest.TestCase):
    """Test 10: OOS + vocab match + Stage 1 HIGH → Stage 2 Gemini is called."""

    def setUp(self) -> None:
        self.mock_collection = MagicMock()
        self.db_path = "memory:dummy"

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_oos_recovery_high_confidence_calls_stage2(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """OOS + holiday vocab + strong retrieval → Stage 2 is called."""
        mock_predict.return_value = "out_of_scope"
        mock_embed.return_value = [0.1] * 768
        # Strong distances → Stage 1 HIGH
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="Bihu holiday is on April 14.",
            citations=[Citation("c1", None, "url", "sec", None)],
        )

        result = run_rag_pipeline("When is the Bihu holiday?", self.mock_collection, self.db_path)

        self.assertEqual(result.status, "answered")
        self.assertEqual(result.intent, "out_of_scope")  # intent still out_of_scope
        self.assertEqual(result.answer, "Bihu holiday is on April 14.")
        mock_generate.assert_called_once()  # Stage 2 was reached
        mock_create_ticket.assert_not_called()


class TestOosRecoveryPipelineStage2Insufficient(unittest.TestCase):
    """Test 11: OOS + vocab + Stage 1 HIGH + Stage 2 INSUFFICIENT_EVIDENCE → escalated."""

    def setUp(self) -> None:
        self.mock_collection = MagicMock()
        self.db_path = "memory:dummy"

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_oos_recovery_insufficient_evidence_escalates(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """OOS + hostel vocab + HIGH gate + INSUFFICIENT_EVIDENCE → oos_recovery_stage2 ticket."""
        mock_predict.return_value = "out_of_scope"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="insufficient_evidence",
            answer="INSUFFICIENT_EVIDENCE",
            citations=[],
        )
        mock_create_ticket.return_value = Ticket(
            "tk-oos-2", "hostel query", "out_of_scope", "open", "now",
            "oos_recovery_stage2_insufficient_evidence", None,
        )

        result = run_rag_pipeline(
            "What is the capacity of the hostel?", self.mock_collection, self.db_path
        )

        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.intent, "out_of_scope")
        self.assertEqual(result.ticket_id, "tk-oos-2")
        mock_generate.assert_called_once()
        mock_create_ticket.assert_called_once()
        self.assertEqual(
            mock_create_ticket.call_args.kwargs["source"],
            "oos_recovery_stage2_insufficient_evidence",
        )


class TestOosRecoveryRetrievelCategory(unittest.TestCase):
    """Test 12: Recovery retrieval uses exactly where={'category':'facilities'}."""

    def setUp(self) -> None:
        self.mock_collection = MagicMock()
        self.db_path = "memory:dummy"

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_recovery_where_filter_is_facilities(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """The Chroma where filter must be exactly {'category': 'facilities'}."""
        mock_predict.return_value = "out_of_scope"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.35, 0.38, 0.40]
        )
        mock_create_ticket.return_value = Ticket(
            "tk-cat", "q", "out_of_scope", "open", "now",
            "oos_recovery_stage1_low", None,
        )

        run_rag_pipeline("hostel capacity", self.mock_collection, self.db_path)

        call_kwargs = self.mock_collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"category": _OOS_RECOVERY_CATEGORY})
        self.assertEqual(_OOS_RECOVERY_CATEGORY, "facilities")


class TestOosRecoveryNoReclassification(unittest.TestCase):
    """Test 13: No recovery loop — classifier is called exactly once."""

    def setUp(self) -> None:
        self.mock_collection = MagicMock()
        self.db_path = "memory:dummy"

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_classifier_called_exactly_once(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """predict_intent must be called exactly once — no reclassification loop."""
        mock_predict.return_value = "out_of_scope"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.35, 0.38, 0.40]
        )
        mock_create_ticket.return_value = Ticket(
            "tk-loop", "q", "out_of_scope", "open", "now",
            "oos_recovery_stage1_low", None,
        )

        run_rag_pipeline("hostel capacity", self.mock_collection, self.db_path)

        mock_predict.assert_called_once()  # exactly one classification, no loop


class TestOosRecoveryIntentPreserved(unittest.TestCase):
    """Test 14: Classifier output out_of_scope is preserved in RagResult.intent."""

    def setUp(self) -> None:
        self.mock_collection = MagicMock()
        self.db_path = "memory:dummy"

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_intent_remains_out_of_scope_after_recovery(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """RagResult.intent must remain 'out_of_scope' even when recovery answers."""
        mock_predict.return_value = "out_of_scope"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="Holiday answer.",
            citations=[Citation("c1", None, "url", "sec", None)],
        )

        result = run_rag_pipeline(
            "When is the Bihu holiday?", self.mock_collection, self.db_path
        )

        self.assertEqual(result.intent, "out_of_scope")  # MUST remain out_of_scope
        self.assertEqual(result.status, "answered")  # status reflects final outcome


class TestNonOosPathsUnchanged(unittest.TestCase):
    """Test 15: Existing non-OOS paths are completely unchanged."""

    def setUp(self) -> None:
        self.mock_collection = MagicMock()
        self.db_path = "memory:dummy"

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_admissions_query_path_unchanged(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """A normal admissions query must follow the original path without any OOS logic."""
        mock_predict.return_value = "admissions"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="BTech admission requires JEE.",
            citations=[Citation("c1", None, "url", "sec", None)],
        )

        result = run_rag_pipeline(
            "How to take admission in BTech?", self.mock_collection, self.db_path
        )

        self.assertEqual(result.status, "answered")
        self.assertEqual(result.intent, "admissions")  # intent is admissions, not out_of_scope
        # Chroma must have been called with the normal intent filter, not facilities
        call_kwargs = self.mock_collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"category": "admissions"})
        mock_predict.assert_called_once()
        mock_generate.assert_called_once()


# ===========================================================================
# Holiday/Calendar Retrieval Normalization (Phase 2)
# ===========================================================================


class TestNormalizeOosRetrievalQueryPure(unittest.TestCase):
    """Tests 1-4: _normalize_oos_retrieval_query as a pure, deterministic helper."""

    def test_bihu_query_appends_academic_calendar(self) -> None:
        """Test 1: Bihu query gets 'academic calendar' appended."""
        result = _normalize_oos_retrieval_query(
            "What is the Bihu holiday?", ["bihu", "holiday"]
        )
        self.assertEqual(result, "What is the Bihu holiday? academic calendar")

    def test_holiday_query_appends_academic_calendar(self) -> None:
        """Test 2: Holiday query gets 'academic calendar' appended."""
        result = _normalize_oos_retrieval_query(
            "When is the next holiday?", ["holiday"]
        )
        self.assertEqual(result, "When is the next holiday? academic calendar")

    def test_no_duplicate_when_academic_calendar_already_present(self) -> None:
        """Test 3: If 'academic calendar' is already present, do not duplicate it."""
        result = _normalize_oos_retrieval_query(
            "Bihu holiday academic calendar", ["bihu", "holiday"]
        )
        self.assertEqual(result, "Bihu holiday academic calendar")

    def test_no_duplicate_case_insensitive(self) -> None:
        """Test 3b: Duplicate check is case-insensitive."""
        result = _normalize_oos_retrieval_query(
            "Bihu Holiday Academic Calendar", ["bihu", "holiday"]
        )
        self.assertEqual(result, "Bihu Holiday Academic Calendar")

    def test_non_holiday_oos_query_unchanged(self) -> None:
        """Test 4: Non-holiday OOS query is unchanged."""
        result = _normalize_oos_retrieval_query(
            "hostel fee per semester", ["hostel", "fee"]
        )
        self.assertEqual(result, "hostel fee per semester")

    def test_no_matched_terms_unchanged(self) -> None:
        """Test 4b: No matched terms at all leaves query unchanged."""
        result = _normalize_oos_retrieval_query("random obscure query", [])
        self.assertEqual(result, "random obscure query")


class TestHolidayNormalizationIntegration(unittest.TestCase):
    """Tests 5-8: end-to-end wiring of the holiday normalization through the pipeline."""

    def setUp(self) -> None:
        self.mock_collection = MagicMock()
        self.db_path = "memory:dummy"

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_normal_inscope_query_not_normalized(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 5: Normal in-scope query (even one containing 'holiday') is unchanged."""
        mock_predict.return_value = "facilities"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="Holiday list is on the notice board.",
            citations=[Citation("c1", None, "url", "sec", None)],
        )

        result = run_rag_pipeline(
            "What is the holiday schedule for hostel?", self.mock_collection, self.db_path
        )

        mock_embed.assert_called_once_with("What is the holiday schedule for hostel?")
        self.assertEqual(result.query, "What is the holiday schedule for hostel?")

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_oos_recovery_embedding_receives_normalized_query(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 6: Only the retrieval embedding call receives the normalized query."""
        mock_predict.return_value = "out_of_scope"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="Bihu holiday is on April 14.",
            citations=[Citation("c1", None, "url", "sec", None)],
        )

        run_rag_pipeline("When is the Bihu holiday?", self.mock_collection, self.db_path)

        mock_embed.assert_called_once_with("When is the Bihu holiday? academic calendar")

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_rag_result_and_generation_preserve_original_query(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 7: RagResult.query, reason, and the Stage 2 call all keep the original query."""
        mock_predict.return_value = "out_of_scope"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="Bihu holiday is on April 14.",
            citations=[Citation("c1", None, "url", "sec", None)],
        )

        result = run_rag_pipeline(
            "When is the Bihu holiday?", self.mock_collection, self.db_path
        )

        self.assertEqual(result.query, "When is the Bihu holiday?")
        self.assertNotIn("academic calendar", result.reason)
        # Stage 2 generation must be grounded on the original user-visible query.
        gen_call_query = mock_generate.call_args[0][0]
        self.assertEqual(gen_call_query, "When is the Bihu holiday?")

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_non_holiday_oos_recovery_logic_unchanged(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 8: Existing recovery + Stage 1/Stage 2 logic is unaffected for non-holiday terms."""
        mock_predict.return_value = "out_of_scope"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.35, 0.38, 0.40]
        )
        mock_create_ticket.return_value = Ticket(
            "tk-oos-8", "hostel fee per semester", "out_of_scope", "open", "now",
            "oos_recovery_stage1_low", None,
        )

        result = run_rag_pipeline(
            "hostel fee per semester", self.mock_collection, self.db_path
        )

        # No normalization applied — embedding call uses the query verbatim.
        mock_embed.assert_called_once_with("hostel fee per semester")
        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.ticket_id, "tk-oos-8")
        mock_generate.assert_not_called()


# ===========================================================================
# Scholarship Monetary Retrieval Override (Phase 3)
# ===========================================================================


class TestIsMonetaryScholarshipQueryPure(unittest.TestCase):
    """Tests 1-7: _is_monetary_scholarship_query as a pure, deterministic helper."""

    def test_cbse_95_percent_scholarship_is_monetary(self) -> None:
        """Test 1: CBSE 95% scholarship is a monetary query."""
        self.assertTrue(_is_monetary_scholarship_query(
            "What scholarship is available for CBSE board students with 95%?"
        ))

    def test_scholarship_percentage_is_monetary(self) -> None:
        """Test 2: 'percentage' signal makes it monetary."""
        self.assertTrue(_is_monetary_scholarship_query(
            "What percentage scholarship can I get?"
        ))

    def test_scholarship_discount_is_monetary(self) -> None:
        """Test 3: 'discount' signal makes it monetary."""
        self.assertTrue(_is_monetary_scholarship_query(
            "How much scholarship discount is available?"
        ))

    def test_fee_waiver_is_monetary(self) -> None:
        """Test 4: 'waiver' signal makes it monetary (fee waiver)."""
        self.assertTrue(_is_monetary_scholarship_query(
            "Is there a fee waiver scholarship for toppers?"
        ))

    def test_topper_scholarship_amount_is_monetary(self) -> None:
        """Test 5: 'amount' signal makes it monetary."""
        self.assertTrue(_is_monetary_scholarship_query("topper scholarship amount"))

    def test_nsp_scholarship_portal_not_monetary(self) -> None:
        """Test 6: application/portal-style scholarship query is NOT monetary."""
        self.assertFalse(_is_monetary_scholarship_query("NSP scholarship portal"))

    def test_scholarship_eligibility_not_monetary(self) -> None:
        """Test 7: eligibility-style scholarship query is NOT monetary."""
        self.assertFalse(_is_monetary_scholarship_query("What is scholarship eligibility?"))

    def test_no_scholarship_vocabulary_not_monetary(self) -> None:
        """No 'scholarship'/'scholarships' term at all -> never monetary, no matter what else is present."""
        self.assertFalse(_is_monetary_scholarship_query("What is the hostel fee discount?"))

    def test_scholarship_documents_not_monetary(self) -> None:
        self.assertFalse(_is_monetary_scholarship_query("scholarship documents required"))

    def test_apply_for_scholarship_not_monetary(self) -> None:
        self.assertFalse(_is_monetary_scholarship_query("How do I apply for NSP scholarship?"))


class TestScholarshipRetrievalIntegration(unittest.TestCase):
    """Tests 8-15: end-to-end wiring of the scholarship monetary override."""

    def setUp(self) -> None:
        self.mock_collection = MagicMock()
        self.db_path = "memory:dummy"

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_unrelated_fees_query_unchanged(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 8: A normal fees query (no scholarship vocabulary) is unaffected."""
        mock_predict.return_value = "fees"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="Semester fee is 85,000.",
            citations=[Citation("c1", None, "url", "sec", None)],
        )

        result = run_rag_pipeline(
            "What is the semester fee?", self.mock_collection, self.db_path
        )

        call_kwargs = self.mock_collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"category": "fees"})
        self.assertEqual(result.intent, "fees")
        self.assertEqual(result.status, "answered")

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_unrelated_facilities_query_unchanged(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 9: A normal facilities query (no scholarship vocabulary) is unaffected."""
        mock_predict.return_value = "facilities"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="Library opens at 8 AM.",
            citations=[Citation("c1", None, "url", "sec", None)],
        )

        result = run_rag_pipeline(
            "What time does the library open?", self.mock_collection, self.db_path
        )

        call_kwargs = self.mock_collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"category": "facilities"})
        self.assertEqual(result.intent, "facilities")
        self.assertEqual(result.status, "answered")

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_classifier_intent_unchanged_when_category_overridden(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 10: RagResult.intent stays the classifier's raw label even when
        the retrieval category is overridden to 'fees'."""
        mock_predict.return_value = "facilities"  # frozen classifier misfires here
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="95% CBSE gets 100% scholarship on semester fee.",
            citations=[Citation("v2_scholarships_0", None, "", "scholarships", "official_pdf")],
        )

        result = run_rag_pipeline(
            "What scholarship is available for CBSE board students with 95%?",
            self.mock_collection, self.db_path,
        )

        # Classifier output is preserved verbatim, even though retrieval used "fees".
        self.assertEqual(result.intent, "facilities")
        call_kwargs = self.mock_collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"category": "fees"})

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_where_filter_becomes_fees_only_for_monetary_scholarship(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 11: Chroma `where` category becomes 'fees' for a monetary
        scholarship query, but stays on the classifier's own category for a
        non-monetary scholarship query (e.g. NSP portal)."""
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered", answer="ok", citations=[Citation("c1", None, "u", "s", None)],
        )

        # Monetary scholarship query, classifier says admissions -> overridden to fees.
        mock_predict.return_value = "admissions"
        run_rag_pipeline(
            "What percentage scholarship can I get?", self.mock_collection, self.db_path
        )
        where_monetary = self.mock_collection.query.call_args.kwargs["where"]
        self.assertEqual(where_monetary, {"category": "fees"})

        # Non-monetary scholarship query, classifier says facilities -> NOT overridden.
        self.mock_collection.reset_mock()
        mock_predict.return_value = "facilities"
        run_rag_pipeline("NSP scholarship portal", self.mock_collection, self.db_path)
        where_nonmonetary = self.mock_collection.query.call_args.kwargs["where"]
        self.assertEqual(where_nonmonetary, {"category": "facilities"})

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_oos_monetary_scholarship_recovery_targets_fees(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 12: OOS recovery redirects to 'fees' for a monetary scholarship query."""
        mock_predict.return_value = "out_of_scope"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="95% CBSE gets 100% scholarship on semester fee.",
            citations=[Citation("v2_scholarships_0", None, "", "scholarships", "official_pdf")],
        )

        result = run_rag_pipeline(
            "Is there any scholarship for 95% marks?", self.mock_collection, self.db_path
        )

        call_kwargs = self.mock_collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"category": _SCHOLARSHIP_RETRIEVAL_CATEGORY})
        self.assertEqual(_SCHOLARSHIP_RETRIEVAL_CATEGORY, "fees")
        self.assertEqual(result.intent, "out_of_scope")  # intent preserved
        self.assertEqual(result.status, "answered")

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_oos_normal_recovery_still_targets_facilities(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 13: Unrelated (non-scholarship) OOS recovery is unaffected — still 'facilities'."""
        mock_predict.return_value = "out_of_scope"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="Hostel capacity is 152.",
            citations=[Citation("c1", None, "u", "s", None)],
        )

        run_rag_pipeline(
            "What is the seat capacity of the boys H Block hostel?",
            self.mock_collection, self.db_path,
        )

        call_kwargs = self.mock_collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"category": _OOS_RECOVERY_CATEGORY})
        self.assertEqual(_OOS_RECOVERY_CATEGORY, "facilities")

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_no_second_embedding_call_for_scholarship_override(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 14: The category override never triggers a second embedding call,
        in either the normal in-scope path or the OOS recovery path."""
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered", answer="ok", citations=[Citation("c1", None, "u", "s", None)],
        )

        # Normal in-scope path.
        mock_predict.return_value = "admissions"
        run_rag_pipeline(
            "What scholarship is available for CBSE board students with 95%?",
            self.mock_collection, self.db_path,
        )
        self.assertEqual(mock_embed.call_count, 1)

        # OOS recovery path.
        mock_embed.reset_mock()
        mock_predict.return_value = "out_of_scope"
        run_rag_pipeline(
            "How much scholarship discount is available?", self.mock_collection, self.db_path
        )
        self.assertEqual(mock_embed.call_count, 1)

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_bihu_normalization_still_intact(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 15: The existing Bihu/holiday retrieval normalization (Phase 2) is
        untouched by the scholarship monetary override (Phase 3)."""
        mock_predict.return_value = "out_of_scope"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="Bihu holiday is on April 14.",
            citations=[Citation("c1", None, "url", "sec", None)],
        )

        result = run_rag_pipeline(
            "When is the Bihu holiday?", self.mock_collection, self.db_path
        )

        # Embedding still gets the holiday-normalized text ...
        mock_embed.assert_called_once_with("When is the Bihu holiday? academic calendar")
        # ... while the recovery category stays "facilities" (no scholarship vocab).
        call_kwargs = self.mock_collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"category": "facilities"})
        # ... and the user-visible query/intent are still preserved.
        self.assertEqual(result.query, "When is the Bihu holiday?")
        self.assertEqual(result.intent, "out_of_scope")


# ===========================================================================
# Scholarship Retrieval-Query Normalization (Phase 4)
# ===========================================================================


class TestNormalizeScholarshipRetrievalQueryPure(unittest.TestCase):
    """Tests 1-10: _normalize_scholarship_retrieval_query as a pure, deterministic helper."""

    def test_cbse_95_percent_scholarship_gets_suffix(self) -> None:
        """Test 1."""
        query = "What scholarship is available for CBSE board students with 95%?"
        is_monetary = _is_monetary_scholarship_query(query)
        self.assertTrue(is_monetary)
        result = _normalize_scholarship_retrieval_query(query, is_monetary)
        self.assertEqual(result, query + " " + _SCHOLARSHIP_RETRIEVAL_SUFFIX)

    def test_scholarship_for_95_percent_marks_gets_suffix(self) -> None:
        """Test 2."""
        query = "Scholarship for 95% marks?"
        is_monetary = _is_monetary_scholarship_query(query)
        self.assertTrue(is_monetary)
        result = _normalize_scholarship_retrieval_query(query, is_monetary)
        self.assertEqual(result, query + " " + _SCHOLARSHIP_RETRIEVAL_SUFFIX)

    def test_scholarship_percentage_for_cbse_95_gets_suffix(self) -> None:
        """Test 3."""
        query = "Scholarship percentage for CBSE 95%"
        is_monetary = _is_monetary_scholarship_query(query)
        self.assertTrue(is_monetary)
        result = _normalize_scholarship_retrieval_query(query, is_monetary)
        self.assertEqual(result, query + " " + _SCHOLARSHIP_RETRIEVAL_SUFFIX)

    def test_how_much_scholarship_for_95_gets_suffix(self) -> None:
        """Test 4."""
        query = "How much scholarship for 95%?"
        is_monetary = _is_monetary_scholarship_query(query)
        self.assertTrue(is_monetary)
        result = _normalize_scholarship_retrieval_query(query, is_monetary)
        self.assertEqual(result, query + " " + _SCHOLARSHIP_RETRIEVAL_SUFFIX)

    def test_scholarship_discount_in_fees_remains_valid(self) -> None:
        """Test 5: already-passing query still gets the (safe, non-regressing) suffix."""
        query = "Scholarship discount in fees"
        is_monetary = _is_monetary_scholarship_query(query)
        self.assertTrue(is_monetary)
        result = _normalize_scholarship_retrieval_query(query, is_monetary)
        self.assertEqual(result, "Scholarship discount in fees scholarship semester fee")

    def test_scholarship_eligibility_gets_no_suffix(self) -> None:
        """Test 6."""
        query = "What is scholarship eligibility?"
        is_monetary = _is_monetary_scholarship_query(query)
        self.assertFalse(is_monetary)
        result = _normalize_scholarship_retrieval_query(query, is_monetary)
        self.assertEqual(result, query)

    def test_nsp_scholarship_portal_gets_no_suffix(self) -> None:
        """Test 7."""
        query = "NSP scholarship portal"
        is_monetary = _is_monetary_scholarship_query(query)
        self.assertFalse(is_monetary)
        result = _normalize_scholarship_retrieval_query(query, is_monetary)
        self.assertEqual(result, query)

    def test_unrelated_hostel_fee_query_gets_no_suffix(self) -> None:
        """Test 8."""
        query = "What is the hostel fee?"
        is_monetary = _is_monetary_scholarship_query(query)
        self.assertFalse(is_monetary)
        result = _normalize_scholarship_retrieval_query(query, is_monetary)
        self.assertEqual(result, query)

    def test_existing_semester_fee_wording_not_duplicated(self) -> None:
        """Test 9."""
        query = "Scholarship discount on semester fee"
        result = _normalize_scholarship_retrieval_query(query, is_monetary=True)
        self.assertEqual(result, query)  # unchanged — no duplicate suffix appended
        self.assertEqual(result.lower().count("semester fee"), 1)

    def test_case_insensitive_duplicate_prevention(self) -> None:
        """Test 10."""
        query = "Scholarship discount on Semester Fee"
        result = _normalize_scholarship_retrieval_query(query, is_monetary=True)
        self.assertEqual(result, query)  # unchanged despite different casing
        self.assertEqual(result.lower().count("semester fee"), 1)

    def test_false_flag_never_appends_regardless_of_text(self) -> None:
        """is_monetary=False must never append the suffix, even if the caller
        passed a scholarship-heavy string — this locks in that the gating
        decision belongs entirely to the caller, not to re-inspection here."""
        query = "scholarship scholarship scholarship 95% 95% 95%"
        result = _normalize_scholarship_retrieval_query(query, is_monetary=False)
        self.assertEqual(result, query)


class TestScholarshipNormalizationIntegration(unittest.TestCase):
    """Tests 11-16: end-to-end wiring of the scholarship retrieval-text normalization."""

    def setUp(self) -> None:
        self.mock_collection = MagicMock()
        self.db_path = "memory:dummy"

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_normalized_string_used_only_for_embedding(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 11: The normalized text goes ONLY to create_query_embedding();
        Chroma's `where` filter and everything else is unaffected by it."""
        mock_predict.return_value = "facilities"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["v2_scholarships_0", "c2", "c3"], [0.20, 0.25, 0.30]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="95% CBSE gets 100% scholarship on semester fee.",
            citations=[Citation("v2_scholarships_0", None, "", "scholarships", "official_pdf")],
        )

        run_rag_pipeline(
            "What scholarship is available for CBSE board students with 95%?",
            self.mock_collection, self.db_path,
        )

        mock_embed.assert_called_once_with(
            "What scholarship is available for CBSE board students with 95%? "
            "scholarship semester fee"
        )
        call_kwargs = self.mock_collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"category": "fees"})

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_ragresult_query_unchanged(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 12: RagResult.query and the Stage 2 call both keep the
        ORIGINAL query, never the normalized retrieval text."""
        mock_predict.return_value = "facilities"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["v2_scholarships_0", "c2", "c3"], [0.20, 0.25, 0.30]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="95% CBSE gets 100% scholarship on semester fee.",
            citations=[Citation("v2_scholarships_0", None, "", "scholarships", "official_pdf")],
        )

        result = run_rag_pipeline(
            "What scholarship is available for CBSE board students with 95%?",
            self.mock_collection, self.db_path,
        )

        self.assertEqual(
            result.query,
            "What scholarship is available for CBSE board students with 95%?",
        )
        gen_call_query = mock_generate.call_args[0][0]
        self.assertEqual(
            gen_call_query,
            "What scholarship is available for CBSE board students with 95%?",
        )

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_bihu_holiday_normalization_still_works(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 13: The Phase 2 Bihu/holiday normalization is unaffected by
        the Phase 4 scholarship suffix (no scholarship vocabulary present,
        so only "academic calendar" is appended)."""
        mock_predict.return_value = "out_of_scope"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="Bihu holiday is on April 14.",
            citations=[Citation("c1", None, "url", "sec", None)],
        )

        result = run_rag_pipeline(
            "When is the Bihu holiday?", self.mock_collection, self.db_path
        )

        mock_embed.assert_called_once_with("When is the Bihu holiday? academic calendar")
        call_kwargs = self.mock_collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"category": "facilities"})
        self.assertEqual(result.query, "When is the Bihu holiday?")

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_oos_recovery_non_scholarship_still_targets_facilities_unnormalized(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 14: A non-scholarship OOS recovery query keeps targeting
        "facilities" with byte-for-byte unchanged retrieval text."""
        mock_predict.return_value = "out_of_scope"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["c1", "c2", "c3"], [0.15, 0.18, 0.20]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="Hostel capacity is 152.",
            citations=[Citation("c1", None, "u", "s", None)],
        )

        run_rag_pipeline(
            "What is the seat capacity of the boys H Block hostel?",
            self.mock_collection, self.db_path,
        )

        mock_embed.assert_called_once_with(
            "What is the seat capacity of the boys H Block hostel?"
        )
        call_kwargs = self.mock_collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"category": _OOS_RECOVERY_CATEGORY})

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_oos_recovery_monetary_scholarship_uses_fees_and_normalized_text(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 15: OOS recovery for a monetary scholarship query targets
        "fees" AND embeds the normalized (suffixed) retrieval text."""
        mock_predict.return_value = "out_of_scope"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["v2_scholarships_0", "c2", "c3"], [0.20, 0.25, 0.30]
        )
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="95% CBSE gets 100% scholarship on semester fee.",
            citations=[Citation("v2_scholarships_0", None, "", "scholarships", "official_pdf")],
        )

        result = run_rag_pipeline(
            "Is there any scholarship for 95% marks?", self.mock_collection, self.db_path
        )

        mock_embed.assert_called_once_with(
            "Is there any scholarship for 95% marks? scholarship semester fee"
        )
        call_kwargs = self.mock_collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"category": _SCHOLARSHIP_RETRIEVAL_CATEGORY})
        self.assertEqual(result.query, "Is there any scholarship for 95% marks?")
        self.assertEqual(result.intent, "out_of_scope")

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_exactly_one_embedding_call(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 16: Regardless of which normalization(s) apply, there is
        never more than one create_query_embedding() call."""
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = _chroma_response(
            ["v2_scholarships_0", "c2", "c3"], [0.20, 0.25, 0.30]
        )
        mock_generate.return_value = GenerationResult(
            status="answered", answer="ok", citations=[Citation("c1", None, "u", "s", None)],
        )

        # Normal in-scope monetary scholarship path.
        mock_predict.return_value = "admissions"
        run_rag_pipeline(
            "What percentage scholarship can I get?", self.mock_collection, self.db_path
        )
        self.assertEqual(mock_embed.call_count, 1)

        # OOS recovery monetary scholarship path.
        mock_embed.reset_mock()
        mock_predict.return_value = "out_of_scope"
        run_rag_pipeline(
            "scholarship discount in fees", self.mock_collection, self.db_path
        )
        self.assertEqual(mock_embed.call_count, 1)

        # OOS recovery Bihu holiday path (stacked normalization: holiday +
        # scholarship normalizers both consulted, still one embedding call).
        mock_embed.reset_mock()
        mock_predict.return_value = "out_of_scope"
        run_rag_pipeline("When is the Bihu holiday?", self.mock_collection, self.db_path)
        self.assertEqual(mock_embed.call_count, 1)


# ===========================================================================
# V2 Class-Routine Sibling Expansion (Phase 5)
# ===========================================================================
#
# Real metadata below is taken verbatim from the live Chroma collection
# (adtu_knowledge), verified via a read-only dump before implementation.
#
# CORRECTION to the earlier read-only audit: it named "_9 <-> _17" as a
# known non-adjacent sibling pair, based only on the `program` field
# matching. Full 4-field verification shows _9 is semester="1st year 1st
# semester" while _17 is semester="2nd year 1st semester" -- they are NOT
# siblings. _9 is in fact self-contained (its own text already includes the
# full heading line) and correctly produces zero sibling matches. Tests
# below verify the CORRECT behavior (real data), not the earlier claim.

_V2_14_HEADING = {
    "topic": "class_routine", "is_table": False,
    "program": "Btech CSE [DS & AI]  IBM (Section A)",
    "specialization": "DS & AI", "semester": "2nd year 1st semester", "section": "A",
}
_V2_15_TABLE = {
    "topic": "class_routine", "is_table": True,
    "program": "Btech CSE [DS & AI]  IBM (Section A)",
    "specialization": "DS & AI", "semester": "2nd year 1st semester", "section": "A",
}
_V2_16_FOOTER = {
    "topic": "class_routine", "is_table": False,
    "program": "Btech CSE [DS & AI]  IBM (Section A)",
    "specialization": "DS & AI", "semester": "2nd year 1st semester", "section": "A",
}
_V2_9_TABLE = {
    "topic": "class_routine", "is_table": True,
    "program": "Btech CSE (Section C)", "specialization": "",
    "semester": "1st year 1st semester", "section": "C",
}
_V2_17_HEADING = {
    "topic": "class_routine", "is_table": False,
    "program": "Btech CSE (Section C)", "specialization": "",
    "semester": "2nd year 1st semester", "section": "C",
}
_V2_18_TABLE = {
    "topic": "class_routine", "is_table": True,
    "program": "Btech CSE (Section C)", "specialization": "",
    "semester": "2nd year 1st semester", "section": "C",
}
# Empty-metadata "orphan" cluster (extraction-gap chunks) -- section falls
# back to the generic document stem "class_routine", which must NOT be
# treated as a real match key.
_V2_36_HEADING_ORPHAN = {
    "topic": "class_routine", "is_table": False,
    "program": "", "specialization": "", "semester": "", "section": "class_routine",
}
_V2_37_TABLE_ORPHAN = {
    "topic": "class_routine", "is_table": True,
    "program": "", "specialization": "", "semester": "", "section": "class_routine",
}
# Fully-populated, UNRELATED section that happens to sit immediately before
# the orphan cluster above (real adjacency in the live corpus: idx 35, 36).
_V2_35_UNRELATED_TABLE = {
    "topic": "class_routine", "is_table": True,
    "program": "Btech CSE [DS & AI]  IBM (Section A)",
    "specialization": "DS & AI", "semester": "3rd year 1st semester", "section": "A",
}
_NON_ROUTINE_CHUNK = {"topic": "fees", "category": "fees"}


class TestParseRoutineChunkId(unittest.TestCase):
    def test_valid_class_routine_id(self) -> None:
        self.assertEqual(_parse_routine_chunk_id("v2_class_routine_15"), ("v2_class_routine", 15))

    def test_non_routine_id_returns_none(self) -> None:
        self.assertIsNone(_parse_routine_chunk_id("v2_scholarships_0"))

    def test_malformed_id_returns_none(self) -> None:
        self.assertIsNone(_parse_routine_chunk_id("v2_class_routine_"))
        self.assertIsNone(_parse_routine_chunk_id("class_routine_15"))


class TestFindClassRoutineSiblingIds(unittest.TestCase):
    """Pure pairing logic — no Chroma or Gemini access."""

    def test_14_pairs_with_15_via_metadata(self) -> None:
        """Test 1: v2_class_routine_14 <-> v2_class_routine_15 via strong metadata match."""
        candidates = [
            ("v2_class_routine_15", _V2_15_TABLE),
            ("v2_class_routine_16", _V2_16_FOOTER),
            ("v2_class_routine_17", _V2_17_HEADING),  # different section — must not match
        ]
        siblings = _find_class_routine_sibling_ids("v2_class_routine_14", _V2_14_HEADING, candidates)
        self.assertIn("v2_class_routine_15", siblings)
        self.assertIn("v2_class_routine_16", siblings)
        self.assertNotIn("v2_class_routine_17", siblings)

    def test_metadata_matching_works_across_non_adjacent_indices(self) -> None:
        """Test 2 (corrected): metadata-tier matching is index-distance-agnostic —
        verified with a synthetic far-apart pair sharing full metadata, since the
        real corpus's only same-program-string pair (_9/_17) turned out to differ
        on `semester` (see module-level correction note) and is NOT a real sibling
        pair — that negative case is covered separately below."""
        far_apart_sibling = {
            "topic": "class_routine", "is_table": True,
            "program": "Btech CSE [DS & AI]  IBM (Section A)",
            "specialization": "DS & AI", "semester": "2nd year 1st semester", "section": "A",
        }
        candidates = [("v2_class_routine_88", far_apart_sibling)]
        siblings = _find_class_routine_sibling_ids("v2_class_routine_14", _V2_14_HEADING, candidates)
        self.assertEqual(siblings, ["v2_class_routine_88"])

    def test_9_and_17_do_not_pair_semester_mismatch(self) -> None:
        """Corrects the earlier audit: _9 (1st year 1st semester) and _17 (2nd
        year 1st semester) share `program`+`section` but differ on `semester` —
        they must NOT be treated as siblings."""
        candidates = [("v2_class_routine_17", _V2_17_HEADING)]
        siblings = _find_class_routine_sibling_ids("v2_class_routine_9", _V2_9_TABLE, candidates)
        self.assertEqual(siblings, [])

    def test_incomplete_metadata_uses_bounded_adjacency_fallback(self) -> None:
        """Test 3: orphan cluster (_36/_37, both empty metadata) pairs via the
        bounded chunk_index adjacency fallback, not via metadata matching."""
        candidates = [
            ("v2_class_routine_37", _V2_37_TABLE_ORPHAN),
            ("v2_class_routine_35", _V2_35_UNRELATED_TABLE),  # must NOT match
        ]
        siblings = _find_class_routine_sibling_ids(
            "v2_class_routine_36", _V2_36_HEADING_ORPHAN, candidates
        )
        self.assertEqual(siblings, ["v2_class_routine_37"])

    def test_unrelated_sections_do_not_cross_pair(self) -> None:
        """Test 4: an empty-metadata orphan adjacent to a fully-populated,
        unrelated section must NOT inherit that section's data (regression
        guard for the v2_class_routine_36 -> _35 contamination case found
        during implementation)."""
        candidates = [("v2_class_routine_35", _V2_35_UNRELATED_TABLE)]
        siblings = _find_class_routine_sibling_ids(
            "v2_class_routine_36", _V2_36_HEADING_ORPHAN, candidates
        )
        self.assertEqual(siblings, [])

    def test_generic_section_fallback_value_not_used_as_match_key(self) -> None:
        """Two orphan chunks sharing only the generic section="class_routine"
        fallback value (not a real section label) must not match on that
        basis alone if nothing else overlaps and they are not adjacent."""
        far_orphan = dict(_V2_37_TABLE_ORPHAN)
        candidates = [("v2_class_routine_99", far_orphan)]
        siblings = _find_class_routine_sibling_ids(
            "v2_class_routine_36", _V2_36_HEADING_ORPHAN, candidates
        )
        self.assertEqual(siblings, [])  # not adjacent, so fallback tier also rejects it

    def test_non_class_routine_anchor_returns_empty(self) -> None:
        """Test 5 (pure half): a non-class_routine chunk never yields siblings."""
        siblings = _find_class_routine_sibling_ids(
            "v2_scholarships_0", _NON_ROUTINE_CHUNK, [("v2_class_routine_14", _V2_14_HEADING)]
        )
        self.assertEqual(siblings, [])

    def test_self_never_returned_as_own_sibling(self) -> None:
        candidates = [("v2_class_routine_14", _V2_14_HEADING), ("v2_class_routine_15", _V2_15_TABLE)]
        siblings = _find_class_routine_sibling_ids("v2_class_routine_14", _V2_14_HEADING, candidates)
        self.assertNotIn("v2_class_routine_14", siblings)


class TestExpandClassRoutineEvidence(unittest.TestCase):
    """Tests the I/O-orchestrating wrapper with a mocked Chroma collection."""

    def _make_collection_with_routine_pool(self) -> MagicMock:
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["v2_class_routine_14", "v2_class_routine_15", "v2_class_routine_16"],
            "documents": ["heading text", "table text with faculty/times", "footer text"],
            "metadatas": [_V2_14_HEADING, _V2_15_TABLE, _V2_16_FOOTER],
        }
        return mock_collection

    def test_non_class_routine_chunks_never_trigger_expansion(self) -> None:
        """Test 5 (integration half): unrelated retrieved chunks never call
        collection.get() and the list is returned unchanged."""
        mock_collection = MagicMock()
        chunks = [make_chunk("v2_scholarships_0", 0.2)]
        chunks[0].metadata["topic"] = "fees"

        result = _expand_class_routine_evidence(chunks, mock_collection)

        self.assertEqual(result, chunks)
        mock_collection.get.assert_not_called()

    def test_original_retrieved_chunks_remain_present(self) -> None:
        """Test 9: original retrieved chunks are all still present after expansion."""
        mock_collection = self._make_collection_with_routine_pool()
        original = RetrievedChunk(
            chunk_id="v2_class_routine_14", document="heading text",
            metadata=_V2_14_HEADING, distance=0.14,
        )
        result = _expand_class_routine_evidence([original], mock_collection)
        self.assertIn(original, result)

    def test_sibling_chunks_have_real_ids_and_metadata(self) -> None:
        """Test 10: expanded chunks carry real chunk_id/metadata/document —
        never fabricated."""
        mock_collection = self._make_collection_with_routine_pool()
        anchor = RetrievedChunk(
            chunk_id="v2_class_routine_14", document="heading text",
            metadata=_V2_14_HEADING, distance=0.14,
        )
        result = _expand_class_routine_evidence([anchor], mock_collection)
        result_ids = {c.chunk_id for c in result}
        self.assertEqual(result_ids, {"v2_class_routine_14", "v2_class_routine_15", "v2_class_routine_16"})

        table_chunk = next(c for c in result if c.chunk_id == "v2_class_routine_15")
        self.assertEqual(table_chunk.document, "table text with faculty/times")
        self.assertEqual(table_chunk.metadata, _V2_15_TABLE)
        self.assertIsNone(table_chunk.metadata.get("parent_chunk_id"))

    def test_no_duplicate_chunks_introduced(self) -> None:
        """Test 11: if the sibling was already retrieved, it is not re-added."""
        mock_collection = self._make_collection_with_routine_pool()
        anchor = RetrievedChunk(
            chunk_id="v2_class_routine_14", document="heading text",
            metadata=_V2_14_HEADING, distance=0.14,
        )
        already_present = RetrievedChunk(
            chunk_id="v2_class_routine_15", document="table text with faculty/times",
            metadata=_V2_15_TABLE, distance=0.30,
        )
        result = _expand_class_routine_evidence([anchor, already_present], mock_collection)
        ids = [c.chunk_id for c in result]
        self.assertEqual(len(ids), len(set(ids)))  # no duplicates
        self.assertEqual(ids.count("v2_class_routine_15"), 1)

    def test_single_metadata_only_fetch_no_embedding(self) -> None:
        """Expansion performs exactly one collection.get() call and never
        touches an embedding function."""
        mock_collection = self._make_collection_with_routine_pool()
        anchor = RetrievedChunk(
            chunk_id="v2_class_routine_14", document="heading text",
            metadata=_V2_14_HEADING, distance=0.14,
        )
        _expand_class_routine_evidence([anchor], mock_collection)
        mock_collection.get.assert_called_once()
        call_kwargs = mock_collection.get.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"topic": "class_routine"})


class TestClassRoutineGateIsolation(unittest.TestCase):
    """Tests 6-8, 12-13: sibling expansion must never influence Stage 1, and
    the query/embedding-call contract must hold exactly as in prior phases."""

    def setUp(self) -> None:
        self.mock_collection = MagicMock()
        self.db_path = "memory:dummy"

    def _routine_chroma_response(self) -> dict:
        # Only the HEADING chunk is retrieved by Chroma — this mirrors the
        # confirmed live behavior (table chunk never appears in top-10).
        return {
            "ids": [["v2_class_routine_14", "v2_class_routine_0", "v2_class_routine_4"]],
            "documents": [["heading A", "heading other 1", "heading other 2"]],
            "metadatas": [[_V2_14_HEADING, {"topic": "class_routine"}, {"topic": "class_routine"}]],
            "distances": [[0.14, 0.17, 0.18]],
        }

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_expansion_does_not_alter_gate_metrics(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 6: gate_metrics reflect the ORIGINAL 3-chunk set (HIGH,
        dp-top3-mean computed from the 3 heading distances) regardless of
        how many sibling chunks get added for Stage 2."""
        mock_predict.return_value = "facilities"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = self._routine_chroma_response()
        self.mock_collection.get.return_value = {
            "ids": ["v2_class_routine_14", "v2_class_routine_15", "v2_class_routine_16"],
            "documents": ["heading A", "table A", "footer A"],
            "metadatas": [_V2_14_HEADING, _V2_15_TABLE, _V2_16_FOOTER],
        }
        mock_generate.return_value = GenerationResult(
            status="answered", answer="Timetable answer.",
            citations=[Citation("v2_class_routine_15", None, "", "A", "official_pdf")],
        )

        result = run_rag_pipeline(
            "What is the class routine for B.Tech CSE DS & AI IBM 2nd year 1st semester?",
            self.mock_collection, self.db_path,
        )

        expected_mean = round((0.14 + 0.17 + 0.18) / 3, 6)
        self.assertEqual(result.gate_metrics.dp_top3_mean_distance, expected_mean)
        self.assertEqual(result.gate_metrics.distinct_parent_count, 3)
        self.assertEqual(result.confidence_status, "high")
        # retrieved_chunks reported to the caller are the ORIGINAL 3, not the expanded set.
        self.assertEqual(len(result.retrieved_chunks), 3)

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_low_original_retrieval_remains_low_after_expansion(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 7: a weak (LOW) original retrieval stays LOW and escalates —
        expansion must never be used to rescue it. generate_grounded_answer
        (and therefore expansion) must not even be called."""
        mock_predict.return_value = "facilities"
        mock_embed.return_value = [0.1] * 768
        weak_response = self._routine_chroma_response()
        weak_response["distances"] = [[0.35, 0.38, 0.40]]  # well above GATE_THETA_D
        self.mock_collection.query.return_value = weak_response
        mock_create_ticket.return_value = Ticket(
            "tk-routine-low", "q", "facilities", "open", "now", "stage_1_low", None,
        )

        result = run_rag_pipeline(
            "What is the class routine for B.Tech CSE DS & AI IBM 2nd year 1st semester?",
            self.mock_collection, self.db_path,
        )

        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.confidence_status, "low")
        mock_generate.assert_not_called()
        self.mock_collection.get.assert_not_called()  # expansion never even attempted

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_high_original_retrieval_proceeds_with_enriched_evidence(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 8: a HIGH original retrieval proceeds to Stage 2 with the
        expanded evidence set (verified via the actual call_args passed to
        generate_grounded_answer)."""
        mock_predict.return_value = "facilities"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = self._routine_chroma_response()
        self.mock_collection.get.return_value = {
            "ids": ["v2_class_routine_14", "v2_class_routine_15", "v2_class_routine_16"],
            "documents": ["heading A", "table A", "footer A"],
            "metadatas": [_V2_14_HEADING, _V2_15_TABLE, _V2_16_FOOTER],
        }
        mock_generate.return_value = GenerationResult(
            status="answered", answer="Timetable answer.",
            citations=[Citation("v2_class_routine_15", None, "", "A", "official_pdf")],
        )

        run_rag_pipeline(
            "What is the class routine for B.Tech CSE DS & AI IBM 2nd year 1st semester?",
            self.mock_collection, self.db_path,
        )

        evidence_passed = mock_generate.call_args[0][1]
        evidence_ids = {c.chunk_id for c in evidence_passed}
        self.assertIn("v2_class_routine_15", evidence_ids)  # the table chunk reached Stage 2

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_original_user_query_unchanged(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 12: RagResult.query and the Stage 2 query argument are both
        the original, untouched user query."""
        mock_predict.return_value = "facilities"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = self._routine_chroma_response()
        self.mock_collection.get.return_value = {
            "ids": ["v2_class_routine_14", "v2_class_routine_15"],
            "documents": ["heading A", "table A"],
            "metadatas": [_V2_14_HEADING, _V2_15_TABLE],
        }
        mock_generate.return_value = GenerationResult(
            status="answered", answer="ok", citations=[],
        )

        query = "What is the class routine for B.Tech CSE DS & AI IBM 2nd year 1st semester?"
        result = run_rag_pipeline(query, self.mock_collection, self.db_path)

        self.assertEqual(result.query, query)
        gen_call_query = mock_generate.call_args[0][0]
        self.assertEqual(gen_call_query, query)

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_exactly_one_embedding_call_with_expansion(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Test 13: sibling expansion (a metadata-only collection.get() call)
        never triggers a second embedding call."""
        mock_predict.return_value = "facilities"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = self._routine_chroma_response()
        self.mock_collection.get.return_value = {
            "ids": ["v2_class_routine_14", "v2_class_routine_15"],
            "documents": ["heading A", "table A"],
            "metadatas": [_V2_14_HEADING, _V2_15_TABLE],
        }
        mock_generate.return_value = GenerationResult(status="answered", answer="ok", citations=[])

        run_rag_pipeline(
            "What is the class routine for B.Tech CSE DS & AI IBM 2nd year 1st semester?",
            self.mock_collection, self.db_path,
        )
        self.assertEqual(mock_embed.call_count, 1)


class TestClassRoutineHeadingOnlyIntegration(unittest.TestCase):
    """Full integration test: a realistic heading-only retrieval must still
    deliver the timetable text (and routine metadata) to Gemini."""

    def setUp(self) -> None:
        self.mock_collection = MagicMock()
        self.db_path = "memory:dummy"

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_heading_only_retrieval_still_delivers_timetable_to_stage2(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        mock_predict.return_value = "facilities"
        mock_embed.return_value = [0.1] * 768

        # Realistic top-10: the DS&AI IBM Section A heading at rank 1, plus
        # 9 OTHER sections' headings/tables -- exactly the failure mode
        # confirmed live (table chunk never appears in top-10).
        other_ids = [f"v2_class_routine_{i}" for i in [0, 4, 10, 17, 21, 27, 30, 33, 44]]
        other_metas = [{"topic": "class_routine", "is_table": False} for _ in other_ids]
        self.mock_collection.query.return_value = {
            "ids": [["v2_class_routine_14"] + other_ids],
            "documents": [["heading A"] + [f"heading {i}" for i in other_ids]],
            "metadatas": [[_V2_14_HEADING] + other_metas],
            "distances": [[0.1425, 0.1697, 0.1721, 0.1724, 0.1844, 0.1858, 0.1906, 0.1949, 0.1975, 0.2065]],
        }
        self.mock_collection.get.return_value = {
            "ids": ["v2_class_routine_14", "v2_class_routine_15", "v2_class_routine_16"],
            "documents": [
                "# Faculty of Computer Technology... Programme: Btech CSE [DS & AI] IBM (Section A)",
                "|D\\T|9:00-10:00|... Dr. Dipjyoti Nath CN: Probability and Statistics ...",
                "**CC: Course Code**",
            ],
            "metadatas": [_V2_14_HEADING, _V2_15_TABLE, _V2_16_FOOTER],
        }
        mock_generate.return_value = GenerationResult(
            status="answered",
            answer="The routine includes Probability and Statistics on Monday...",
            citations=[Citation("v2_class_routine_15", None, "", "A", "official_pdf")],
        )

        result = run_rag_pipeline(
            "What is the class routine for B.Tech CSE DS & AI IBM 2nd year 1st semester?",
            self.mock_collection, self.db_path,
        )

        # Gate metrics computed from the ORIGINAL (unexpanded) 10-chunk set.
        expected_mean = round((0.1425 + 0.1697 + 0.1721) / 3, 6)
        self.assertEqual(result.gate_metrics.dp_top3_mean_distance, expected_mean)
        self.assertEqual(result.confidence_status, "high")
        self.assertEqual(len(result.retrieved_chunks), 10)  # unexpanded

        # Evidence actually sent to Gemini includes the TABLE chunk's real content.
        evidence_passed = mock_generate.call_args[0][1]
        evidence_ids = {c.chunk_id for c in evidence_passed}
        self.assertIn("v2_class_routine_15", evidence_ids)
        table_evidence = next(c for c in evidence_passed if c.chunk_id == "v2_class_routine_15")
        self.assertIn("Probability and Statistics", table_evidence.document)
        # Routine metadata travels with the table chunk for generator.py to expose.
        self.assertEqual(table_evidence.metadata["program"], "Btech CSE [DS & AI]  IBM (Section A)")
        self.assertEqual(table_evidence.metadata["semester"], "2nd year 1st semester")

        self.assertEqual(result.status, "answered")


# ===========================================================================
# Phase 5.1 — Finding B: rank-1-only sibling expansion
# ===========================================================================
#
# These tests use a REALISTIC multi-anchor pool. The Phase 5 tests above all
# mocked a 2-3 chunk routine pool containing only the _14/_15/_16 group, so
# multi-anchor fan-out was structurally invisible to them (audit Finding C).
# Every metadata value below is verbatim from the live Chroma collection.

def _routine_meta(
    program: str = "", specialization: str = "", semester: str = "",
    section: str = "class_routine", is_table: bool = False,
) -> dict:
    return {
        "topic": "class_routine", "is_table": is_table, "program": program,
        "specialization": specialization, "semester": semester, "section": section,
    }


# Six real timetable groups from the live corpus, each heading + table(s).
_DSAI_A_2ND = dict(program="Btech CSE [DS & AI]  IBM (Section A)",
                   specialization="DS & AI", semester="2nd year 1st semester", section="A")
_DSAI_A_1ST = dict(program="Btech CSE [DS & AI]  IBM (Section A)",
                   specialization="DS & AI", semester="1st year 1st semester", section="A")
_DSAI_A_3RD = dict(program="Btech CSE [DS & AI]  IBM (Section A)",
                   specialization="DS & AI", semester="3rd year 1st semester", section="A")
_DSAI_B_1ST = dict(program="Btech CSE [DS & AI] IBM (Section B)",
                   specialization="DS & AI", semester="1st year 1st semester", section="B")
_CSE_C_2ND = dict(program="Btech CSE (Section C)", specialization="",
                  semester="2nd year 1st semester", section="C")
_SAP_D_2ND = dict(program="Btech CSE + Btech CSE SAP (Section D)", specialization="",
                  semester="2nd year 1st semester", section="D")

# Full realistic routine pool: 6 groups, 17 chunks.
_REALISTIC_ROUTINE_POOL: dict[str, dict] = {
    # rank-1 target group (DS&AI IBM Section A, 2nd year 1st sem)
    "v2_class_routine_14": _routine_meta(**_DSAI_A_2ND, is_table=False),
    "v2_class_routine_15": _routine_meta(**_DSAI_A_2ND, is_table=True),
    "v2_class_routine_16": _routine_meta(**_DSAI_A_2ND, is_table=False),
    # other groups — must NOT contribute siblings
    "v2_class_routine_0": _routine_meta(**_DSAI_A_1ST, is_table=False),
    "v2_class_routine_1": _routine_meta(**_DSAI_A_1ST, is_table=True),
    "v2_class_routine_2": _routine_meta(**_DSAI_A_1ST, is_table=True),
    "v2_class_routine_3": _routine_meta(**_DSAI_A_1ST, is_table=False),
    "v2_class_routine_4": _routine_meta(**_DSAI_B_1ST, is_table=False),
    "v2_class_routine_5": _routine_meta(**_DSAI_B_1ST, is_table=True),
    "v2_class_routine_6": _routine_meta(**_DSAI_B_1ST, is_table=True),
    "v2_class_routine_17": _routine_meta(**_CSE_C_2ND, is_table=False),
    "v2_class_routine_18": _routine_meta(**_CSE_C_2ND, is_table=True),
    "v2_class_routine_19": _routine_meta(**_CSE_C_2ND, is_table=True),
    "v2_class_routine_21": _routine_meta(**_SAP_D_2ND, is_table=False),
    "v2_class_routine_22": _routine_meta(**_SAP_D_2ND, is_table=True),
    "v2_class_routine_33": _routine_meta(**_DSAI_A_3RD, is_table=False),
    "v2_class_routine_34": _routine_meta(**_DSAI_A_3RD, is_table=True),
}

# Realistic retrieval: rank-1 is the target heading _14, followed by NINE
# headings belonging to OTHER timetables — the exact live failure shape.
_REALISTIC_RETRIEVAL_IDS = [
    "v2_class_routine_14",  # rank 1 — target
    "v2_class_routine_0", "v2_class_routine_4", "v2_class_routine_17",
    "v2_class_routine_21", "v2_class_routine_33",
]
_REALISTIC_RETRIEVAL_DISTANCES = [0.1425, 0.1697, 0.1721, 0.1858, 0.1975, 0.1844]


def _realistic_pool_get_return() -> dict:
    ids = list(_REALISTIC_ROUTINE_POOL.keys())
    return {
        "ids": ids,
        "documents": [
            "TIMETABLE-GRID Probability and Statistics" if _REALISTIC_ROUTINE_POOL[i]["is_table"]
            else f"heading text for {i}"
            for i in ids
        ],
        "metadatas": [_REALISTIC_ROUTINE_POOL[i] for i in ids],
    }


class TestRank1OnlyExpansion(unittest.TestCase):
    """Finding B regression suite — realistic multi-anchor pool."""

    def setUp(self) -> None:
        self.mock_collection = MagicMock()
        self.mock_collection.get.return_value = _realistic_pool_get_return()
        self.db_path = "memory:dummy"
        self.original = [
            RetrievedChunk(cid, f"doc {cid}", _REALISTIC_ROUTINE_POOL[cid], dist)
            for cid, dist in zip(_REALISTIC_RETRIEVAL_IDS, _REALISTIC_RETRIEVAL_DISTANCES)
        ]

    def test_only_rank1_anchor_expands(self) -> None:
        """Test 1: only the rank-1 routine anchor (_14) contributes siblings."""
        expanded = _expand_class_routine_evidence(self.original, self.mock_collection)
        added = [c.chunk_id for c in expanded if c.chunk_id not in _REALISTIC_RETRIEVAL_IDS]
        # _14's real siblings, and nothing else.
        self.assertEqual(sorted(added), ["v2_class_routine_15", "v2_class_routine_16"])

    def test_lower_ranked_anchors_present_but_contribute_no_siblings(self) -> None:
        """Test 2: every lower-ranked routine anchor is still in evidence, but
        none of THEIR siblings were pulled in."""
        expanded = _expand_class_routine_evidence(self.original, self.mock_collection)
        expanded_ids = {c.chunk_id for c in expanded}

        # all originals still present
        for cid in _REALISTIC_RETRIEVAL_IDS:
            self.assertIn(cid, expanded_ids)

        # siblings of the NINE lower-ranked anchors must be absent
        forbidden = {
            "v2_class_routine_1", "v2_class_routine_2", "v2_class_routine_3",  # _0's group
            "v2_class_routine_5", "v2_class_routine_6",                        # _4's group
            "v2_class_routine_18", "v2_class_routine_19",                      # _17's group
            "v2_class_routine_22",                                             # _21's group
            "v2_class_routine_34",                                             # _33's group
        }
        self.assertEqual(forbidden & expanded_ids, set())

    def test_target_table_still_reaches_stage2(self) -> None:
        """Test 3: _15 (the answer-bearing timetable) still reaches evidence."""
        expanded = _expand_class_routine_evidence(self.original, self.mock_collection)
        ids = [c.chunk_id for c in expanded]
        self.assertIn("v2_class_routine_15", ids)
        table = next(c for c in expanded if c.chunk_id == "v2_class_routine_15")
        self.assertIn("TIMETABLE-GRID", table.document)
        self.assertEqual(table.metadata["program"], "Btech CSE [DS & AI]  IBM (Section A)")
        self.assertEqual(table.metadata["semester"], "2nd year 1st semester")

    def test_stage1_metrics_identical_before_and_after_expansion(self) -> None:
        """Test 4: gate metrics computed from the original set are byte-identical
        before and after expansion runs, and the original list is not mutated."""
        status_before, _, metrics_before = assess_retrieval_confidence(self.original)
        snapshot = [c.chunk_id for c in self.original]

        _expand_class_routine_evidence(self.original, self.mock_collection)

        status_after, _, metrics_after = assess_retrieval_confidence(self.original)
        self.assertEqual(status_before, status_after)
        self.assertEqual(metrics_before, metrics_after)
        self.assertAlmostEqual(metrics_after.threshold, GATE_THETA_D, places=6)
        # original list untouched (length and order)
        self.assertEqual([c.chunk_id for c in self.original], snapshot)

    def test_no_cross_section_semester_or_program_contamination(self) -> None:
        """Test 5: every ADDED chunk shares the rank-1 anchor's program,
        specialization, semester and section exactly."""
        expanded = _expand_class_routine_evidence(self.original, self.mock_collection)
        anchor_meta = _REALISTIC_ROUTINE_POOL["v2_class_routine_14"]
        added = [c for c in expanded if c.chunk_id not in _REALISTIC_RETRIEVAL_IDS]

        self.assertTrue(added, "expected at least one sibling to be added")
        for chunk in added:
            for field in ("program", "specialization", "semester", "section"):
                self.assertEqual(
                    chunk.metadata.get(field), anchor_meta.get(field),
                    f"{chunk.chunk_id} differs from anchor on {field}",
                )

    def test_evidence_set_size_is_bounded(self) -> None:
        """Fan-out regression guard: 6 routine anchors must yield 6 + 2 = 8
        chunks, not the pre-fix 6 + 11."""
        expanded = _expand_class_routine_evidence(self.original, self.mock_collection)
        self.assertEqual(len(expanded), len(_REALISTIC_RETRIEVAL_IDS) + 2)
        ids = [c.chunk_id for c in expanded]
        self.assertEqual(len(ids), len(set(ids)))  # still no duplicates

    def test_only_one_distinct_timetable_is_assembled(self) -> None:
        """The whole point of Finding B: exactly ONE timetable ends up fully
        assembled (heading + table), instead of six."""
        expanded = _expand_class_routine_evidence(self.original, self.mock_collection)
        table_groups = {
            (c.metadata.get("program"), c.metadata.get("semester"))
            for c in expanded if c.metadata.get("is_table")
        }
        self.assertEqual(
            table_groups,
            {("Btech CSE [DS & AI]  IBM (Section A)", "2nd year 1st semester")},
        )

    @patch("app.rag.pipeline.create_ticket")
    @patch("app.rag.pipeline.generate_grounded_answer")
    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_end_to_end_rank1_only_through_pipeline(
        self, mock_predict, mock_embed, mock_generate, mock_create_ticket
    ) -> None:
        """Full pipeline: gate uses the original 6, Stage 2 receives 8."""
        mock_predict.return_value = "facilities"
        mock_embed.return_value = [0.1] * 768
        self.mock_collection.query.return_value = {
            "ids": [_REALISTIC_RETRIEVAL_IDS],
            "documents": [[f"doc {c}" for c in _REALISTIC_RETRIEVAL_IDS]],
            "metadatas": [[_REALISTIC_ROUTINE_POOL[c] for c in _REALISTIC_RETRIEVAL_IDS]],
            "distances": [_REALISTIC_RETRIEVAL_DISTANCES],
        }
        mock_generate.return_value = GenerationResult(
            status="answered", answer="Timetable answer.",
            citations=[Citation("v2_class_routine_15", None, "", "A", "official_pdf")],
        )

        result = run_rag_pipeline(
            "What is the class routine for B.Tech CSE DS & AI IBM 2nd year 1st semester?",
            self.mock_collection, self.db_path,
        )

        # gate from the ORIGINAL retrieval only
        expected_mean = round((0.1425 + 0.1697 + 0.1721) / 3, 6)
        self.assertEqual(result.gate_metrics.dp_top3_mean_distance, expected_mean)
        self.assertEqual(len(result.retrieved_chunks), len(_REALISTIC_RETRIEVAL_IDS))

        # Stage 2 receives rank-1 expansion only
        evidence = mock_generate.call_args[0][1]
        self.assertEqual(len(evidence), len(_REALISTIC_RETRIEVAL_IDS) + 2)
        self.assertIn("v2_class_routine_15", {c.chunk_id for c in evidence})
        self.assertEqual(mock_embed.call_count, 1)
