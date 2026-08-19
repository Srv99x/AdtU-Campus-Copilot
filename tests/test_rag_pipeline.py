from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.rag.pipeline import RetrievedChunk, assess_retrieval_confidence, run_rag_pipeline


class TestRagPipeline(unittest.TestCase):

    def setUp(self) -> None:
        self.mock_collection = MagicMock()

    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_out_of_scope_short_circuit(self, mock_predict: MagicMock, mock_embed: MagicMock) -> None:
        """Test that out_of_scope queries return immediately without embedding or retrieval."""
        mock_predict.return_value = "out_of_scope"
        
        result = run_rag_pipeline("how to cook maggi", self.mock_collection)
        
        # Verify result structure
        self.assertTrue(result.is_out_of_scope)
        self.assertEqual(result.intent, "out_of_scope")
        self.assertEqual(result.confidence_status, "not_applicable")
        self.assertEqual(len(result.retrieved_chunks), 0)
        
        # Verify bypassed dependencies
        mock_embed.assert_not_called()
        self.mock_collection.query.assert_not_called()

    @patch("app.rag.pipeline.create_query_embedding")
    @patch("app.rag.pipeline.predict_intent")
    def test_in_scope_retrieval_and_filtering(self, mock_predict: MagicMock, mock_embed: MagicMock) -> None:
        """Test that in-scope queries generate embeddings and filter Chroma by category."""
        mock_predict.return_value = "admissions"
        mock_embed.return_value = [0.1, 0.2, 0.3]
        
        self.mock_collection.query.return_value = {
            "ids": [["chunk_1"]],
            "documents": [["doc_text"]],
            "metadatas": [[{"category": "admissions", "chunk_id": "chunk_1"}]],
            "distances": [[0.15]]
        }

        result = run_rag_pipeline("how to apply for btech", self.mock_collection)
        
        # Verify the classifier was called
        mock_predict.assert_called_once_with("how to apply for btech")
        
        # Verify the embedding was generated
        mock_embed.assert_called_once_with("how to apply for btech")
        
        # Verify Chroma was queried with the correct filter and embedding
        self.mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=10,
            where={"category": "admissions"},
            include=["documents", "metadatas", "distances"]
        )
        
        # Verify structured output
        self.assertFalse(result.is_out_of_scope)
        self.assertEqual(result.intent, "admissions")
        self.assertEqual(len(result.retrieved_chunks), 1)
        
        chunk = result.retrieved_chunks[0]
        self.assertEqual(chunk.chunk_id, "chunk_1")
        self.assertEqual(chunk.document, "doc_text")
        self.assertEqual(chunk.distance, 0.15)
        self.assertEqual(chunk.metadata["category"], "admissions")

    def test_assess_retrieval_confidence_interface(self) -> None:
        """Test the confidence gate placeholder."""
        # Empty chunks
        status, reason = assess_retrieval_confidence([])
        self.assertEqual(status, "low")
        self.assertIn("No chunks", reason)
        
        # With chunks
        chunk = RetrievedChunk(chunk_id="1", document="test", metadata={}, distance=0.1)
        status, reason = assess_retrieval_confidence([chunk])
        self.assertEqual(status, "unassessed")
        self.assertIn("not yet calibrated", reason)

if __name__ == "__main__":
    unittest.main()
