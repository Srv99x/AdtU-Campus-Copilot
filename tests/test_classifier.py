"""
tests/test_classifier.py

Unit tests for the AdtU Campus Copilot intent classifier.

Tests are ADDITIVE — they do not modify existing test files.

Coverage:
    - Model artefact exists after training
    - predict_intent returns valid locked labels
    - Representative probe queries for all four classes
    - Empty / non-string input raises correct exceptions
    - Whitespace-only input raises ValueError
    - Label guard: returned value is always in LOCKED_LABELS
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.classifier import LOCKED_LABELS
from app.classifier.predict import predict_intent, _MODEL_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_locked(label: str) -> None:
    """Assert that *label* is one of the four locked runtime labels."""
    assert label in LOCKED_LABELS, (
        f"predict_intent returned {label!r}, expected one of {LOCKED_LABELS}"
    )


# ---------------------------------------------------------------------------
# 1. Artefact existence
# ---------------------------------------------------------------------------

class TestModelArtefact:
    def test_model_file_exists(self) -> None:
        """The trained model pkl must be present on disk."""
        assert _MODEL_PATH.exists(), (
            f"Model not found at {_MODEL_PATH}. "
            "Run `python -m app.classifier.train` first."
        )

    def test_model_file_is_nonempty(self) -> None:
        """The pkl file must not be a zero-byte placeholder."""
        assert _MODEL_PATH.stat().st_size > 0


# ---------------------------------------------------------------------------
# 2. Output label validity
# ---------------------------------------------------------------------------

class TestOutputLabelValidity:
    @pytest.mark.parametrize("query", [
        "btech admission process",
        "fees installment",
        "hostel room",
        "how to reverse a string in python",
    ])
    def test_label_is_locked(self, query: str) -> None:
        label = predict_intent(query)
        _assert_locked(label)


# ---------------------------------------------------------------------------
# 3. Representative queries — one probe per class
# ---------------------------------------------------------------------------

class TestRepresentativeQueries:
    def test_admissions_probe(self) -> None:
        label = predict_intent("what documents are needed for btech admission")
        assert label == "admissions", f"Expected 'admissions', got {label!r}"

    def test_fees_probe(self) -> None:
        label = predict_intent("mba tuition fee structure")
        assert label == "fees", f"Expected 'fees', got {label!r}"

    def test_facilities_probe(self) -> None:
        label = predict_intent("hostel room availability on campus")
        assert label == "facilities", f"Expected 'facilities', got {label!r}"

    def test_out_of_scope_probe(self) -> None:
        label = predict_intent("how to reverse a linked list in python")
        assert label == "out_of_scope", f"Expected 'out_of_scope', got {label!r}"


# ---------------------------------------------------------------------------
# 4. Additional in-class probes (broader coverage)
# ---------------------------------------------------------------------------

class TestAdditionalProbes:
    @pytest.mark.parametrize("query,expected", [
        # admissions
        ("cutoff for bpharma", "admissions"),
        ("cuet required for bba", "admissions"),
        ("gap certificate needed for admission", "admissions"),
        # fees
        ("how to pay college fees online", "fees"),
        ("refund policy for withdrawn students", "fees"),
        ("security deposit amount", "fees"),
        # facilities
        ("library timing", "facilities"),
        ("academic calendar 2024", "facilities"),
        ("bus routes from jalukbari", "facilities"),
        # out_of_scope
        ("capital of australia", "out_of_scope"),
        ("solve quadratic equation", "out_of_scope"),
        ("best netflix shows right now", "out_of_scope"),
    ])
    def test_parametrized_probes(self, query: str, expected: str) -> None:
        label = predict_intent(query)
        assert label == expected, (
            f"Query {query!r}: expected {expected!r}, got {label!r}"
        )


# ---------------------------------------------------------------------------
# 5. Invalid / edge-case input handling
# ---------------------------------------------------------------------------

class TestInvalidInput:
    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            predict_intent("")

    def test_whitespace_only_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            predict_intent("   ")

    def test_none_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="str"):
            predict_intent(None)  # type: ignore[arg-type]

    def test_integer_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="str"):
            predict_intent(42)  # type: ignore[arg-type]

    def test_list_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="str"):
            predict_intent(["query"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 6. LOCKED_LABELS contract
# ---------------------------------------------------------------------------

class TestLockedLabels:
    def test_exactly_four_labels(self) -> None:
        assert len(LOCKED_LABELS) == 4

    def test_label_names(self) -> None:
        expected = {"admissions", "fees", "facilities", "out_of_scope"}
        assert set(LOCKED_LABELS) == expected
