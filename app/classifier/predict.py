"""
AdtU Campus Copilot — Classifier Inference Module.

Public API:
    predict_intent(query: str) -> str

Returns one of the four locked runtime labels:
    admissions | fees | facilities | out_of_scope

The module is self-contained; it has no dependency on ChromaDB, RAG,
FastAPI, Streamlit, or any Google API.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from app.classifier import LOCKED_LABELS

# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------
_MODEL_PATH: Path = Path(__file__).resolve().parent / "model" / "intent_classifier.pkl"


# ---------------------------------------------------------------------------
# Model loader (cached — loaded once per process)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_model() -> Any:
    """Load the trained pipeline from disk (cached after first call).

    Raises:
        FileNotFoundError: if the model artefact has not been created yet.
    """
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classifier model not found at {_MODEL_PATH}. "
            "Run `python -m app.classifier.train` first."
        )
    return joblib.load(_MODEL_PATH)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def predict_intent(query: str) -> str:
    """Classify a student query into one of the four locked intent classes.

    Args:
        query: Raw user query string.

    Returns:
        One of: "admissions", "fees", "facilities", "out_of_scope".

    Raises:
        TypeError:  if *query* is not a string.
        ValueError: if *query* is empty or whitespace-only.
        RuntimeError: if the model returns an unexpected label (guard rail).
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be a str, got {type(query).__name__!r}")

    query = query.strip()
    if not query:
        raise ValueError("query must not be empty or whitespace-only")

    pipeline = _load_model()
    label: str = str(pipeline.predict([query])[0])

    if label not in LOCKED_LABELS:
        raise RuntimeError(
            f"Model returned unexpected label {label!r}. "
            f"Expected one of {LOCKED_LABELS}."
        )

    return label
