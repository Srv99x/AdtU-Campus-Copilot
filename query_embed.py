"""Create Gemini query embeddings compatible with the ingestion vectors."""

from __future__ import annotations

import math
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


ROOT = Path(__file__).resolve().parent
DIMENSION = 768


def embedding_model() -> str:
    """Return the configured Gemini embedding model after loading local settings."""

    load_dotenv(ROOT / ".env")
    return os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")


def create_query_embedding(query: str) -> list[float]:
    """Embed one non-empty query using the configured Gemini embedding model."""

    if not query.strip():
        raise ValueError("Query must not be empty.")

    model = embedding_model()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(
        model=model,
        contents=[types.Content(parts=[types.Part(text=query)])],
        config=types.EmbedContentConfig(output_dimensionality=DIMENSION),
    )

    if not result.embeddings:
        raise RuntimeError("No embedding was returned.")

    vector = list(result.embeddings[0].values)
    if len(vector) != DIMENSION or not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in vector
    ):
        raise RuntimeError("Gemini returned an invalid query embedding.")

    return [float(value) for value in vector]


if __name__ == "__main__":
    question = input("Enter your question: ")
    embedding = create_query_embedding(question)
    print("Embedding created successfully.")
    print(f"Dimensions: {len(embedding)}")
