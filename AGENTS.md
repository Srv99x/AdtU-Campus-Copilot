# AdtU Campus Copilot — AI Coding Rules

## Locked Architecture

Query
→ TF-IDF Classifier
→ Filtered ChromaDB Retrieval
→ Confidence Gate
→ Gemini Flash OR SQLite Ticket Escalation
→ FastAPI
→ Streamlit

## Rules for AI Agents

1. Inspect existing code before modifying it.
2. Do not rewrite unrelated files.
3. Preserve the locked application architecture.
4. Do not replace ChromaDB or the Gemini runtime embedding/generation architecture unless explicitly instructed.
5. Use `gemini-embedding-2` for runtime embeddings.
6. Use the configurable Gemini Flash generation model from `GEMINI_GENERATION_MODEL`.
7. Never hardcode API keys. Use environment variables.
8. Use Python type hints.
9. Add tests for new logic.
10. Do not delete data.
11. Do not modify the SQLite schema without explicit approval.
12. Do not change API contracts without explicit approval.
13. Only implement the specific milestone requested.
14. Do not introduce unnecessary dependencies.
15. Run tests after non-trivial changes.
16. Do not modify `.env`.
- The canonical 842-chunk dataset is immutable unless explicitly approved.
- Existing valid embeddings must never be deleted or regenerated unnecessarily.
- Do not call paid APIs without explicit approval.
- Do not expose or commit API keys.
- Do not commit private/student/WhatsApp data to the public repository.
- New KB V2 sources must preserve provenance.
- Every new fact must be traceable to an authoritative source.
- Unverified information must never enter the trusted knowledge base.
- Do not start Streamlit until the backend/RAG foundation is complete.

Generated checkpoints such as:
- data/processed/embeddings.json
- data/processed/v2_embeddings.json
- derived_embeddings.json

must never be restored, reset, deleted, or overwritten without explicit approval.