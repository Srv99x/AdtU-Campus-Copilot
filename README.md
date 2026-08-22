# AdtU Campus Copilot

A hackathon-MVP RAG assistant for Assam down town University (AdtU). It answers
campus questions (admissions, fees, facilities) **only from a locked, curated
knowledge base**, always attaches citations to a grounded answer, and — when
the evidence isn't strong enough to trust — escalates to a human via a SQLite
support ticket instead of guessing. This README is a practical runbook for
getting the system running and demoing it, not general project documentation.

## 1. Project overview

AdtU Campus Copilot takes a natural-language question and returns one of three
outcomes: a **grounded answer with citations**, a safe **out-of-scope**
refusal, or a **human escalation ticket** when the retrieved evidence isn't
strong enough to answer confidently. Nothing is ever fabricated: the system
either cites real knowledge-base evidence or says it doesn't know and hands
the question to a human (staff review via the ticket queue).

Locked architecture (do not reorder or bypass — see [AGENTS.md](AGENTS.md)):

```
Query
  -> TF-IDF Classifier (frozen)
  -> Filtered ChromaDB retrieval
  -> Confidence gate (Stage 1)
  -> Gemini Flash grounded generation (Stage 2) OR SQLite ticket escalation
  -> FastAPI
  -> Streamlit
```

Streamlit never talks to Chroma or Gemini directly — it only calls FastAPI
over HTTP.

## 2. Key capabilities

- Natural-language campus Q&A over a curated, locked knowledge base
- 4-class intent routing (`admissions`, `fees`, `facilities`, `out_of_scope`)
- Out-of-scope recovery for campus-vocabulary queries the frozen classifier
  still labels `out_of_scope` (Policy B — see [Section 9](#9-guided-demo-script))
- Scholarship monetary-query retrieval correction (routes amount/percentage
  scholarship questions to the category that actually holds the scholarship
  table, without changing the classifier's own intent label)
- Class-routine sibling expansion (reunites a routine's heading chunk with its
  timetable-table chunk before generation, for the single top-ranked routine
  match)
- Grounded Gemini generation with a strict evidence-only system prompt
- Non-fabricated citations (a documented fallback renders when a source's
  `source_url` is blank — see [Current limitations](#current-limitations))
- Safe SQLite escalation whenever confidence or evidence is insufficient
- Ticket resolution / staff admin view (sidebar ticket queue with Resolve)
- Readiness monitoring (`/health` vs `/ready`, three-state UI indicator)
- Guided demo mode (one-click scenario buttons in the Streamlit sidebar)
- Trust & Evidence panel (an expandable "why this answer?" / "not enough
  verified evidence" panel shown under every answered/escalated response)

## 3. Prerequisites

- Python 3.12 (the version this project's test environment is verified against;
  the pinned dependencies in `requirements.txt` do not declare a stricter floor)
- A Gemini API key (for query embedding and answer generation)
- Git
- A web browser (for the Streamlit UI)

## 4. Fresh-clone setup

```powershell
git clone <YOUR_REPO_URL>
cd AdtU-Campus-Copilot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 5. Environment configuration

Copy the template and fill in your own key:

```powershell
Copy-Item .env.example .env
```

`.env.example` documents exactly the three variables the runtime reads:

```
GEMINI_API_KEY=
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_GENERATION_MODEL=gemini-2.5-flash
```

- `GEMINI_API_KEY` — your personal Gemini API key. Leave it blank in
  `.env.example`; put the real value only in your local `.env`.
- `GEMINI_EMBEDDING_MODEL` — the embedding model used for both the original
  ingestion and every runtime query embedding. Do not change this unless you
  are deliberately re-embedding everything (see [Section 6](#6-critical-runtime-data-snapshot)).
- `GEMINI_GENERATION_MODEL` — the Gemini Flash model used for grounded answer
  generation.

**Never commit `.env`.** It is already listed in `.gitignore`. **Never paste
your API key into source code** — every runtime module reads it from the
environment (via `.env`), never a hardcoded literal.

## 6. CRITICAL: Runtime data snapshot

A fresh `git clone` of this repository does **not** contain the validated
ChromaDB runtime database or `data/processed/derived_embeddings.json` — both
are intentionally excluded via `.gitignore` (`chroma_db/`,
`data/processed/derived_embeddings.json`) because they are large, binary,
generated artifacts, not source.

Without them, `/chat` cannot retrieve anything and `/ready` will report the
Chroma check as failing. Regenerating them from scratch is **not** a
fresh-clone step you want to take casually: rebuilding
`derived_embeddings.json` and re-ingesting the collection requires making
real Gemini embedding API calls (`gemini-embedding-2`) against your own quota
— it costs time, quota, and money, and it is exactly the kind of
regeneration [AGENTS.md](AGENTS.md) says not to do without explicit approval.

The validated runtime snapshot contains exactly **957 vectors**:

- 836 V1 direct canonical embeddings
- 45 V1 derived-child embeddings (for the 6 canonical chunks too long for a
  single embedding call)
- 76 V2 direct embeddings

This is published as a GitHub Release artifact:

- **File:** `adtu_kb_snapshot_957v_20260822.zip`
- **SHA-256:** `2d6ae1b6ecfb20e03ca69e40220f0c6a3bcd7a236d613203f0b96410a086f009`
- **Download:** `<GITHUB_RELEASE_URL>` — replace this placeholder with the
  actual release URL once the release is published; do not guess or invent
  one in the meantime.

Setup:

1. Download `adtu_kb_snapshot_957v_20260822.zip` from `<GITHUB_RELEASE_URL>`.
2. Verify its checksum before trusting it:
   ```powershell
   Get-FileHash .\adtu_kb_snapshot_957v_20260822.zip -Algorithm SHA256
   ```
   Confirm the output matches
   `2d6ae1b6ecfb20e03ca69e40220f0c6a3bcd7a236d613203f0b96410a086f009`.
3. Extract it from the **repository root** so the archive's internal paths
   land exactly here:
   ```
   data/processed/derived_embeddings.json
   data/processed/chroma_db/
   ```
   (The zip's internal paths already start with `data/processed/...`, so
   extracting it at the repo root reproduces both paths directly — no manual
   file moving should be needed.)
4. Confirm both paths exist before starting the backend:
   ```powershell
   Test-Path .\data\processed\derived_embeddings.json
   Test-Path .\data\processed\chroma_db\chroma.sqlite3
   ```

## 7. Startup

Two terminals, both from the repository root with the virtual environment
activated.

**Terminal 1 — backend:**

```powershell
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — frontend:**

```powershell
streamlit run app/ui/streamlit_app.py --server.port 8501
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

## 8. Health / readiness

- **`/health`** — process-alive check only. Always returns `{"status": "ok"}`
  if uvicorn is running, even if Chroma or the Gemini key is broken. It never
  touches Chroma or Gemini.
- **`/ready`** — dependency-aware readiness check. Confirms the Gemini API
  key is present and non-blank, and that the Chroma `adtu_knowledge`
  collection is reachable and non-empty. **It never calls the Gemini API** —
  it only checks that a key value is present, never that the key is valid.
  Returns HTTP 200 `{"status": "ready"}` when every check passes, or HTTP 503
  `{"status": "not_ready", "checks": {...}}` naming which check failed.

The Streamlit top bar reflects this as one of three states (calls `/ready`,
not `/health`, for exactly this reason):

- ✅ **Backend: Ready** — `/ready` returned 200.
- ⚠️ **Backend: Running but not ready** — uvicorn answered, but a dependency
  check failed (names the failing check, e.g. `chroma` or `gemini_api_key`).
- 🚨 **Backend: Offline or Unreachable** — the request itself failed
  (connection refused, timeout, etc.) — uvicorn probably isn't running.

## 9. Guided demo script

Streamlit's **"🎯 Try a scenario"** expander has one-click buttons for most of
these; each button submits the same real query a typed message would, through
the same code path — nothing is a canned/hardcoded answer. Two steps below
(4 and 7) are not buttons and are called out as such.

1. **Normal Q&A** — button *📚 Campus Q&A*:
   `What documents are required for BTech admission at AdtU?`
   Demonstrates the baseline path: classifier → category-filtered retrieval →
   confidence gate → grounded Gemini answer with citations.

2. **Scholarship** — button *🎓 Scholarship amount*:
   `What scholarship is available for CBSE board students with 95%?`
   Demonstrates the scholarship monetary-retrieval override: this query is
   inconsistently classified by the frozen TF-IDF classifier, but the
   pipeline detects the monetary/percentage signal and forces retrieval
   against the `fees` category (where the scholarship table actually lives)
   regardless of the classifier's own label.

3. **Class routine** — button *🗓️ Class routine*:
   `What is the class routine for B.Tech CSE DS & AI IBM 2nd year 1st semester?`
   Demonstrates class-routine sibling expansion: the top-ranked routine
   "heading" chunk is paired with its separately-stored timetable "table"
   chunk before the evidence is handed to Gemini, so generation isn't left
   with only a heading and no timetable data.

4. **OOS recovery** *(type this manually — there is no dedicated button)*:
   `When is the Bihu holiday?`
   The frozen classifier still labels this `out_of_scope` — that label is
   never changed. But the pipeline's Policy B vocabulary gate detects campus
   terms (`bihu`, `holiday`) and runs one controlled recovery retrieval
   against the `facilities` category (with a "academic calendar" retrieval-
   query normalization) before falling back to the same Stage 1/Stage 2
   pipeline used everywhere else. It may answer or may safely escalate,
   depending on retrieval confidence — either outcome demonstrates the
   recovery path running, not a guaranteed answer.

5. **Hard OOS** — button *🚫 Out of scope*:
   `What is the capital of France?`
   No campus vocabulary matches at all, so Policy B recovery is not
   triggered: no embedding call, no retrieval, no Gemini call — an immediate,
   cheap `out_of_scope` response.

6. **Safe escalation** — button *🎫 Safe escalation*:
   `What is the WiFi password for the boys hostel?`
   Retrieval runs normally, but the knowledge base has no such credential —
   the system does not invent one. Depending on retrieval confidence this
   either escalates at Stage 1 or Stage 2, always producing a SQLite ticket
   and never a fabricated answer.

7. **Trust & Evidence panel** *(not a separate query — expand it after any
   of the above)*: every `answered` response shows a **🔎 Why this answer?**
   expander (intent, confidence, cited evidence); every `escalated` response
   shows a **🟡 Not enough verified evidence** expander (confidence, backend
   reasoning, ticket ID). Both render only fields the backend actually
   returned — nothing here is computed or guessed client-side.

8. **Staff ticket resolution**: open the sidebar, click **Load tickets**,
   find the ticket created in step 6 (or step 4/2 if those escalated), and
   click **Resolve**. See [Section 10](#10-staffadmin-flow).

## 10. Staff/Admin flow

The sidebar's **🛠️ Staff / Admin** panel is a queue over the same tickets
created by Stage 1/Stage 2 escalation:

1. **Load tickets** — fetches the current ticket list from `GET /tickets`.
2. **Inspect** — each ticket card shows its query, predicted intent, source
   (which stage escalated it), creation time, and ID.
3. **Resolve** — calls `PATCH /tickets/{id}` with `{"status": "resolved"}`
   and refreshes the list.

**`PATCH /tickets/{id}` is unauthenticated.** Any client that can reach the
backend can resolve any ticket. This is acceptable only for a local/hackathon
MVP running on `127.0.0.1` — **it is not, and must not be described as,
production-ready authentication.** Do not expose this backend on a public
network without adding real auth first.

## 11. Testing

```powershell
pytest tests/ -q
```

Currently-verified baseline on this branch: **302 passed** (plus 26 subtests).
Treat this as the last-verified snapshot, not a guarantee for future changes
— re-run it yourself after any change and trust that output over this number.

**Always scope pytest to `tests/`, never run bare `pytest` from the repo
root.** `scripts/chroma_retrieval_test.py` matches pytest's default
`*_test.py` discovery pattern, so an unscoped `pytest` run also collects its
`test_query(...)` function — a diagnostic script with required parameters
pytest can't supply as fixtures, and one that (when actually run standalone
via `python scripts/chroma_retrieval_test.py`) makes real Gemini embedding
calls. It is not part of the test suite's contract and will only produce
noise or errors if pytest tries to collect it.

## 12. Troubleshooting

- **Backend offline** (Streamlit shows 🚨 Offline or Unreachable): confirm
  the `uvicorn` terminal is still running and listening on port 8000, and
  that `API_BASE_URL` (defaults to `http://127.0.0.1:8000`) matches it.
- **`/ready` returns 503**: check the JSON body's `checks` object — it names
  exactly which dependency failed (`chroma` or `gemini_api_key`) and why.
- **Chroma collection missing/unreachable**: almost always means the runtime
  snapshot wasn't extracted (see [Section 6](#6-critical-runtime-data-snapshot)).
  Confirm `data/processed/chroma_db/chroma.sqlite3` exists.
- **API key missing**: confirm `.env` exists at the repo root (not just
  `.env.example`) and `GEMINI_API_KEY` is set and non-blank in it.
- **Port 8000 already in use**: stop whatever is bound to it, or start
  uvicorn with a different `--port` and update `API_BASE_URL` for Streamlit
  accordingly.
- **Port 8501 already in use**: start Streamlit with a different
  `--server.port`.
- **Runtime snapshot not extracted correctly**: re-check that extraction ran
  from the repository root (not from inside `data/` or `data/processed/`) —
  a common mistake produces a nested `data/processed/data/processed/...` path
  instead of overlaying directly onto the existing `data/processed/`
  directory. Re-verify with the `Test-Path` commands in Section 6.

## 13. Architecture / safety principles

- The classifier (`app/classifier/`) is frozen — its weights, labels, and
  training/eval datasets are not to be modified casually.
- Embeddings and the ChromaDB collection are protected data, not
  regenerated/reset without explicit approval.
- No fabricated answers: generation is evidence-only, and insufficient
  evidence escalates to a human ticket rather than producing a best guess.
- Secrets are never committed: `.env` is git-ignored, and every runtime
  module reads `GEMINI_API_KEY` from the environment, never a literal.
- Streamlit only talks to FastAPI over HTTP — it never imports or calls
  Chroma, the classifier, or Gemini directly.

## 14. Hackathon demo checklist

- [ ] `.env` exists at the repo root with a valid `GEMINI_API_KEY`
- [ ] Runtime snapshot restored (`data/processed/chroma_db/` and
      `data/processed/derived_embeddings.json` both present)
- [ ] `GET /ready` returns `{"status": "ready"}`
- [ ] Streamlit opens and shows ✅ Backend: Ready
- [ ] Guided demo buttons in "🎯 Try a scenario" work
- [ ] The safe-escalation scenario creates a ticket
- [ ] Staff/Admin sidebar can load and resolve that ticket
- [ ] The Trust & Evidence panel renders under an answered/escalated response
- [ ] A backup copy of the runtime snapshot archive exists somewhere outside
      the working tree (in case `data/processed/` needs to be re-extracted
      mid-event)

## Current limitations

- The `PATCH /tickets/{id}` endpoint is unauthenticated — acceptable for a
  local hackathon MVP only, not production-ready auth.
- Live integration testing against real Gemini/Chroma is limited; most of
  `tests/` exercises the pipeline with mocked dependencies rather than a live
  end-to-end call.
- [README_INGESTION.md](README_INGESTION.md) documents the ingestion pipeline
  as of an earlier phase and may contain historical/stale figures — this
  README's [Section 6](#6-critical-runtime-data-snapshot) is the current
  source of truth for the validated runtime vector count (957).
- This is a hackathon MVP, not a production deployment.
