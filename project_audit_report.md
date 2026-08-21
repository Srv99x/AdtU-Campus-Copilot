# AdtU Campus Copilot - Read-Only System Audit Report

Date: 2026-08-20  
Scope: Full read-only architecture, classifier, KB, retrieval, gate, grounding, SQLite, FastAPI, Streamlit, testing, and robustness audit.

---

## 1. Executive Summary

Overall assessment: **Partially demo-ready with known correctness risks**.

- The locked architecture is largely implemented correctly in runtime path:
  Query -> TF-IDF classifier -> out_of_scope short-circuit -> query embedding -> category-filtered Chroma retrieval -> Stage 1 gate -> Gemini grounding -> answer/citations or SQLite escalation -> FastAPI -> Streamlit.
- Core immutable data counts are consistent with locked state:
  - V1 chunks: 842
  - V2 chunks: 76
  - Combined chunks: 918
  - V1 direct embeddings: 836
  - V1 derived embeddings: 45
  - V2 direct embeddings: 76
  - Total vectors: 957
- High-impact failure modes remain, especially:
  - classifier false out_of_scope for valid campus questions
  - scholarship intent boundary inconsistency across classifier labels and KB category mapping
  - weak citation usability for V2 PDF chunks (blank source_url)
  - health endpoint may show green while backend dependencies are unavailable
  - integration tests are mostly mocked, leaving true end-to-end behavior under-tested

**Demo Readiness Score: 6.8 / 10**

- Good for controlled demos using curated query phrasing.
- Risky for open-ended live Q&A without a pre-screened query set.

---

## 2. Architecture Audit

### What is correct

- Canonical orchestration exists in `app/rag/pipeline.py` and enforces the expected stage order.
- FastAPI `/chat` endpoint uses the canonical orchestrator (`run_rag_pipeline`) rather than duplicating logic (`app/api/main.py`).
- Streamlit talks only to FastAPI via HTTP (`app/ui/streamlit_app.py`), no direct Chroma, classifier, or Gemini calls.

### Bypass / duplicate paths

- No production bypass path detected inside `app/` runtime modules.
- Non-runtime direct-path scripts do exist:
  - `scratch/audit_queries.py` directly calls classifier, embedding, Chroma, and pipeline.
  - `scripts/chroma_retrieval_test.py` performs direct retrieval checks.
- These are acceptable for diagnostics but can confuse operators if mistaken for runtime pathways.

### Architecture violations

- No hard architecture violation found in runtime app path.
- Minor operational ambiguity: `README.md` is empty, increasing risk of incorrect operator startup flow.

---

## 3. Classifier Audit

### Frozen interface status

- Inference is isolated and frozen at `app/classifier/predict.py`.
- Locked labels are enforced by guard (`admissions`, `fees`, `facilities`, `out_of_scope`) in `app/classifier/__init__.py` and `predict.py`.

### Dataset and boundary consistency findings

- Dataset distribution in `app/classifier/train.py` matches frozen expectation (62/46/52/68).
- Structural boundary inconsistency exists for scholarship-like queries:
  - `data/classifier/training_queries_v3.csv` includes scholarship terms in both `fees` and `facilities` classes (for example, "topper scholarship", "minority scholrship" labeled facilities; "scholarship discount in fees?" labeled fees).
- Holiday/hostel-seat phrasing coverage is sparse and skewed:
  - "Bihu" token not represented in facilities examples.
  - seat-capacity wording for hostel is not strongly represented in facilities labels.

### Observed/likely misroutes

Backed by calibration and eval artifacts:

- "how many seats are available in the H Block boys hostel" -> predicted `out_of_scope` in `data/rag/gate_calibration_results.csv`.
- "holidays that can be changed by the Assam government" -> predicted `out_of_scope`.
- scholarship queries frequently misroute to `facilities` or `out_of_scope` instead of the intended scholarship-supporting class path.

### Test adequacy for classifier

- `tests/test_classifier.py` verifies shape and representative probes.
- Missing robust regression tests for known observed failure prompts and lexical variants.

---

## 4. Knowledge Base Audit

### Confirmed KB state

- V1 canonical chunk count and V2 canonical chunk count align with locked state.
- V2 metadata inventory exists with provenance fields in `data/processed/knowledge_v2/canonical_chunks.jsonl` and `source_inventory.csv`.

### Topic/category consistency findings

- Scholarships are stored under V2 category `fees` (`v2_scholarships_0`, `v2_scholarships_1`) as documented in `ambiguous_category_assignments.csv`.
- Natural semantic topic of scholarships is explicitly acknowledged as ambiguous in that same file.
- Holiday and class-routine data are under `facilities` category.
- Admission procedural gaps are documented in `knowledge_gaps` style rows of calibration data.

### Answer-exists vs missing-information split

- Answer exists but hard to reach:
  - H-block seat capacity exists in V1 (`hostel-details-html_33` evidence in calibration rows).
  - holiday variance note exists (`v2_holiday_calendar_2`).
  - scholarship slab entries exist (`v2_scholarships_0`).
- True KB gaps (cannot be safely answered):
  - several process and volatile operational questions (transport exact timings, grievance SLA precision, etc.) reflected in calibration `knowledge_gap` rows and V2 research gap artifacts.

---

## 5. Retrieval and Chunking Audit

### Retrieval implementation

- Category-filtered Chroma query is correctly applied in pipeline:
  `where={"category": intent}`, `n_results=10` (`app/rag/pipeline.py`).
- Cosine-space Chroma config is defined in ingestion (`scripts/chroma_ingest.py`).

### Chunking risks

- V2 routine documents often split context header and timetable body into separate chunks:
  - example: `v2_class_routine_14` (heading) then `v2_class_routine_15` (table).
- V2 content has significant PDF extraction artifacts in timetable/holiday/fees tables (`quality_report.csv`, `holiday_calendar.md` raw markdown artifacts).
- Table-heavy chunks with broken tokens/fragmentation increase retrieval and grounding brittleness.

### Derived child retrieval risks

- Parent-child dedup strategy exists and is used by gate (`parent_chunk_id` logic).
- Risk remains when many retrieved hits collapse to fewer distinct parents, causing low-confidence escalation despite semantically close snippets.

---

## 6. Confidence-Gate Audit

### Implementation check

- Current runtime threshold in pipeline is **0.275** and min distinct parents is 3 (`app/rag/pipeline.py`).
- Distinct-parent dedup is implemented in `_distinct_parent_chunks` and used for gate score.

### Calibration artifacts

- `data/rag/gate_calibration_report.md` recommends 0.220 and notes compound logic not implemented.
- Runtime code remains at 0.275, matching the locked state provided for this audit.

### Gate behavior implications

- Stage 1 cannot bypass Stage 2 for high-confidence path; low confidence escalates immediately.
- High-confidence retrieval can still escalate at Stage 2 via `INSUFFICIENT_EVIDENCE`.

---

## 7. Gemini Grounding Audit

### Grounding contract status

- System prompt enforces evidence-only answering and exact `INSUFFICIENT_EVIDENCE` fallback (`app/rag/generator.py`).
- Temperature is fixed at 0.0.
- Citation extraction deduplicates by canonical parent lineage.

### Risks

- Prompt-injection resilience depends mainly on model obedience to system prompt; no additional policy layer or output schema validator is present.
- If model returns non-empty text that is not actually grounded, there is no deterministic post-validation of factual alignment.
- V2 citations often contain blank `source_url` values, reducing user trust and auditability in UI.

---

## 8. SQLite Audit

### Strengths

- Parameterized SQL queries throughout (`app/database/tickets.py`).
- UUID ticket IDs and UTC ISO timestamps are correctly used.
- Schema creation is idempotent.

### Risks

- No retry/backoff strategy for transient `database is locked` conditions.
- `status` field is free-text with no enum/check constraint; functional now, but can drift as admin lifecycle grows.
- No migration/versioning framework yet (acceptable for MVP, but relevant for future admin review workflows).

---

## 9. FastAPI Audit

### Strengths

- Endpoint contract matches expected set: `/health`, `/chat`, `/tickets`, `/tickets/{ticket_id}`.
- Query length and whitespace checks are present.
- Error handling avoids raw traceback leakage to clients.

### Demo-breaking or near-breaking issues

- `/health` only returns static `{status: ok}` and does not verify Chroma collection accessibility or Gemini key availability.
  - Can show green while `/chat` is effectively down.
- On `rag_result.status == error`, API returns internal reason text directly.
  - Better than traceback leakage, but still can expose implementation-level phrasing to users.

---

## 10. Streamlit Audit

### Strengths

- Talks only to FastAPI (`API_BASE_URL` + HTTP requests).
- Displays answered/out_of_scope/escalated/error states clearly.
- New Chat resets local message state.

### Issues

- Citation rendering assumes URL availability; V2 citations with blank `source_url` produce low-value or broken source links.
- Escalated state hides detailed reason (good for user simplicity) but may hinder demo operator diagnostics unless backend logs are monitored.

---

## 11. End-to-End Failure Matrix

| Query | Classifier result | Retrieval result | Stage 1 result | Stage 2 result | Final behavior | Expected behavior | Root cause |
|---|---|---|---|---|---|---|---|
| When is the Bihu holiday? | out_of_scope (observed in calibration analog) | skipped | not_applicable | skipped | out_of_scope | answer from holiday calendar | classifier lexical gap on Bihu/holiday variants |
| What is the seat capacity of the boys H Block hostel? | out_of_scope (observed) | skipped | not_applicable | skipped | out_of_scope | answer 152 seats | classifier lexical gap; hard short-circuit blocks retrieval |
| What scholarship is available for CBSE board students with 95%? | facilities (observed) | facilities-filtered hits, scholarship chunk missed | typically low or weak high | often escalates/irrelevant | incorrect/escalated | answer from v2_scholarships_0 | intent boundary inconsistency + category-filter mismatch |
| How to take admission in BTech CSE AdtU? | admissions | admissions retrieval may be relevant but procedural detail missing | can be high or low | often insufficient evidence | escalated | escalate if process not in trusted KB | expected if exact procedure absent |
| What is the class routine for B.Tech CSE DS & AI IBM 2nd year 1st semester? | facilities | heading chunk often retrieved; table chunk may be separated | high possible | brittle grounding depending on retrieved set | sometimes fails/escalates | robust routine answer with timetable context | heading/table chunk split + extraction artifacts |
| Total fee structure for BCA with IBM collaboration | fees | v2_fees_6 found | high | answerable | usually answered | answered | working path |
| Holidays that can be changed by Assam government | out_of_scope (observed in calibration row variant) | skipped | skipped | skipped | out_of_scope | answer from v2_holiday_calendar_2 note | classifier false negative |
| Warden name for A Block girls hostel | facilities | hostel details retrieved | near-threshold/high | can answer if generation aligns | mixed | answer from hostel details | extraction noise + citation URL quality |
| Exam schedule for first semester 2024 | facilities | mixed retrieval with weak relevance in calibration | low/high variable | often insufficient | escalated/mis-answer risk | likely escalate unless explicit schedule in trusted source | ambiguous schedule phrasing + mixed chunks |
| WiFi password for boys hostel | facilities | hostel chunks retrieved but no credential | low/high variable | should be insufficient | escalated | escalated | acceptable safety behavior |
| What clubs and societies can I join at AdtU? | out_of_scope in calibration boundary row | skipped | skipped | skipped | out_of_scope | likely escalate or partial facilities answer | classifier boundary weakness |
| Capital of France | out_of_scope | skipped | skipped | skipped | out_of_scope | out_of_scope | correct |

Notes:
- Matrix uses direct evidence from calibration CSV and known observed failures provided in scope.
- Some stage outputs are variability-prone because they depend on lexical phrasing and retrieved chunk mix.

---

## 12. Security and Robustness Audit

### Positive findings

- No hardcoded Gemini API key in runtime modules (`query_embed.py`, `app/rag/generator.py`).
- SQL queries are parameterized in ticket storage.
- Runtime pipeline does not mutate embeddings/chunks/Chroma schema.

### Risks

- Prompt injection: user query is passed through directly to generation prompt as data. System prompt is strict, but no second-pass validator exists.
- Input bound: `/chat` max length 1000 is good, but no deeper semantic input normalization/rate limit at API layer.
- Health endpoint overstates readiness by not checking dependent services.
- Chroma/Gemini runtime failures can produce 500 responses during demos.

---

## 13. Demo Readiness Score

**6.8 / 10**

Scoring rationale:
- + strong locked pipeline implementation and immutable data integrity
- + escalation-first safety posture
- - classifier false negatives for valid campus intents
- - scholarship/holiday/class-routine boundary failures
- - weak source-link usability for many V2 answers
- - health check does not reflect true service readiness

---

## 14. Critical Issues

### C1 - Valid in-scope queries are dropped as out_of_scope before retrieval

- Severity: Critical
- Files: `app/rag/pipeline.py`, `data/rag/gate_calibration_results.csv`, `data/classifier/training_queries_v3.csv`
- Why it matters: Produces confident false negatives for real campus questions in live demo.
- Evidence:
  - OOS short-circuit in pipeline stage 2.
  - Calibration rows show answerable queries blocked OOS (for example H-block seats, holiday-change query).
- Recommended fix: Add a controlled fallback policy for uncertain OOS predictions (for example OOS confidence band fallback to facilities retrieval + strict Stage 2 grounding + escalation).
- Safe without changing locked architecture: **Partially** (policy tuning in orchestrator can preserve architecture but changes behavior contract).

### C2 - Scholarship intent/category inconsistency causes systematic misrouting

- Severity: Critical
- Files: `data/classifier/training_queries_v3.csv`, `data/processed/knowledge_v2/ambiguous_category_assignments.csv`, `data/rag/gate_calibration_results.csv`
- Why it matters: Known user-facing scholarship questions route to wrong class and miss existing scholarship chunks.
- Evidence:
  - Scholarships stored in category `fees` but natural topic acknowledged as ambiguous.
  - Training data labels scholarship-like terms across `fees` and `facilities`.
  - Calibration shows scholarship query misclassified and known chunk not found.
- Recommended fix: Introduce deterministic intent override rules for scholarship lexemes before category filter, while preserving 4-class output contract.
- Safe without changing locked architecture: **Yes**.

### C3 - Citation usability gap for V2 answers (blank source_url)

- Severity: Critical
- Files: `data/processed/knowledge_v2/canonical_chunks.jsonl`, `app/rag/generator.py`, `app/ui/streamlit_app.py`
- Why it matters: Demo users cannot verify claims via clickable sources for many V2 answers, reducing trust.
- Evidence:
  - V2 chunks repeatedly have empty `source_url`.
  - UI prints source links directly from citation URL.
- Recommended fix: Add deterministic fallback citation rendering (document + section + chunk id) when URL is blank; optionally map PDFs to canonical public landing pages.
- Safe without changing locked architecture: **Yes**.

---

## 15. Medium Issues

### M1 - Health endpoint can report healthy while chat path is effectively unavailable

- Severity: Medium
- Files: `app/api/main.py`
- Why it matters: Operators may start demo believing system is ready when Chroma collection or Gemini key is missing.
- Evidence: `/health` is static and does not dependency-check.
- Recommended fix: Keep lightweight health but add optional readiness endpoint checking Chroma collection open + key presence.
- Safe without changing locked architecture: **Yes**.

### M2 - Gate calibration report recommends 0.220 while runtime uses 0.275

- Severity: Medium
- Files: `data/rag/gate_calibration_report.md`, `app/rag/pipeline.py`
- Why it matters: Documentation/runtime mismatch may confuse operators and postmortems.
- Evidence: report explicitly recommends 0.220 and says compound gate not implemented; code uses 0.275.
- Recommended fix: update report notes to current locked policy or produce a locked-policy report variant.
- Safe without changing locked architecture: **Yes**.

### M3 - Table extraction artifacts degrade retrieval and grounding quality

- Severity: Medium
- Files: `data/processed/knowledge_v2/markdown/holiday_calendar.md`, `data/processed/knowledge_v2/quality_report.csv`, `scripts/chunk_v2_data.py`
- Why it matters: noisy text can raise false semantic matches and unstable grounding.
- Evidence: multiple artifact-heavy chunks and informational quality flags.
- Recommended fix: targeted cleanup/rechunking for problematic V2 sources only, preserving provenance and immutable V1.
- Safe without changing locked architecture: **Yes** (if done as approved data maintenance).

### M4 - SQLite ticket writes may fail under lock contention without retry

- Severity: Medium
- Files: `app/database/tickets.py`, `tests/test_rag_pipeline.py`
- Why it matters: transient DB lock can turn an escalation into 500 error during demo spikes.
- Evidence: tests explicitly simulate DB lock failure path.
- Recommended fix: add bounded retry/backoff around ticket creation.
- Safe without changing locked architecture: **Yes**.

### M5 - End-to-end tests are heavily mocked and miss true integration behavior

- Severity: Medium
- Files: `tests/test_api.py`, `tests/test_rag_pipeline.py`, `tests/test_rag_generator.py`
- Why it matters: real regressions in Chroma retrieval mix, prompt outputs, and citation quality may pass unit test suite.
- Evidence: tests patch core functions and use synthetic chunk fixtures.
- Recommended fix: add read-only integration test pack using frozen local collection and deterministic expected escalations.
- Safe without changing locked architecture: **Yes**.

---

## 16. Nice-to-Have Issues

### N1 - Empty project README

- Severity: Nice-to-have
- Files: `README.md`
- Why it matters: higher operator error risk during demo setup.
- Evidence: file currently empty.
- Recommended fix: add startup/runbook and demo-safe query list.
- Safe without changing locked architecture: **Yes**.

### N2 - Minor runtime code hygiene (unused imports)

- Severity: Nice-to-have
- Files: `app/api/main.py`, `tests/test_api.py`
- Why it matters: code clarity only; no direct functional impact.
- Evidence: imports such as `GateMetrics`, `RetrievedChunk`, and others are not used in runtime.
- Recommended fix: prune unused imports.
- Safe without changing locked architecture: **Yes**.

### N3 - Diagnostic scripts can be mistaken for production commands

- Severity: Nice-to-have
- Files: `scratch/audit_queries.py`, `scripts/chroma_retrieval_test.py`, `scripts/test_chunk_configs.py`
- Why it matters: accidental misuse may create confusion about canonical operation.
- Evidence: multiple direct pipeline/retrieval diagnostic entry points.
- Recommended fix: add clear "diagnostic-only" headers and operator docs.
- Safe without changing locked architecture: **Yes**.

---

## 17. Recommended Fix Order

1. Prevent false out_of_scope hard failures for high-value in-scope intents (scholarship/holiday/hostel seat-capacity phrasing).
2. Add deterministic scholarship intent override before category-filtered retrieval.
3. Improve citation fallback rendering when `source_url` is empty.
4. Add readiness endpoint and demo preflight checklist.
5. Add bounded retry on SQLite ticket creation lock errors.
6. Add read-only integration regression suite for known failures and known-safe escalations.
7. Update documentation/runbook and calibration note clarity.

---

## What Works Reliably Now

- Core pipeline orchestration and stage ordering
- Escalation-first safety posture when evidence is weak or insufficient
- Immutable dataset and embedding coverage consistency
- Basic API and UI connectivity in local MVP mode

## What Can Fail in Live Demo

- Valid campus queries misclassified as out_of_scope
- Scholarship and holiday queries routed to wrong class and filtered away from relevant chunks
- V2 source links not usable for audience verification
- Backend appears healthy via `/health` while `/chat` fails due missing runtime dependencies

## Acceptable vs Must-Fix Before Demo

Acceptable:
- escalation on true KB gaps
- occasional boundary escalations

Must-fix before high-stakes live demo:
- false out_of_scope for known in-scope intents
- scholarship routing inconsistency
- weak citation usability for V2 evidence
- lack of readiness signal fidelity

## What Should Remain Frozen

- Locked architecture stages and order
- Frozen classifier weights and training/eval datasets
- V1 immutable 842 chunk corpus
- Existing valid embedding checkpoints unless explicitly approved
- FastAPI external endpoint contracts

## What Should Not Be Touched (risk > benefit, for this phase)

- Rebuilding classifier from scratch
- Broad re-chunking/regeneration of all embeddings
- Replacing Chroma/Gemini runtime architecture
- Schema redesign of SQLite for non-essential demo goals
