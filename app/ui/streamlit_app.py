"""
AdtU Campus Copilot — Streamlit MVP
A lightweight UI wrapping the FastAPI orchestration backend.
"""
import os
import sys
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.generator import format_citation_reference

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT_SEC = 30

# ---------------------------------------------------------------------------
# Guided Demo Mode (Phase 7A) — curated example queries.
#
# Purely presentational data: each entry is a real query string fed through
# the exact same send_chat_query()/_handle_user_query() path as manually
# typed chat input. No answers are hardcoded here -- clicking a scenario
# only submits its `query` to the real backend, same as typing it.
#
# Safe to edit or remove entries here without touching any other logic.
# ---------------------------------------------------------------------------
DEMO_SCENARIOS: list[dict[str, str]] = [
    {
        "label": "📚 Campus Q&A",
        "query": "What documents are required for BTech admission at AdtU?",
    },
    {
        "label": "🎓 Scholarship amount",
        "query": "What scholarship is available for CBSE board students with 95%?",
    },
    {
        "label": "🗓️ Class routine",
        "query": "What is the class routine for B.Tech CSE DS & AI IBM 2nd year 1st semester?",
    },
    {
        "label": "🚫 Out of scope",
        "query": "What is the capital of France?",
    },
    {
        "label": "🎫 Safe escalation",
        "query": "What is the WiFi password for the boys hostel?",
    },
]
_PENDING_DEMO_QUERY_KEY = "_pending_demo_query"

# ---------------------------------------------------------------------------
# Staff / Admin view (Phase 7C) — session-state keys.
# ---------------------------------------------------------------------------
_ADMIN_TICKETS_KEY = "_admin_tickets"
_ADMIN_ERROR_KEY = "_admin_error"
_ADMIN_NOTICE_KEY = "_admin_notice"

st.set_page_config(page_title="AdtU Campus Copilot", layout="centered")

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Backend Interaction Helpers
# ---------------------------------------------------------------------------
def check_backend_health() -> bool:
    """Check if the FastAPI backend is reachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except (requests.exceptions.RequestException, ValueError):
        return False

def send_chat_query(query: str) -> dict:
    """Send user query to the backend and return the parsed response."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"query": query},
            timeout=TIMEOUT_SEC
        )
        if response.status_code == 422:
            return {"status": "error", "reason": "Query rejected by server validation (e.g. too long)."}
        if response.status_code >= 500:
            return {"status": "error", "reason": "Backend internal error. Please try again later."}
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"status": "error", "reason": "Request timed out. The backend is taking too long to respond."}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "reason": "Backend is unreachable. Please verify the API server is running."}
    except Exception as e:
        return {"status": "error", "reason": "An unexpected network error occurred."}


def fetch_tickets() -> dict:
    """Fetch the escalation ticket list from the backend (Phase 7C).

    Returns {"ok": bool, "tickets": list, "error": str | None}. Network and
    backend failures are reported, never raised, so the admin panel can show
    a safe message instead of crashing the page.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/tickets", timeout=TIMEOUT_SEC)
        if response.status_code >= 500:
            return {"ok": False, "tickets": [], "error": "Backend internal error while loading tickets."}
        response.raise_for_status()
        return {"ok": True, "tickets": response.json(), "error": None}
    except requests.exceptions.Timeout:
        return {"ok": False, "tickets": [], "error": "Request timed out while loading tickets."}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "tickets": [], "error": "Backend is unreachable. Please verify the API server is running."}
    except Exception:
        return {"ok": False, "tickets": [], "error": "An unexpected error occurred while loading tickets."}


def resolve_ticket(ticket_id: str) -> dict:
    """Mark a ticket resolved via the backend PATCH endpoint (Phase 7C).

    Returns {"ok": bool, "ticket": dict | None, "error": str | None}.
    """
    try:
        response = requests.patch(
            f"{API_BASE_URL}/tickets/{ticket_id}",
            json={"status": "resolved"},
            timeout=TIMEOUT_SEC,
        )
        if response.status_code == 404:
            return {"ok": False, "ticket": None, "error": "Ticket not found."}
        if response.status_code == 422:
            return {"ok": False, "ticket": None, "error": "Ticket update rejected by server validation."}
        if response.status_code >= 500:
            return {"ok": False, "ticket": None, "error": "Backend internal error while resolving the ticket."}
        response.raise_for_status()
        return {"ok": True, "ticket": response.json(), "error": None}
    except requests.exceptions.Timeout:
        return {"ok": False, "ticket": None, "error": "Request timed out while resolving the ticket."}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "ticket": None, "error": "Backend is unreachable. Please verify the API server is running."}
    except Exception:
        return {"ok": False, "ticket": None, "error": "An unexpected error occurred while resolving the ticket."}


# ---------------------------------------------------------------------------
# Trust & Evidence Panel (Phase 7B)
#
# Both panels below render ONLY fields that already exist on the backend's
# ChatResponse (see app/api/main.py) -- confidence_status, reason, ticket_id,
# and citations (chunk_id/parent_chunk_id/source_url/section/source_type).
# Nothing here is computed, guessed, or fabricated client-side; a field that
# the backend didn't return simply isn't shown. Citations are rendered with
# the same format_citation_reference() used everywhere else (Phase 6A),
# including its existing non-fabricated fallback when source_url is blank.
# ---------------------------------------------------------------------------
def _render_evidence_panel(metadata: dict) -> None:
    """'Why this answer?' -- shown for a grounded (status == 'answered') reply."""
    with st.expander("🔎 Why this answer?"):
        summary_bits = []
        if metadata.get("intent"):
            summary_bits.append(f"Intent: `{metadata['intent']}`")
        if metadata.get("confidence_status"):
            summary_bits.append(f"Confidence: `{metadata['confidence_status']}`")
        citations = metadata.get("citations")
        if citations:
            summary_bits.append(f"Sources cited: {len(citations)}")
        if summary_bits:
            st.caption(" · ".join(summary_bits))

        if citations:
            st.markdown("**Cited evidence**")
            for c in citations:
                st.markdown(f"- {format_citation_reference(**c)}")
        else:
            st.caption("No citation metadata was returned with this answer.")

        if metadata.get("reason"):
            st.caption(f"Backend reasoning: {metadata['reason']}")


def _render_escalation_panel(metadata: dict) -> None:
    """'Not enough verified evidence' -- shown for status == 'escalated'."""
    with st.expander("🟡 Not enough verified evidence"):
        if metadata.get("confidence_status"):
            st.caption(f"Retrieval confidence: `{metadata['confidence_status']}`")
        if metadata.get("reason"):
            st.caption(f"Backend reasoning: {metadata['reason']}")
        if metadata.get("ticket_id"):
            st.caption(f"A support ticket (`{metadata['ticket_id']}`) was created for staff review.")


# ---------------------------------------------------------------------------
# Staff / Admin view (Phase 7C)
#
# Lives in the sidebar so the student chat experience is visually unchanged.
# Renders ONLY fields the ticket API actually returns (ticket_id, status,
# created_at, predicted_intent, query, source) -- there is no staff name,
# resolution timestamp, priority, or SLA in the ticket model, so none is
# shown or invented.
# ---------------------------------------------------------------------------
def _load_admin_tickets() -> None:
    """Refresh the cached ticket list from the backend into session state."""
    result = fetch_tickets()
    if result["ok"]:
        st.session_state[_ADMIN_TICKETS_KEY] = result["tickets"]
        st.session_state[_ADMIN_ERROR_KEY] = None
    else:
        st.session_state[_ADMIN_ERROR_KEY] = result["error"]


def _render_admin_panel() -> None:
    with st.sidebar:
        st.markdown("### 🛠️ Staff / Admin")
        st.caption("Escalated queries awaiting staff review.")

        if st.button("Load tickets", key="admin_load_tickets", use_container_width=True):
            _load_admin_tickets()
            st.rerun()

        notice = st.session_state.pop(_ADMIN_NOTICE_KEY, None)
        if notice:
            st.success(notice)

        error = st.session_state.get(_ADMIN_ERROR_KEY)
        if error:
            st.error(error)

        tickets = st.session_state.get(_ADMIN_TICKETS_KEY)
        if tickets is None:
            return

        if not tickets:
            st.caption("No escalation tickets yet.")
            return

        open_count = sum(1 for t in tickets if t.get("status") == "open")
        st.caption(f"{len(tickets)} ticket(s) · {open_count} open")

        for ticket in tickets:
            ticket_id = ticket.get("ticket_id", "")
            ticket_status = ticket.get("status", "")
            badge = "🟡 OPEN" if ticket_status == "open" else "✅ RESOLVED"

            with st.container(border=True):
                st.markdown(f"**{badge}**")
                if ticket.get("query"):
                    st.caption(f"Query: {ticket['query']}")
                meta_bits = []
                if ticket.get("predicted_intent"):
                    meta_bits.append(f"Intent: `{ticket['predicted_intent']}`")
                if ticket.get("source"):
                    meta_bits.append(f"Source: `{ticket['source']}`")
                if meta_bits:
                    st.caption(" · ".join(meta_bits))
                if ticket.get("created_at"):
                    st.caption(f"Created: {ticket['created_at']}")
                st.caption(f"ID: `{ticket_id}`")

                if ticket_status == "open":
                    if st.button(
                        "Resolve",
                        key=f"admin_resolve_{ticket_id}",
                        use_container_width=True,
                    ):
                        outcome = resolve_ticket(ticket_id)
                        if outcome["ok"]:
                            st.session_state[_ADMIN_NOTICE_KEY] = f"Ticket {ticket_id} resolved."
                            _load_admin_tickets()
                        else:
                            st.session_state[_ADMIN_ERROR_KEY] = outcome["error"]
                        st.rerun()


def _handle_user_query(prompt: str) -> None:
    """Submit *prompt* through the chat workflow: call the backend, render
    the exchange, and append it to session history.

    Single code path for both manually typed chat input and Guided Demo Mode
    scenario buttons — neither has its own logic; both call this function.
    """
    if not prompt.strip():
        st.warning("Please enter a valid query.")
        return

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Fetch assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = send_chat_query(prompt)

        status = result.get("status", "error")
        metadata = {
            "status": status,
            "intent": result.get("intent"),
            "confidence_status": result.get("confidence_status"),
            "ticket_id": result.get("ticket_id"),
            "citations": result.get("citations"),
            "reason": result.get("reason"),
        }

        if status == "answered":
            answer = result.get("answer", "No answer provided.")
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer, "metadata": metadata})

            # Display metadata inline for immediate feedback
            indicators = []
            if metadata["intent"]: indicators.append(f"Intent: `{metadata['intent']}`")
            if metadata["confidence_status"]: indicators.append(f"Confidence: `{metadata['confidence_status']}`")
            if indicators:
                st.caption(" | ".join(indicators))

            _render_evidence_panel(metadata)

        elif status == "out_of_scope":
            msg = "This assistant only handles AdtU-related queries (admissions, fees, facilities). Please rephrase or ask a campus-related question."
            st.warning(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "metadata": metadata})
            st.caption(f"Intent: `{metadata['intent']}`")

        elif status == "escalated":
            msg = "I could not verify the answer to your question in the official knowledge base. A support ticket has been created for staff review."
            st.info(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "metadata": metadata})
            indicators = [f"Intent: `{metadata['intent']}`", f"Ticket: `{metadata['ticket_id']}`"]
            st.caption(" | ".join(indicators))
            _render_escalation_panel(metadata)

        elif status == "error":
            msg = f"An error occurred: {result.get('reason', 'Unknown error')}"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "metadata": metadata})
        else:
            msg = f"Unexpected response status: {status}"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "metadata": metadata})

# ---------------------------------------------------------------------------
# UI Rendering
# ---------------------------------------------------------------------------

st.title("AdtU Campus Copilot")
st.markdown("##### Official institutional assistant for Assam down town University")

# Staff/Admin ticket queue (Phase 7C) — sidebar only; the student-facing
# chat experience below is unchanged.
_render_admin_panel()

# Top bar actions
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("New Chat", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if check_backend_health():
        st.success("Backend: Online", icon="✅")
    else:
        st.error("Backend: Offline or Unreachable", icon="🚨")

st.divider()

# ---------------------------------------------------------------------------
# Guided Demo Mode (Phase 7A)
# ---------------------------------------------------------------------------
with st.expander("🎯 Try a scenario", expanded=False):
    st.caption("Each button submits a real query through the chat below — nothing here is a canned answer.")
    demo_cols = st.columns(len(DEMO_SCENARIOS))
    for demo_col, scenario in zip(demo_cols, DEMO_SCENARIOS):
        with demo_col:
            if st.button(
                scenario["label"],
                key=f"demo_{scenario['label']}",
                help=scenario["query"],
                use_container_width=True,
            ):
                st.session_state[_PENDING_DEMO_QUERY_KEY] = scenario["query"]
                st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # Display metadata if it's an assistant response
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]

            # Status Indicators
            indicators = []
            if meta.get("intent"): indicators.append(f"Intent: `{meta['intent']}`")
            if meta.get("confidence_status"): indicators.append(f"Confidence: `{meta['confidence_status']}`")
            if meta.get("ticket_id"): indicators.append(f"Ticket: `{meta['ticket_id']}`")

            if indicators:
                st.caption(" | ".join(indicators))

            if meta.get("status") == "answered":
                _render_evidence_panel(meta)
            elif meta.get("status") == "escalated":
                _render_escalation_panel(meta)

# Chat Input
typed_prompt = st.chat_input("Ask me about AdtU admissions, fees, or facilities...")
pending_demo_prompt = st.session_state.pop(_PENDING_DEMO_QUERY_KEY, None)
submitted_prompt = pending_demo_prompt or typed_prompt

if submitted_prompt:
    _handle_user_query(submitted_prompt)
