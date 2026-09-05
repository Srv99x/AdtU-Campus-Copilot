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
# Guided Demo Mode / Home suggestions — six approved AdtU queries (Final UI
# Pass). These are the home-screen suggestion cards AND the guided-demo
# buttons: one list, one mechanism. Each entry is a real query string fed
# through the exact same send_chat_query()/_handle_user_query() path as
# manually typed chat input. No answers are hardcoded here -- clicking a
# card only submits its `query` to the real backend, same as typing it.
# ---------------------------------------------------------------------------
# Every query below is verified against the live 957-vector
# `adtu_knowledge` collection: classifier intent, retrieval category, the
# chunk that actually holds the answer, and the Stage 1 gate margin. Kept
# identical to frontend/js/render-home.js's HOME_SUGGESTIONS, where the
# supporting chunk id for each one is documented in full.
DEMO_SCENARIOS: list[dict[str, str]] = [
    {
        "label": "📄 What is the minimum eligibility for B.Sc. Microbiology at AdtU?",
        "query": "What is the minimum eligibility for B.Sc. Microbiology at AdtU?",
    },
    {
        "label": "💰 What is the total programme fee for B.Pharm at AdtU?",
        "query": "What is the total programme fee for B.Pharm at AdtU?",
    },
    {
        "label": "🎓 What scholarship is available for CBSE board students with 95%?",
        "query": "What scholarship is available for CBSE board students with 95%?",
    },
    {
        "label": "🏠 What are the names of the girls hostel blocks at AdtU?",
        "query": "What are the names of the girls hostel blocks at AdtU?",
    },
    {
        "label": "🗓️ Which room is the B.Tech CSE DS and AI IBM Section A first semester class held in?",
        "query": "Which room is the B.Tech CSE DS and AI IBM Section A first semester class held in?",
    },
    {
        "label": "📚 How can I search for a book in the AdtU library?",
        "query": "How can I search for a book in the AdtU library?",
    },
]
_PENDING_DEMO_QUERY_KEY = "_pending_demo_query"

# ---------------------------------------------------------------------------
# Staff / Admin view (Phase 7C) — session-state keys.
# ---------------------------------------------------------------------------
_ADMIN_TICKETS_KEY = "_admin_tickets"
_ADMIN_ERROR_KEY = "_admin_error"
_ADMIN_NOTICE_KEY = "_admin_notice"

st.set_page_config(page_title="AdtU Campus Copilot", layout="wide", page_icon="🎓")

# ---------------------------------------------------------------------------
# Global visual system — ADTU purple/gold on a warm off-white ground,
# Playfair Display for headings, Plus Jakarta Sans for body text.
#
# Deliberately CSS-only, layered on top of Streamlit's own generated
# elements (targeted via their stable `data-testid` attributes) rather than
# replacing them with hand-rolled HTML -- every existing st.* call below
# (labels, captions, expander titles, markdown strings) is untouched, so
# every test that asserts on that content keeps passing unchanged.
#
# Light theme only. A real runtime dark/light toggle would need to override
# Streamlit's own internal theme end-to-end and was judged too brittle for
# a demo-stability pass -- see the final report for this intentional
# omission.
# ---------------------------------------------------------------------------
_CC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
  --cc-bg:#FBFAF7; --cc-card:#FFFFFF; --cc-border:rgba(46,42,114,.14); --cc-border-soft:rgba(46,42,114,.08);
  --cc-text:#1E1B4B; --cc-text2:#2B2752; --cc-muted:#6B6787;
  --cc-brand:#2E2A72; --cc-brand-hover:#26226A; --cc-brand-tint:#4B45A8;
  --cc-accent:#F0B41C; --cc-accent-bg:#FDF4DC; --cc-accent-text:#8A6A08;
  --cc-ok-bg:#E4F5EA; --cc-ok-text:#2F7A4E;
  --cc-pill:#EAE8F8; --cc-tint:#EFEEFA;
}

[data-testid="stAppViewContainer"], [data-testid="stSidebar"], [data-testid="stHeader"] {
  background:var(--cc-bg) !important;
}
.stApp, .stApp p, .stApp span, .stApp div, .stApp label {
  font-family:'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
}
.stApp h1, .stApp h2, .stApp h3, .cc-serif {
  font-family:'Playfair Display', Georgia, serif !important; letter-spacing:-.015em; color:var(--cc-text);
}
[data-testid="stMainBlockContainer"], .block-container {
  max-width:1180px; padding-top:1.5rem;
}

[data-testid="stSidebar"] { border-right:1px solid var(--cc-border); }

/* buttons */
[data-testid="stButton"] button, button[kind="secondary"], button[kind="primary"] {
  border-radius:12px !important; font-weight:600 !important;
  transition:box-shadow .15s ease, border-color .15s ease, transform .15s ease;
}
[data-testid="stButton"] button:hover { transform:translateY(-1px); }
[data-testid="stButton"] button[kind="primary"], button[data-testid="baseButton-primary"] {
  background:var(--cc-brand) !important; border-color:var(--cc-brand) !important; color:#fff !important;
  box-shadow:0 8px 20px rgba(46,42,114,.24) !important;
}
[data-testid="stButton"] button[kind="primary"]:hover { background:var(--cc-brand-hover) !important; }
[data-testid="stButton"] button[kind="secondary"] {
  border:1px solid var(--cc-border) !important; color:var(--cc-text2) !important; background:var(--cc-card) !important;
}
[data-testid="stButton"] button[kind="secondary"]:hover {
  border-color:var(--cc-brand-tint) !important; box-shadow:0 8px 20px rgba(30,27,75,.10) !important;
}

/* status banners -> subtle pills */
[data-testid="stAlert"] {
  border-radius:999px !important; padding:.55rem 1.1rem !important; border:1px solid var(--cc-border-soft) !important;
  width:fit-content;
}

/* evidence / escalation panels */
[data-testid="stExpander"] {
  border:1px solid var(--cc-border) !important; border-radius:16px !important; background:var(--cc-card) !important;
  box-shadow:0 6px 20px rgba(30,27,75,.05) !important; overflow:hidden;
}

/* bordered containers -> ticket cards */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius:14px !important; border-color:var(--cc-border) !important;
}

/* chat bubbles */
[data-testid="stChatMessage"] {
  border-radius:16px !important; border:1px solid var(--cc-border-soft) !important; background:var(--cc-card) !important;
}

/* chat input pill */
[data-testid="stChatInput"] { border-radius:18px !important; }

[data-testid="stCaptionContainer"] { color:var(--cc-muted) !important; }

.cc-home-kicker { text-align:center; color:var(--cc-muted); font-size:.95rem; margin:.35rem 0 0; }
.cc-home-divider { width:58px; height:2px; border-radius:2px; background:var(--cc-accent); margin:16px auto 0; }
.cc-home-heading { text-align:center; margin:20px 0 0 !important; }
.cc-home-sub { text-align:center; color:var(--cc-muted); margin:8px 0 0; }
.cc-hero-title { text-align:center; margin:0; }
.cc-sidebar-brand { display:flex; flex-direction:column; gap:2px; padding:4px 2px 10px; }
.cc-sidebar-brand .cc-mark { font-family:'Playfair Display', Georgia, serif; font-weight:700; font-size:1.5rem; color:var(--cc-brand); }
.cc-sidebar-brand .cc-sub { font-size:.72rem; font-weight:700; letter-spacing:.08em; color:var(--cc-accent-text); text-transform:uppercase; }
.cc-nav-label { font-size:.68rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--cc-muted); margin:1.1rem 0 .4rem 2px; }
.cc-trust-caption { display:flex; align-items:flex-start; gap:8px; justify-content:center; text-align:center; color:var(--cc-muted); font-size:.8rem; margin:.75rem 0 0; }
</style>
"""
st.markdown(_CC_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Backend Interaction Helpers
# ---------------------------------------------------------------------------
def check_backend_status() -> dict:
    """Resolve the backend into one of three distinct operator-facing states.

    Uses /ready (not /health): /health is a static liveness ping that stays
    green even when Chroma is missing or a provider key is absent, which hid
    a broken backend behind an "Online" badge. /ready reports the real
    dependency state, and one call distinguishes all three cases:

        "offline"   -- unreachable: connection refused, timeout, DNS, etc.
        "not_ready" -- process alive but a dependency check failed
        "ready"     -- alive and every dependency check passed

    Returns {"state": str, "failed": list[str]}. `failed` names the failing
    checks (e.g. "chroma", "groq_api_key", "gemini_api_key") straight from the /ready
    payload, which reports only presence/absence -- never a key value.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/ready", timeout=3)
    except requests.exceptions.RequestException:
        return {"state": "offline", "failed": []}

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if response.status_code == 200 and payload.get("status") == "ready":
        return {"state": "ready", "failed": []}

    checks = payload.get("checks") or {}
    failed = [
        name for name, result in checks.items()
        if isinstance(result, dict) and not result.get("ok")
    ]
    return {"state": "not_ready", "failed": failed}

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
        st.markdown(
            "Campus Copilot is designed to escalate questions when reliable "
            "evidence is unavailable."
        )
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
        st.markdown('<div class="cc-nav-label">Staff workspace</div>', unsafe_allow_html=True)
        st.markdown("### 🛠️ Staff / Admin")
        st.caption("Review questions that Campus Copilot could not safely answer.")

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

        # Real, backend-derived counts -- never fabricated.
        open_count = sum(1 for t in tickets if t.get("status") == "open")
        resolved_count = len(tickets) - open_count
        stat_cols = st.columns(2)
        with stat_cols[0]:
            st.metric("Open", open_count)
        with stat_cols[1]:
            st.metric("Resolved", resolved_count)
        st.caption(f"{len(tickets)} ticket(s) total")

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

    Single code path for both manually typed chat input and the home-screen
    suggestion cards / Guided Demo Mode -- neither has its own logic; both
    call this function.
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
        with st.spinner("Searching verified university sources"):
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

# Sidebar brand + primary nav — brand mark is a styled wordmark, not an
# image: no ADTU logo asset exists anywhere in this repository (verified by
# search before implementing), and the design instructions explicitly
# forbid inventing/redrawing one, so a text mark is used instead. See the
# final report for this intentional omission.
with st.sidebar:
    st.markdown(
        '<div class="cc-sidebar-brand">'
        '<span class="cc-mark">AdtU</span>'
        '<span class="cc-sub">Campus Copilot</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.button("➕ New Chat", key="clear_chat", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Staff/Admin ticket queue (Phase 7C) — sidebar only; the student-facing
# chat experience below is unchanged.
_render_admin_panel()

# Backend status — real /ready-derived state, three distinct outcomes.
_backend = check_backend_status()
if _backend["state"] == "ready":
    st.success("Backend: Ready", icon="✅")
elif _backend["state"] == "not_ready":
    _failed = ", ".join(_backend["failed"]) if _backend["failed"] else "one or more dependencies"
    st.warning(f"Backend: Running but not ready — {_failed}", icon="⚠️")
else:
    st.error("Backend: Offline or Unreachable", icon="🚨")

# ---------------------------------------------------------------------------
# Home / empty-state hero — shown only before the first exchange, matching
# the approved design (the hero and suggestion cards give way to the
# conversation transcript once a chat is underway).
# ---------------------------------------------------------------------------
show_home = not st.session_state.messages

if show_home:
    st.markdown('<h1 class="cc-serif cc-hero-title">AdtU Campus Copilot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="cc-home-kicker">Your AI guide to Assam Down Town University</p>', unsafe_allow_html=True)
    st.markdown('<div class="cc-home-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="cc-serif cc-home-heading">How can I help you navigate AdtU?</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="cc-home-sub">Ask about admissions, fees, scholarships, facilities, class routines and more.</p>',
        unsafe_allow_html=True,
    )

    st.write("")
    demo_rows = [DEMO_SCENARIOS[0:3], DEMO_SCENARIOS[3:6]]
    for row in demo_rows:
        demo_cols = st.columns(3)
        for demo_col, scenario in zip(demo_cols, row):
            with demo_col:
                if st.button(
                    scenario["label"],
                    key=f"demo_{scenario['label']}",
                    use_container_width=True,
                ):
                    st.session_state[_PENDING_DEMO_QUERY_KEY] = scenario["query"]
                    st.rerun()

st.divider()

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
st.markdown(
    '<div class="cc-trust-caption">🎓 Answers are grounded in verified university information. '
    "If evidence is insufficient, Campus Copilot won't guess.</div>",
    unsafe_allow_html=True,
)
typed_prompt = st.chat_input("Ask anything about AdtU...")
pending_demo_prompt = st.session_state.pop(_PENDING_DEMO_QUERY_KEY, None)
submitted_prompt = pending_demo_prompt or typed_prompt

if submitted_prompt:
    _handle_user_query(submitted_prompt)
