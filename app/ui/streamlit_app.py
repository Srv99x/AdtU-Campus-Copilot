"""
AdtU Campus Copilot — Streamlit MVP
A lightweight UI wrapping the FastAPI orchestration backend.
"""
import os
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT_SEC = 30

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

# ---------------------------------------------------------------------------
# UI Rendering
# ---------------------------------------------------------------------------

st.title("AdtU Campus Copilot")
st.markdown("##### Official institutional assistant for Assam down town University")

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
            
            # Citations (expandable)
            citations = meta.get("citations")
            if citations:
                with st.expander("Sources"):
                    for c in citations:
                        line = f"- [{c['section']}]({c['source_url']})"
                        if c.get("parent_chunk_id"):
                            line += f" (via `{c['parent_chunk_id']}`)"
                        st.markdown(line)

# Chat Input
if prompt := st.chat_input("Ask me about AdtU admissions, fees, or facilities..."):
    if not prompt.strip():
        st.warning("Please enter a valid query.")
        st.stop()
        
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
            "intent": result.get("intent"),
            "confidence_status": result.get("confidence_status"),
            "ticket_id": result.get("ticket_id"),
            "citations": result.get("citations")
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
                
            if metadata["citations"]:
                with st.expander("Sources"):
                    for c in metadata["citations"]:
                        line = f"- [{c['section']}]({c['source_url']})"
                        if c.get("parent_chunk_id"):
                            line += f" (via `{c['parent_chunk_id']}`)"
                        st.markdown(line)

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

        elif status == "error":
            msg = f"An error occurred: {result.get('reason', 'Unknown error')}"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "metadata": metadata})
        else:
            msg = f"Unexpected response status: {status}"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "metadata": metadata})
