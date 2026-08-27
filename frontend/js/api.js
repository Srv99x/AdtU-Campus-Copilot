// AdtU Campus Copilot — backend API wrapper.
//
// Mirrors the exact error-handling contract of app/ui/streamlit_app.py's
// check_backend_status()/send_chat_query()/fetch_tickets()/resolve_ticket():
// network/timeout/5xx/validation failures are always caught and turned into
// a safe, generic message -- never a raw exception string, never a guessed
// field. Every function here calls the real FastAPI backend at
// API_BASE_URL; nothing in this module fabricates a response.
import { API_BASE_URL } from "./config.js";

const TIMEOUT_MS = 30000;

class TimeoutError extends Error {}

async function request(path, options = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    return await fetch(`${API_BASE_URL}${path}`, { ...options, signal: controller.signal });
  } catch (err) {
    if (err.name === "AbortError") {
      throw new TimeoutError("Request timed out.");
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Resolve the backend into one of three operator-facing states, exactly
 * mirroring check_backend_status() in the Streamlit build: "offline"
 * (unreachable), "not_ready" (alive but a dependency check failed), or
 * "ready". Uses /ready, not /health, so a missing Chroma collection or
 * Gemini key is never hidden behind a green badge.
 */
export async function checkReady() {
  let response;
  try {
    response = await request("/ready", { method: "GET" });
  } catch {
    return { state: "offline", failed: [] };
  }

  let payload = {};
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }

  if (response.status === 200 && payload.status === "ready") {
    return { state: "ready", failed: [] };
  }

  const checks = payload.checks || {};
  const failed = Object.entries(checks)
    .filter(([, result]) => result && typeof result === "object" && !result.ok)
    .map(([name]) => name);
  return { state: "not_ready", failed };
}

/** POST /chat — the single real source of every answer/citation/confidence/ticket shown in the UI. */
export async function postChat(query) {
  let response;
  try {
    response = await request("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
  } catch (err) {
    if (err instanceof TimeoutError) {
      return { status: "error", reason: "Request timed out. The backend is taking too long to respond." };
    }
    return { status: "error", reason: "Backend is unreachable. Please verify the API server is running." };
  }

  if (response.status === 422) {
    return { status: "error", reason: "Query rejected by server validation (e.g. too long)." };
  }
  if (response.status >= 500) {
    return { status: "error", reason: "Backend internal error. Please try again later." };
  }
  if (!response.ok) {
    return { status: "error", reason: "Backend rejected the request." };
  }
  try {
    return await response.json();
  } catch {
    return { status: "error", reason: "Backend returned an unreadable response." };
  }
}

/** GET /tickets — real ticket list for the Staff/Admin workspace. */
export async function getTickets() {
  let response;
  try {
    response = await request("/tickets", { method: "GET" });
  } catch (err) {
    if (err instanceof TimeoutError) {
      return { ok: false, tickets: [], error: "Request timed out while loading tickets." };
    }
    return { ok: false, tickets: [], error: "Backend is unreachable. Please verify the API server is running." };
  }

  if (response.status >= 500) {
    return { ok: false, tickets: [], error: "Backend internal error while loading tickets." };
  }
  if (!response.ok) {
    return { ok: false, tickets: [], error: "Backend rejected the request." };
  }
  try {
    return { ok: true, tickets: await response.json(), error: null };
  } catch {
    return { ok: false, tickets: [], error: "Backend returned an unreadable response." };
  }
}

/** PATCH /tickets/{id} — the only exposed transition is open -> resolved, matching the backend's own restriction. */
export async function resolveTicket(ticketId) {
  let response;
  try {
    response = await request(`/tickets/${encodeURIComponent(ticketId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: "resolved" }),
    });
  } catch (err) {
    if (err instanceof TimeoutError) {
      return { ok: false, ticket: null, error: "Request timed out while resolving the ticket." };
    }
    return { ok: false, ticket: null, error: "Backend is unreachable. Please verify the API server is running." };
  }

  if (response.status === 404) {
    return { ok: false, ticket: null, error: "Ticket not found." };
  }
  if (response.status === 422) {
    return { ok: false, ticket: null, error: "Ticket update rejected by server validation." };
  }
  if (response.status >= 500) {
    return { ok: false, ticket: null, error: "Backend internal error while resolving the ticket." };
  }
  if (!response.ok) {
    return { ok: false, ticket: null, error: "Backend rejected the request." };
  }
  try {
    return { ok: true, ticket: await response.json(), error: null };
  } catch {
    return { ok: false, ticket: null, error: "Backend returned an unreadable response." };
  }
}
