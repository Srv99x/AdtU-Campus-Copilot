// AdtU Campus Copilot — Staff/Admin ticket workspace.
//
// Renders ONLY fields the real GET/PATCH /tickets API actually returns
// (ticket_id, query, predicted_intent, status, created_at, source,
// user_metadata) -- there is no staff name, resolution timestamp,
// priority, or SLA in the ticket model, so none is shown or invented.
// Mirrors _render_admin_panel()/_load_admin_tickets() in
// app/ui/streamlit_app.py.
import { getTickets, resolveTicket } from "./api.js";

/**
 * Format a real created_at timestamp for display without changing the
 * underlying value -- if it can't be parsed, show the raw string verbatim
 * rather than fabricating a friendly date.
 */
function formatCreatedAt(rawTimestamp) {
  if (!rawTimestamp) return "";
  const date = new Date(rawTimestamp);
  if (Number.isNaN(date.getTime())) return rawTimestamp;
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function buildStatTile(label, value) {
  const tile = document.createElement("div");
  tile.className = "card stat-tile";
  const val = document.createElement("div");
  val.className = "stat-value";
  val.textContent = String(value);
  const lab = document.createElement("div");
  lab.className = "stat-label";
  lab.textContent = label;
  tile.append(val, lab);
  return tile;
}

function buildTicketCard(ticket, refs) {
  const card = document.createElement("div");
  card.className = "card ticket-card";

  const badge = document.createElement("span");
  badge.className = `badge ${ticket.status === "open" ? "badge-open" : "badge-resolved"}`;
  badge.textContent = ticket.status === "open" ? "🟡 OPEN" : "✅ RESOLVED";
  card.appendChild(badge);

  if (ticket.query) {
    const q = document.createElement("p");
    q.className = "ticket-query";
    q.textContent = `Query: ${ticket.query}`;
    card.appendChild(q);
  }

  const metaBits = [];
  if (ticket.predicted_intent) metaBits.push(`Intent: ${ticket.predicted_intent}`);
  if (ticket.source) metaBits.push(`Source: ${ticket.source}`);
  if (metaBits.length) {
    const meta = document.createElement("p");
    meta.className = "muted";
    meta.textContent = metaBits.join(" · ");
    card.appendChild(meta);
  }

  if (ticket.created_at) {
    const created = document.createElement("p");
    created.className = "muted";
    created.textContent = `Created: ${formatCreatedAt(ticket.created_at)}`;
    card.appendChild(created);
  }

  const idLine = document.createElement("p");
  idLine.className = "muted ticket-id";
  idLine.textContent = `ID: ${ticket.ticket_id}`;
  card.appendChild(idLine);

  if (ticket.status === "open") {
    const resolveBtn = document.createElement("button");
    resolveBtn.type = "button";
    resolveBtn.className = "btn btn-primary";
    resolveBtn.textContent = "Resolve";
    resolveBtn.addEventListener("click", async () => {
      resolveBtn.disabled = true;
      const outcome = await resolveTicket(ticket.ticket_id);
      if (outcome.ok) {
        setNotice(refs, `Ticket ${ticket.ticket_id} resolved.`);
        await loadAndRenderAdmin(refs);
      } else {
        setError(refs, outcome.error);
        resolveBtn.disabled = false;
      }
    });
    card.appendChild(resolveBtn);
  }

  return card;
}

function setNotice(refs, message) {
  refs.notice.textContent = message;
  refs.notice.hidden = false;
  refs.error.hidden = true;
}

function setError(refs, message) {
  refs.error.textContent = message;
  refs.error.hidden = false;
  refs.notice.hidden = true;
}

export function renderAdminTickets(refs, tickets) {
  refs.stats.replaceChildren();
  refs.list.replaceChildren();

  if (!tickets || tickets.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No escalation tickets yet.";
    refs.list.appendChild(empty);
    return;
  }

  const openCount = tickets.filter((t) => t.status === "open").length;
  const resolvedCount = tickets.length - openCount;

  refs.stats.append(
    buildStatTile("Open", openCount),
    buildStatTile("Resolved", resolvedCount),
    buildStatTile("Total", tickets.length),
  );

  for (const ticket of tickets) {
    refs.list.appendChild(buildTicketCard(ticket, refs));
  }
}

export async function loadAndRenderAdmin(refs) {
  const result = await getTickets();
  if (!result.ok) {
    setError(refs, result.error);
    return;
  }
  refs.error.hidden = true;
  renderAdminTickets(refs, result.tickets);
}
