// AdtU Campus Copilot — chat transcript rendering.
//
// Every branch below renders ONLY fields the real ChatResponse actually
// returned (status, intent, answer, citations, confidence_status,
// ticket_id, reason) -- exactly mirroring _handle_user_query(),
// _render_evidence_panel(), and _render_escalation_panel() in
// app/ui/streamlit_app.py, including the same five-way status branch
// (answered / out_of_scope / escalated / error / unexpected). Nothing here
// is computed, guessed, or fabricated client-side.

/**
 * Port of app/rag/generator.py's format_citation_reference(): the exact
 * same non-fabrication rule -- a clickable link when source_url is
 * present, otherwise a plain-text identity built only from metadata that
 * was actually returned (section/chunk_id/source_type), NEVER a fabricated
 * URL. Parent-chunk lineage is appended in both cases when present.
 */
function renderCitation(citation) {
  const { chunk_id, parent_chunk_id, source_url, section, source_type } = citation;
  const sectionLabel = section && section.trim() ? section.trim() : chunk_id;
  const cleanUrl = source_url ? source_url.trim() : "";

  const line = document.createElement("div");
  line.className = "citation-line";

  if (cleanUrl) {
    const link = document.createElement("a");
    link.href = cleanUrl;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = sectionLabel;
    line.appendChild(link);
  } else {
    let identity = sectionLabel !== chunk_id
      ? `${sectionLabel} (chunk \`${chunk_id}\`)`
      : `chunk \`${chunk_id}\``;
    if (source_type) identity += ` — ${source_type}`;
    const span = document.createElement("span");
    span.textContent = `${identity} — source link unavailable`;
    line.appendChild(span);
  }

  if (parent_chunk_id) {
    const via = document.createElement("span");
    via.className = "citation-via";
    via.textContent = ` (via \`${parent_chunk_id}\`)`;
    line.appendChild(via);
  }

  return line;
}

/** "Why this answer?" -- shown for a grounded (status === "answered") reply. */
function buildEvidencePanel(metadata) {
  const details = document.createElement("details");
  details.className = "panel evidence-panel";

  const summary = document.createElement("summary");
  summary.textContent = "🔎 Why this answer?";
  details.appendChild(summary);

  const body = document.createElement("div");
  body.className = "panel-body";

  const summaryBits = [];
  if (metadata.intent) summaryBits.push(`Intent: ${metadata.intent}`);
  if (metadata.confidence_status) summaryBits.push(`Confidence: ${metadata.confidence_status}`);
  const citations = metadata.citations;
  if (citations && citations.length) summaryBits.push(`Sources cited: ${citations.length}`);
  if (summaryBits.length) {
    const p = document.createElement("p");
    p.className = "muted panel-summary";
    p.textContent = summaryBits.join(" · ");
    body.appendChild(p);
  }

  if (citations && citations.length) {
    const heading = document.createElement("p");
    heading.className = "panel-subheading";
    heading.textContent = "Cited evidence";
    body.appendChild(heading);
    const list = document.createElement("div");
    list.className = "citation-list";
    for (const citation of citations) list.appendChild(renderCitation(citation));
    body.appendChild(list);
  } else {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = "No citation metadata was returned with this answer.";
    body.appendChild(p);
  }

  if (metadata.reason) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = `Backend reasoning: ${metadata.reason}`;
    body.appendChild(p);
  }

  details.appendChild(body);
  return details;
}

/** "Not enough verified evidence" -- shown for status === "escalated". */
function buildEscalationPanel(metadata) {
  const details = document.createElement("details");
  details.className = "panel evidence-panel escalation-panel";
  details.open = true;

  const summary = document.createElement("summary");
  summary.textContent = "🟡 Not enough verified evidence";
  details.appendChild(summary);

  const body = document.createElement("div");
  body.className = "panel-body";

  const principle = document.createElement("p");
  principle.className = "escalation-principle";
  principle.textContent = "Campus Copilot is designed to escalate questions when reliable evidence is unavailable.";
  body.appendChild(principle);

  if (metadata.confidence_status) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = `Retrieval confidence: ${metadata.confidence_status}`;
    body.appendChild(p);
  }
  if (metadata.reason) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = `Backend reasoning: ${metadata.reason}`;
    body.appendChild(p);
  }
  if (metadata.ticket_id) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = `A support ticket (${metadata.ticket_id}) was created for staff review.`;
    body.appendChild(p);
  }

  details.appendChild(body);
  return details;
}

export function appendUserMessage(transcript, text) {
  const row = document.createElement("div");
  row.className = "msg-row msg-row-user";
  const bubble = document.createElement("div");
  bubble.className = "bubble bubble-user";
  bubble.textContent = text;
  row.appendChild(bubble);
  transcript.appendChild(row);
  transcript.scrollTop = transcript.scrollHeight;
  return row;
}

/** Real-request loading state (no fake timer) -- caller removes the returned row once postChat() resolves. */
export function appendLoadingMessage(transcript) {
  const row = document.createElement("div");
  row.className = "msg-row msg-row-assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble bubble-assistant bubble-loading";

  const spinner = document.createElement("span");
  spinner.className = "spinner";
  spinner.setAttribute("aria-hidden", "true");

  const label = document.createElement("span");
  label.textContent = "Searching verified university sources";

  bubble.append(spinner, label);
  row.appendChild(bubble);
  transcript.appendChild(row);
  transcript.scrollTop = transcript.scrollHeight;
  return row;
}

export function appendAssistantMessage(transcript, result) {
  const row = document.createElement("div");
  row.className = "msg-row msg-row-assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble bubble-assistant";

  const status = result.status || "error";
  const metadata = {
    status,
    intent: result.intent,
    confidence_status: result.confidence_status,
    ticket_id: result.ticket_id,
    citations: result.citations,
    reason: result.reason,
  };

  if (status === "answered") {
    const text = document.createElement("p");
    text.textContent = result.answer || "No answer provided.";
    bubble.appendChild(text);

    const indicators = [];
    if (metadata.intent) indicators.push(`Intent: ${metadata.intent}`);
    if (metadata.confidence_status) indicators.push(`Confidence: ${metadata.confidence_status}`);
    if (indicators.length) {
      const cap = document.createElement("p");
      cap.className = "muted indicators";
      cap.textContent = indicators.join(" | ");
      bubble.appendChild(cap);
    }
    bubble.appendChild(buildEvidencePanel(metadata));
  } else if (status === "out_of_scope") {
    bubble.classList.add("bubble-out-of-scope");
    const text = document.createElement("p");
    text.textContent =
      "This assistant only handles AdtU-related queries (admissions, fees, facilities). " +
      "Please rephrase or ask a campus-related question.";
    bubble.appendChild(text);
    if (metadata.intent) {
      const cap = document.createElement("p");
      cap.className = "muted";
      cap.textContent = `Intent: ${metadata.intent}`;
      bubble.appendChild(cap);
    }
  } else if (status === "escalated") {
    bubble.classList.add("bubble-escalated");
    const text = document.createElement("p");
    text.textContent =
      "I could not verify the answer to your question in the official knowledge base. " +
      "A support ticket has been created for staff review.";
    bubble.appendChild(text);

    const indicators = [`Intent: ${metadata.intent || "unknown"}`, `Ticket: ${metadata.ticket_id || "n/a"}`];
    const cap = document.createElement("p");
    cap.className = "muted indicators";
    cap.textContent = indicators.join(" | ");
    bubble.appendChild(cap);

    bubble.appendChild(buildEscalationPanel(metadata));
  } else if (status === "error") {
    bubble.classList.add("bubble-error");
    const text = document.createElement("p");
    text.textContent = `An error occurred: ${result.reason || "Unknown error"}`;
    bubble.appendChild(text);
  } else {
    bubble.classList.add("bubble-error");
    const text = document.createElement("p");
    text.textContent = `Unexpected response status: ${status}`;
    bubble.appendChild(text);
  }

  row.appendChild(bubble);
  transcript.appendChild(row);
  transcript.scrollTop = transcript.scrollHeight;
  return row;
}
