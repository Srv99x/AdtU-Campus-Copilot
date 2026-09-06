// AdtU Campus Copilot — home/empty-state hero, the six verified suggestion
// cards, and one separate safety demonstration.
//
// Clicking any card (verified or safety) submits its query through the exact
// same onSuggestionClick callback as manually typed chat input -- there is no
// second, parallel answer path, and no answer text is ever stored here.
//
// SELECTION RULE for HOME_SUGGESTIONS (Innovation Mela demo pack): every
// query below was checked against the live 957-vector `adtu_knowledge`
// collection before being listed -- classifier intent, retrieval category,
// the chunk that actually holds the answer, and the Stage 1 gate margin
// against GATE_THETA_D = 0.275 (lower dp-top3-mean = stronger). Questions the
// KB cannot actually answer were removed rather than kept for looks: the
// previous set asked for hostel *amenities* (0 chunks in the corpus contain
// any), an admission *document checklist* (not in the corpus), and the *next*
// holiday (needs date arithmetic the grounding prompt forbids).
//
// The `evidence` note on each entry is developer documentation only. It is
// never read by renderHomeHero and never rendered in the UI.
export const HOME_SUGGESTIONS = [
  {
    icon: "📄",
    query: "What is the minimum eligibility for B.Sc. Microbiology at AdtU?",
    // ews2025_43 (admissions) rank 1 -- "Seat Available for the session
    // 2024-25" table: "BSc. Microbiology | 60% in 10+2 with English, Biology".
    // intent=admissions, dp-top3-mean 0.2190 (margin +0.0560).
    evidence: "ews2025_43",
  },
  {
    icon: "💰",
    query: "What is the total programme fee for B.Pharm at AdtU?",
    // v2_fees_4 (fees, official_pdf, session 2026-27) rank 1 -- B.Pharm row
    // gives per-semester fees and a total of 820,000.
    // intent=fees, dp-top3-mean 0.2138 (margin +0.0612).
    evidence: "v2_fees_4",
  },
  {
    icon: "🎓",
    query: "What scholarship is available for CBSE board students with 95%?",
    // v2_scholarships_0 (fees, official_pdf, session 2025-26) rank 1 -- row
    // 2C: "95% & ABOVE (100% SCHOLARSHIP ON SEMESTER FEE)". The "%" in the
    // query triggers the existing monetary-scholarship retrieval override, so
    // this searches the fees category where the scholarship table lives.
    // intent=facilities, dp-top3-mean 0.2671 (margin +0.0079 -- the thinnest
    // of the six, but this is the one query already proven end-to-end through
    // Gemini, returning 10 citations).
    evidence: "v2_scholarships_0",
  },
  {
    icon: "🏠",
    query: "What are the names of the girls hostel blocks at AdtU?",
    // hostel-details-html_33 (facilities) rank 1 -- "University Hostel
    // Details" table lists Girls Hostel A Block (130), E Block (185), F Block
    // (181). Block names and seat capacities are the ONLY hostel facts in the
    // corpus; amenities are not present, which is why the old "hostel
    // facilities" card correctly refused to answer.
    // intent=facilities, dp-top3-mean 0.1610 (margin +0.1140 -- strongest).
    evidence: "hostel-details-html_33",
  },
  {
    icon: "🗓️",
    query: "Which room is the B.Tech CSE DS and AI IBM Section A first semester class held in?",
    // v2_class_routine_0 (facilities, official_pdf, 2026-27) rank 1 -- header
    // line: "Block/Room No.(Floor): B424(-1 Floor), Time Table for: 1st year
    // 1st semester, 2026-30 Batch, Programme: Btech CSE [DS & AI] IBM
    // (Section A)". A single explicit fact rather than a whole timetable grid,
    // which the PDF extraction leaves fragmented.
    // intent=facilities, dp-top3-mean 0.1579 (margin +0.1171 -- strongest).
    evidence: "v2_class_routine_0",
  },
  {
    icon: "📚",
    query: "How can I search for a book in the AdtU library?",
    // hndb-library_35 (facilities) -- "Web-OPAC Facility: Search your book by
    // Book Name/Author Name/Publisher Name/ISBN No/Accession No".
    // intent=facilities, dp-top3-mean 0.2221 (margin +0.0529).
    evidence: "hndb-library_35",
  },
];

// A deliberately UNSUPPORTED question, kept separate from the verified cards
// above so it can never be mistaken for an ordinary successful answer.
//
// Campus network credentials are not a public knowledge-base item, so no
// chunk contains them. Verified against the live collection: retrieval scores
// dp-top3-mean 0.3249 against GATE_THETA_D = 0.275, so the Stage 1 gate
// rejects it and the pipeline escalates to a SQLite ticket -- demonstrating
// the non-fabrication guarantee end-to-end. It escalates at Stage 1, before
// Gemini is called, so it consumes no generation quota.
export const SAFETY_DEMO = {
  icon: "🛡️",
  query: "What is the WiFi password for the boys hostel?",
  label: "Safety demo — watch it refuse to guess",
  evidence: "(none — intentionally unanswerable)",
};

export function renderHomeHero(container, { onSuggestionClick }) {
  container.replaceChildren();

  const title = document.createElement("h1");
  title.className = "hero-title";
  title.textContent = "AdtU Campus Copilot";

  const kicker = document.createElement("p");
  kicker.className = "hero-kicker";
  kicker.textContent = "Your AI guide to Assam Down Town University";

  const divider = document.createElement("div");
  divider.className = "hero-divider";
  divider.setAttribute("aria-hidden", "true");

  const heading = document.createElement("h2");
  heading.className = "hero-heading";
  heading.textContent = "How can I help you navigate AdtU?";

  const sub = document.createElement("p");
  sub.className = "hero-sub";
  sub.textContent = "Ask about admissions, fees, scholarships, facilities, class routines and more.";

  const grid = document.createElement("div");
  grid.className = "suggestion-grid";
  for (const item of HOME_SUGGESTIONS) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "suggestion-card";

    const iconSpan = document.createElement("span");
    iconSpan.className = "suggestion-icon";
    iconSpan.textContent = item.icon;
    iconSpan.setAttribute("aria-hidden", "true");

    const textSpan = document.createElement("span");
    textSpan.className = "suggestion-text";
    textSpan.textContent = item.query;

    card.append(iconSpan, textSpan);
    card.addEventListener("click", () => onSuggestionClick(item.query));
    grid.appendChild(card);
  }

  // Safety demonstration -- visually separated from the verified cards, and
  // explicitly labelled, so a visitor never reads it as a normal example.
  const safetyRow = document.createElement("div");
  safetyRow.className = "safety-demo-row";

  const safetyNote = document.createElement("p");
  safetyNote.className = "safety-demo-note";
  safetyNote.textContent = "Ask something the university has not published:";

  const safetyCard = document.createElement("button");
  safetyCard.type = "button";
  safetyCard.className = "safety-demo-card";
  safetyCard.setAttribute("aria-label", `${SAFETY_DEMO.label}: ${SAFETY_DEMO.query}`);

  const safetyIcon = document.createElement("span");
  safetyIcon.className = "suggestion-icon";
  safetyIcon.textContent = SAFETY_DEMO.icon;
  safetyIcon.setAttribute("aria-hidden", "true");

  const safetyText = document.createElement("span");
  safetyText.className = "suggestion-text";
  safetyText.textContent = SAFETY_DEMO.query;

  safetyCard.append(safetyIcon, safetyText);
  safetyCard.addEventListener("click", () => onSuggestionClick(SAFETY_DEMO.query));
  safetyRow.append(safetyNote, safetyCard);

  container.append(title, kicker, divider, heading, sub, grid, safetyRow);
}
