// AdtU Campus Copilot — home/empty-state hero and the six approved
// suggestion cards. These are the exact six queries fixed by the approved
// design (Final UI Pass), matching frontend/../app/ui/streamlit_app.py's
// DEMO_SCENARIOS for consistency across both frontends. Clicking a card
// submits its query through the exact same onSuggestionClick callback as
// manually typed chat input -- there is no second, parallel answer path,
// and no answer text is ever stored here.
export const HOME_SUGGESTIONS = [
  { icon: "📄", query: "What documents are required for BTech admission at AdtU?" },
  { icon: "🎓", query: "What scholarships are available?" },
  { icon: "🗓️", query: "Show me the CSE DS & AI IBM class routine" },
  { icon: "🏠", query: "What are the hostel facilities?" },
  { icon: "📅", query: "When is the next university holiday?" },
  { icon: "💰", query: "How much are the BTech fees?" },
];

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

  container.append(title, kicker, divider, heading, sub, grid);
}
