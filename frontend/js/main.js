// AdtU Campus Copilot — entry point. Wires DOM events to the render/api
// modules; no framework, no build step. Loaded as a single
// `<script type="module">` from index.html.
import { checkReady, postChat } from "./api.js";
import { appState } from "./state.js";
import { renderHomeHero } from "./render-home.js";
import { appendUserMessage, appendLoadingMessage, appendAssistantMessage } from "./render-chat.js";
import { loadAndRenderAdmin } from "./render-admin.js";
import { initTheme } from "./theme.js";

const refs = {
  statusPill: document.getElementById("status-pill"),
  themeToggle: document.getElementById("theme-toggle"),
  screenChat: document.getElementById("screen-chat"),
  screenAdmin: document.getElementById("screen-admin"),
  homeHero: document.getElementById("home-hero"),
  transcript: document.getElementById("chat-transcript"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  newChatBtn: document.getElementById("new-chat-btn"),
  navAdminBtn: document.getElementById("nav-admin-btn"),
  navChatBtn: document.getElementById("nav-chat-btn"),
  loadTicketsBtn: document.getElementById("load-tickets-btn"),
  adminStats: document.getElementById("admin-stats"),
  adminNotice: document.getElementById("admin-notice"),
  adminError: document.getElementById("admin-error"),
  adminList: document.getElementById("admin-tickets"),
};

const adminRefs = {
  stats: refs.adminStats,
  notice: refs.adminNotice,
  error: refs.adminError,
  list: refs.adminList,
};

function updateHomeVisibility() {
  refs.homeHero.hidden = appState.hasMessages;
}

async function submitQuery(rawQuery) {
  const query = (rawQuery || "").trim();
  if (!query) return;

  appState.hasMessages = true;
  updateHomeVisibility();

  appendUserMessage(refs.transcript, query);
  const loadingRow = appendLoadingMessage(refs.transcript);

  const result = await postChat(query);

  loadingRow.remove();
  appendAssistantMessage(refs.transcript, result);
}

function showChatScreen() {
  appState.screen = "chat";
  refs.screenChat.hidden = false;
  refs.screenAdmin.hidden = true;
  refs.navAdminBtn.hidden = false;
  refs.navChatBtn.hidden = true;
}

function showAdminScreen() {
  appState.screen = "admin";
  refs.screenChat.hidden = true;
  refs.screenAdmin.hidden = false;
  refs.navAdminBtn.hidden = true;
  refs.navChatBtn.hidden = false;
}

function renderStatus(backend) {
  refs.statusPill.classList.remove("status-ready", "status-not-ready", "status-offline");
  if (backend.state === "ready") {
    refs.statusPill.classList.add("status-ready");
    refs.statusPill.textContent = "✅ Backend: Ready";
  } else if (backend.state === "not_ready") {
    refs.statusPill.classList.add("status-not-ready");
    const failed = backend.failed.length ? backend.failed.join(", ") : "one or more dependencies";
    refs.statusPill.textContent = `⚠️ Backend: Running but not ready — ${failed}`;
  } else {
    refs.statusPill.classList.add("status-offline");
    refs.statusPill.textContent = "🚨 Backend: Offline or Unreachable";
  }
}

async function refreshStatus() {
  renderStatus(await checkReady());
}

function init() {
  initTheme(refs.themeToggle);

  renderHomeHero(refs.homeHero, { onSuggestionClick: submitQuery });
  updateHomeVisibility();

  refs.chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const value = refs.chatInput.value;
    refs.chatInput.value = "";
    submitQuery(value);
  });

  refs.newChatBtn.addEventListener("click", () => {
    appState.hasMessages = false;
    refs.transcript.replaceChildren();
    updateHomeVisibility();
    showChatScreen();
  });

  refs.navAdminBtn.addEventListener("click", showAdminScreen);
  refs.navChatBtn.addEventListener("click", showChatScreen);

  refs.loadTicketsBtn.addEventListener("click", () => {
    loadAndRenderAdmin(adminRefs);
  });

  refreshStatus();
}

init();
