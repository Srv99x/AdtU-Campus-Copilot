// AdtU Campus Copilot — light/dark theme toggle.
//
// No external UI framework: a `data-theme` attribute on <html> selects
// between the two CSS variable blocks defined in css/styles.css, persisted
// to localStorage so the choice survives a reload. Storage access is
// wrapped defensively (private browsing / disabled storage) so a blocked
// localStorage never breaks the toggle itself -- it just won't persist.
const STORAGE_KEY = "cc-theme";

function safeGet(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Storage unavailable (private browsing, disabled) -- theme choice
    // simply won't persist across reloads; not a functional failure.
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
}

export function initTheme(toggleBtn) {
  const stored = safeGet(STORAGE_KEY);
  const theme = stored === "dark" ? "dark" : "light";
  applyTheme(theme);

  toggleBtn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
    const next = current === "dark" ? "light" : "dark";
    applyTheme(next);
    safeSet(STORAGE_KEY, next);
  });
}
