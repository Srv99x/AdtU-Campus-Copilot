// AdtU Campus Copilot — minimal shared app state.
//
// Deliberately not a framework store: the chat transcript is append-only
// (each turn is rendered once by render-chat.js and never re-rendered), so
// the only state worth centralizing is "has a conversation started" (which
// screen elements are visible) and the current top-level screen. Admin
// ticket data lives in the DOM it renders into (render-admin.js re-fetches
// and re-renders on demand) rather than being duplicated here.
export const appState = {
  screen: "chat", // "chat" | "admin"
  hasMessages: false,
};
