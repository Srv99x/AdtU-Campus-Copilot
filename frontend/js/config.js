// AdtU Campus Copilot — frontend runtime configuration.
//
// The only value the static frontend needs from its environment: the public
// base URL of the FastAPI backend (Render in production, local uvicorn in
// development). This is a hostname, not a secret -- never put GEMINI_API_KEY
// or any Gemini model name here. The backend never returns the key's value
// (only presence/absence via GET /ready), and Gemini is called exclusively
// server-side in app/rag/generator.py.
//
// Change this one line when promoting to a deployed Render backend, e.g.:
//   export const API_BASE_URL = "https://adtu-campus-copilot.onrender.com";
export const API_BASE_URL = "https://adtu-campus-copilot.onrender.com";
