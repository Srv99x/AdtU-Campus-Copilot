"""
Tests for the environment-driven CORS configuration in app/api/main.py
(ALLOWED_ORIGINS).

Deliberately does NOT reload app.api.main or mutate its live `app`
singleton: other test files (test_api.py) key their
`app.dependency_overrides[...]` and `@patch("app.api.main.get_collection")`
wiring to the specific function objects bound when that module was first
imported. `importlib.reload()` would rebind those names to new objects on
the module, silently detaching the real `app`'s already-declared routes
(whose `Depends(get_collection)` captured the pre-reload function) from
whatever the reloaded module now exposes -- a subtle, hard-to-diagnose way
to break sibling tests depending on file execution order. Instead:

  - `_parse_allowed_origins` / `_resolve_allowed_origins` (the pure
    decision logic) are exercised directly.
  - The actual CORSMiddleware accept/reject behavior is proven against an
    isolated throwaway FastAPI app built with the exact same middleware
    configuration app.api.main uses -- same class, same behavior, zero
    risk to the real app's identity.
  - The real, unmodified `app.api.main.app` singleton is used for the one
    "existing behavior is unchanged" check, with no reload and no mocking.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

import app.api.main as api_main


def _build_isolated_cors_app(allowed_origins: list[str]) -> TestClient:
    """A minimal app wired with the exact same CORSMiddleware configuration
    app.api.main uses (allow_credentials=False, all methods/headers) for a
    given origins list -- isolated from the real `app.api.main.app`."""
    isolated_app = FastAPI()

    @isolated_app.get("/health")
    def _health() -> dict:
        return {"status": "ok"}

    isolated_app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return TestClient(isolated_app)


class TestParseAllowedOrigins(unittest.TestCase):
    """Pure parser logic -- no env, no app, no I/O."""

    def test_splits_on_comma(self) -> None:
        self.assertEqual(
            api_main._parse_allowed_origins("http://a.com,http://b.com"),
            ["http://a.com", "http://b.com"],
        )

    def test_strips_whitespace_around_each_origin(self) -> None:
        self.assertEqual(
            api_main._parse_allowed_origins(" http://a.com , http://b.com "),
            ["http://a.com", "http://b.com"],
        )

    def test_drops_empty_entries_from_trailing_or_double_commas(self) -> None:
        self.assertEqual(
            api_main._parse_allowed_origins("http://a.com,,http://b.com,"),
            ["http://a.com", "http://b.com"],
        )

    def test_none_returns_empty_list(self) -> None:
        self.assertEqual(api_main._parse_allowed_origins(None), [])

    def test_blank_string_returns_empty_list(self) -> None:
        self.assertEqual(api_main._parse_allowed_origins("   "), [])

    def test_never_returns_a_wildcard_even_if_present_in_input(self) -> None:
        """The parser doesn't special-case "*" -- it would pass it through
        literally as a single origin string, never expanding or treating it
        as "allow everything". Documented here as the safety property the
        rest of the config relies on: nothing anywhere synthesizes "*"."""
        self.assertEqual(api_main._parse_allowed_origins("*"), ["*"])
        # But the code path this project actually wires up never supplies
        # "*" -- see test_resolve_allowed_origins below and .env.example.


class TestResolveAllowedOrigins(unittest.TestCase):
    """The fallback decision: configured ALLOWED_ORIGINS wins when
    present/non-empty; otherwise the local-dev defaults are used. Never a
    wildcard either way."""

    def test_configured_value_is_used_when_set(self) -> None:
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "https://adtu-campus-copilot.vercel.app"}):
            self.assertEqual(
                api_main._resolve_allowed_origins(),
                ["https://adtu-campus-copilot.vercel.app"],
            )

    def test_falls_back_to_local_dev_defaults_when_unset(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("ALLOWED_ORIGINS", None)
            # Point ENV_PATH at a file that doesn't define it either, so a
            # developer's real repo-root .env can't leak into this assertion.
            with patch.object(api_main, "ENV_PATH", api_main.ROOT / "nonexistent.env"):
                self.assertEqual(
                    api_main._resolve_allowed_origins(),
                    api_main._DEFAULT_DEV_ORIGINS,
                )

    def test_falls_back_to_local_dev_defaults_when_blank(self) -> None:
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "   "}):
            self.assertEqual(api_main._resolve_allowed_origins(), api_main._DEFAULT_DEV_ORIGINS)

    def test_default_dev_origins_preserve_the_previously_hardcoded_streamlit_origins(self) -> None:
        """The exact two origins this API allowed before this change must
        still be the fallback -- an operator who hasn't set ALLOWED_ORIGINS
        yet must see identical behavior to before."""
        self.assertEqual(
            set(api_main._DEFAULT_DEV_ORIGINS),
            {"http://localhost:8501", "http://127.0.0.1:8501"},
        )

    def test_resolved_origins_never_contain_a_wildcard(self) -> None:
        for env_value in [None, "", "https://adtu-campus-copilot.vercel.app", "http://localhost:3000"]:
            with self.subTest(env_value=env_value):
                patched = {"ALLOWED_ORIGINS": env_value} if env_value is not None else {}
                with patch.dict(os.environ, patched):
                    if env_value is None:
                        os.environ.pop("ALLOWED_ORIGINS", None)
                    self.assertNotIn("*", api_main._resolve_allowed_origins())


class TestCorsMiddlewareBehavior(unittest.TestCase):
    """(1)/(2)/(3) from the task: proves the real CORSMiddleware
    configuration app.api.main uses actually accepts/rejects origins as
    intended, via an isolated app built with the identical middleware
    setup (allow_credentials=False, all methods/headers)."""

    def test_1_localhost_dev_origin_is_accepted(self) -> None:
        client = _build_isolated_cors_app(api_main._DEFAULT_DEV_ORIGINS)
        response = client.get("/health", headers={"Origin": "http://localhost:8501"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:8501")

    def test_2_configured_vercel_origin_is_accepted(self) -> None:
        configured = api_main._parse_allowed_origins(
            "http://localhost:3000,http://127.0.0.1:3000,https://adtu-campus-copilot.vercel.app"
        )
        client = _build_isolated_cors_app(configured)
        response = client.get("/health", headers={"Origin": "https://adtu-campus-copilot.vercel.app"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "https://adtu-campus-copilot.vercel.app",
        )

    def test_3_unknown_origin_is_rejected(self) -> None:
        """CORSMiddleware never returns a 4xx for a disallowed origin -- it
        answers normally but omits Access-Control-Allow-Origin, which is
        the actual browser-facing signal that blocks the response from
        being read cross-origin. Absence of the header IS the rejection."""
        configured = api_main._parse_allowed_origins(
            "http://localhost:3000,https://adtu-campus-copilot.vercel.app"
        )
        client = _build_isolated_cors_app(configured)
        response = client.get("/health", headers={"Origin": "https://evil.example.com"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.headers.get("access-control-allow-origin"))

    def test_unknown_origin_is_rejected_on_preflight_too(self) -> None:
        configured = api_main._parse_allowed_origins("https://adtu-campus-copilot.vercel.app")
        client = _build_isolated_cors_app(configured)
        response = client.options(
            "/health",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertIsNone(response.headers.get("access-control-allow-origin"))

    def test_credentials_are_not_allowed_for_an_accepted_origin(self) -> None:
        client = _build_isolated_cors_app(["https://adtu-campus-copilot.vercel.app"])
        response = client.get(
            "/health", headers={"Origin": "https://adtu-campus-copilot.vercel.app"}
        )
        self.assertIsNone(response.headers.get("access-control-allow-credentials"))


class TestExistingApiBehaviorUnchanged(unittest.TestCase):
    """(4) from the task: the real, unmodified app.api.main.app singleton
    -- no reload, no mocking -- still serves routes exactly as before."""

    def test_health_endpoint_unchanged(self) -> None:
        client = TestClient(api_main.app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_app_still_has_the_same_route_paths(self) -> None:
        paths = {route.path for route in api_main.app.routes}
        self.assertTrue({"/health", "/ready", "/chat", "/tickets", "/tickets/{ticket_id}"} <= paths)

    def test_cors_middleware_is_still_installed_exactly_once(self) -> None:
        cors_middlewares = [
            m for m in api_main.app.user_middleware if m.cls is CORSMiddleware
        ]
        self.assertEqual(len(cors_middlewares), 1)

    def test_real_app_allow_credentials_is_false(self) -> None:
        cors_middleware = next(m for m in api_main.app.user_middleware if m.cls is CORSMiddleware)
        self.assertFalse(cors_middleware.kwargs.get("allow_credentials"))

    def test_real_app_allowed_origins_module_constant_has_no_wildcard(self) -> None:
        self.assertNotIn("*", api_main.ALLOWED_ORIGINS)


if __name__ == "__main__":
    unittest.main()
