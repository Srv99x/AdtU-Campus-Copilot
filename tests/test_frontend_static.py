"""
Lightweight structural checks for frontend/ (the vanilla HTML/CSS/JS
production frontend). Deliberately NOT a JS test framework -- this reuses
pytest (already the project's test runner) to verify the concrete minimum
this task called for: the HTML parses, every JS module is syntactically
valid and every local path it references actually resolves, no Gemini
key/model name leaks into client code, no dependency on support.js/the
Claude Design runtime remains, and none of the design mockup's known
fabricated content (fake ticket IDs, fake answer text, fake timers) was
carried over. These are static-content assertions only -- they don't spin
up a browser or execute the JS.
"""
from __future__ import annotations

import html.parser
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
INDEX_HTML = FRONTEND / "index.html"

JS_FILES = [
    "js/config.js",
    "js/api.js",
    "js/state.js",
    "js/render-home.js",
    "js/render-chat.js",
    "js/render-admin.js",
    "js/theme.js",
    "js/main.js",
]

# Literal fabricated content from the approved Claude Design mockup
# (Campus Copilot.dc.html) that must never appear in the production
# frontend -- fake ticket IDs, fake answer body, fake source cards, fake
# timestamps, and the fake 1400ms streaming timer.
FORBIDDEN_MOCK_STRINGS = [
    "A7F2C1", "B31D92", "C82A14", "D94E77", "F19B55",  # fake ticket IDs
    "Showing 1 to 5 of 5 tickets",
    "Academic Records:", "Entrance Exam Documents:",  # fake static answer body
    "Undergraduate Admissions",  # fake static source card
    "10:24 AM", "10:27 AM",  # fake static timestamps
    "1400",  # the design's fake streaming setTimeout delay (ms)
]


class _LenientHTMLValidator(html.parser.HTMLParser):
    """Collects parse errors instead of raising -- html.parser is very
    forgiving by design, so this exists to surface unclosed/mismatched tags
    if they occur, without depending on an external HTML validator."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []
        self._open_stack: list[str] = []

    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID_TAGS:
            self._open_stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID_TAGS:
            return
        if not self._open_stack or self._open_stack[-1] != tag:
            self.errors.append(f"mismatched closing tag </{tag}>")
            return
        self._open_stack.pop()


class TestIndexHtmlParses(unittest.TestCase):
    def test_file_exists(self) -> None:
        self.assertTrue(INDEX_HTML.is_file())

    def test_parses_without_tag_mismatches(self) -> None:
        content = INDEX_HTML.read_text(encoding="utf-8")
        validator = _LenientHTMLValidator()
        validator.feed(content)
        self.assertEqual(validator.errors, [])

    def test_loads_main_js_as_a_module(self) -> None:
        content = INDEX_HTML.read_text(encoding="utf-8")
        self.assertRegex(content, r'<script[^>]+type="module"[^>]+src="js/main\.js"')


class TestNoLocalPathsAreBroken(unittest.TestCase):
    """Every local file this page/its modules reference must actually
    exist on disk -- except the ADTU logo, which is a documented, expected
    placeholder (see frontend/assets/README.md)."""

    def test_stylesheet_referenced_and_present(self) -> None:
        content = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn('href="css/styles.css"', content)
        self.assertTrue((FRONTEND / "css" / "styles.css").is_file())

    def test_all_js_modules_present_on_disk(self) -> None:
        for rel_path in JS_FILES:
            with self.subTest(rel_path=rel_path):
                self.assertTrue((FRONTEND / rel_path).is_file())

    def test_relative_imports_between_js_modules_resolve(self) -> None:
        import_re = re.compile(r'from\s+["\'](\./[^"\']+)["\']')
        for rel_path in JS_FILES:
            js_file = FRONTEND / rel_path
            content = js_file.read_text(encoding="utf-8")
            for match in import_re.finditer(content):
                imported = (js_file.parent / match.group(1)).resolve()
                with self.subTest(source=rel_path, imported=match.group(1)):
                    self.assertTrue(imported.is_file())

    def test_logo_path_is_a_documented_placeholder_not_a_silent_gap(self) -> None:
        content = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn('src="assets/adtu-logo.png"', content)
        # The missing asset must degrade gracefully, not silently break.
        self.assertIn("addEventListener(\"error\"", content)
        self.assertTrue((FRONTEND / "assets" / "README.md").is_file())


class TestJsModulesAreSyntacticallyValid(unittest.TestCase):
    """`node --check` parses each file without executing it -- no test
    framework required, just Node's own parser."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.node = shutil.which("node")

    def test_every_module_passes_node_check(self) -> None:
        if not self.node:
            self.skipTest("node not found on PATH; syntax not verified")
        for rel_path in JS_FILES:
            js_file = FRONTEND / rel_path
            with self.subTest(rel_path=rel_path):
                result = subprocess.run(
                    [self.node, "--check", str(js_file)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode, 0,
                    f"node --check failed for {rel_path}:\n{result.stderr}",
                )


class TestApiCallsUseConfiguredBaseUrl(unittest.TestCase):
    def test_config_defines_api_base_url(self) -> None:
        content = (FRONTEND / "js" / "config.js").read_text(encoding="utf-8")
        self.assertRegex(content, r'export const API_BASE_URL\s*=')

    def test_api_module_imports_and_uses_api_base_url(self) -> None:
        content = (FRONTEND / "js" / "api.js").read_text(encoding="utf-8")
        self.assertIn('from "./config.js"', content)
        self.assertIn("API_BASE_URL", content)
        # Every fetch() call must build its URL from the configured base,
        # never a second hardcoded origin.
        fetch_calls = re.findall(r"fetch\(([^,)]+)", content)
        self.assertTrue(fetch_calls, "expected at least one fetch() call in api.js")
        for call_arg in fetch_calls:
            with self.subTest(call_arg=call_arg):
                self.assertIn("API_BASE_URL", call_arg)

    def test_no_other_module_calls_fetch_directly(self) -> None:
        """All network access must go through api.js -- render/main modules
        call its exported functions, never fetch() themselves."""
        for rel_path in JS_FILES:
            if rel_path == "js/api.js":
                continue
            content = (FRONTEND / rel_path).read_text(encoding="utf-8")
            with self.subTest(rel_path=rel_path):
                self.assertNotIn("fetch(", content)


_JS_LINE_COMMENT_RE = re.compile(r"//.*")
_JS_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def _strip_comments(rel_path: str, content: str) -> str:
    """Strip comments before scanning for real code -- explanatory prose
    like "never put GEMINI_API_KEY here" documenting the security boundary
    is exactly what this codebase should say, not a leak; only actual
    executable/markup code should ever be free of the word."""
    if rel_path.endswith((".js",)):
        content = _JS_BLOCK_COMMENT_RE.sub("", content)
        content = _JS_LINE_COMMENT_RE.sub("", content)
    elif rel_path.endswith((".css",)):
        content = _JS_BLOCK_COMMENT_RE.sub("", content)
    elif rel_path.endswith((".html",)):
        content = _HTML_COMMENT_RE.sub("", content)
    return content


class TestNoSecretsOrDesignRuntimeLeak(unittest.TestCase):
    def _all_frontend_text(self) -> dict[str, str]:
        texts: dict[str, str] = {}
        for path in FRONTEND.rglob("*"):
            if path.is_file() and path.suffix in {".html", ".css", ".js", ".json", ".md"}:
                texts[str(path.relative_to(FRONTEND))] = path.read_text(encoding="utf-8")
        return texts

    def test_no_gemini_key_or_model_names_in_frontend_code(self) -> None:
        """Scans CODE only (comments/markdown stripped) -- a comment
        explaining the security boundary ("never put the key here") is
        expected and correct; an actual value or reference in executable
        code/markup would not be."""
        for rel_path, content in self._all_frontend_text().items():
            if rel_path.endswith(".md"):
                continue  # documentation is expected to discuss the boundary in prose
            code_only = _strip_comments(rel_path, content)
            with self.subTest(rel_path=rel_path):
                self.assertNotIn("GEMINI", code_only.upper())

    def test_no_support_js_or_dc_runtime_dependency(self) -> None:
        forbidden = ["support.js", "dc-runtime", "<x-dc", "sc-if", "DCLogic", "x-import"]
        for rel_path, content in self._all_frontend_text().items():
            for token in forbidden:
                with self.subTest(rel_path=rel_path, token=token):
                    self.assertNotIn(token, content)

    def test_no_react_or_unpkg_dependency(self) -> None:
        for rel_path, content in self._all_frontend_text().items():
            with self.subTest(rel_path=rel_path):
                self.assertNotIn("unpkg.com", content)
                self.assertNotIn("react-dom", content.lower())


class TestNoFabricatedMockDataCarriedOver(unittest.TestCase):
    def test_none_of_the_design_mockups_fabricated_strings_appear(self) -> None:
        for path in FRONTEND.rglob("*"):
            if not (path.is_file() and path.suffix in {".html", ".css", ".js"}):
                continue
            content = path.read_text(encoding="utf-8")
            for needle in FORBIDDEN_MOCK_STRINGS:
                with self.subTest(rel_path=str(path.relative_to(FRONTEND)), needle=needle):
                    self.assertNotIn(needle, content)

    def test_home_suggestions_are_the_six_approved_real_queries(self) -> None:
        content = (FRONTEND / "js" / "render-home.js").read_text(encoding="utf-8")
        approved_queries = [
            "What is the minimum eligibility for B.Sc. Microbiology at AdtU?",
            "What is the total programme fee for B.Pharm at AdtU?",
            "What scholarship is available for CBSE board students with 95%?",
            "What are the names of the girls hostel blocks at AdtU?",
            "Which room is the B.Tech CSE DS and AI IBM Section A first semester class held in?",
            "How can I search for a book in the AdtU library?",
        ]
        for query in approved_queries:
            with self.subTest(query=query):
                self.assertIn(query, content)

    def test_safety_demo_query_is_separate_from_the_verified_cards(self) -> None:
        """The one deliberately-unanswerable question must live in its own
        SAFETY_DEMO export, never inside HOME_SUGGESTIONS, so it can never
        be read as an ordinary successful example."""
        content = (FRONTEND / "js" / "render-home.js").read_text(encoding="utf-8")
        safety_query = "What is the WiFi password for the boys hostel?"
        self.assertIn("SAFETY_DEMO", content)
        self.assertIn(safety_query, content)
        suggestions_block = content[
            content.index("export const HOME_SUGGESTIONS"):content.index("export const SAFETY_DEMO")
        ]
        self.assertNotIn(safety_query, suggestions_block)

    def test_loading_state_has_no_fake_delay_timer(self) -> None:
        content = (FRONTEND / "js" / "render-chat.js").read_text(encoding="utf-8")
        self.assertNotIn("setTimeout", content)
        content_main = (FRONTEND / "js" / "main.js").read_text(encoding="utf-8")
        self.assertNotIn("setTimeout", content_main)


if __name__ == "__main__":
    unittest.main()
