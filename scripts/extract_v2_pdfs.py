"""
Extract V2 PDF sources to Markdown using pymupdf4llm.

Reads:
    data/raw/knowledge_v2/*.pdf

Creates:
    data/processed/knowledge_v2/markdown/*.md

Does NOT modify V1 files under data/processed/markdown/.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf
import pymupdf4llm


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "raw" / "knowledge_v2"

OUTPUT_DIR = ROOT / "data" / "processed" / "knowledge_v2" / "markdown"


# --------------------------------------------------
# SOURCE METADATA
# --------------------------------------------------

# Metadata for each PDF source, matched to V1 conventions.
# source_url is left blank when no public URL is known for
# the specific PDF document.

SOURCE_META: dict[str, dict[str, str]] = {
    "fees.pdf": {
        "source_url": "",
        "intent_category": "fees",
        "topic": "fees",
        "last_updated": "2026-03-05",
        "effective_date": "2026-27",
        "published_date": "unknown",
        "source_type": "official_pdf",
    },
    "class_routine.pdf": {
        "source_url": "",
        "intent_category": "facilities",
        "topic": "class_routine",
        "last_updated": "2026-08-08",
        "effective_date": "2026-27",
        "published_date": "unknown",
        "source_type": "official_pdf",
    },
    "academic_calendar.pdf": {
        "source_url": "",
        "intent_category": "facilities",
        "topic": "academic_calendar",
        "last_updated": "2026-07-28",
        "effective_date": "2026-27",
        "published_date": "2026-07-28",
        "source_type": "official_pdf",
    },
    "holiday_calendar.pdf": {
        "source_url": "",
        "intent_category": "facilities",
        "topic": "holiday_calendar",
        "last_updated": "2026-06-22",
        "effective_date": "2026-27",
        "published_date": "2026-06-22",
        "source_type": "official_pdf",
    },
    "scholarships.pdf": {
        "source_url": "",
        "intent_category": "fees",
        "topic": "scholarship",
        "last_updated": "2025-03-25",
        "effective_date": "2025-26",
        "published_date": "unknown",
        "source_type": "official_pdf",
    },
}


# --------------------------------------------------
# CLASS ROUTINE FILTER
# --------------------------------------------------

# Patterns that identify B.Tech CSE pages in class_routine.pdf.

CSE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Btech\s+CSE", re.IGNORECASE),
    re.compile(r"B\.?\s*Tech\s+CSE", re.IGNORECASE),
    re.compile(r"B\.?\s*Tech\s+Computer\s+Science", re.IGNORECASE),
    re.compile(r"Faculty\s+of\s+Computer\s+Technology", re.IGNORECASE),
]


def _is_cse_page(page_text: str) -> bool:
    """Return True if the page contains a B.Tech CSE routine header."""

    return any(
        pattern.search(page_text)
        for pattern in CSE_PATTERNS
    )


def _extract_cse_routine_metadata(heading: str) -> dict[str, str]:
    """Parse program, specialization, semester, section from heading.

    Example heading:
        Faculty of Computer Technology, Block/Room No.(Floor): B424(-1 Floor),
        Time Table for: 1st year 1st semester, 2026-30 Batch,
        Programme: Btech CSE [DS & AI] IBM (Section A), SS: 45
    """

    result: dict[str, str] = {
        "program": "",
        "specialization": "",
        "semester": "",
        "section": "",
    }

    # Programme extraction
    prog_match = re.search(
        r"Programme:\s*(.+?)(?:,\s*SS:|$)",
        heading,
        re.IGNORECASE,
    )
    if prog_match:
        programme_raw = prog_match.group(1).strip()
        result["program"] = programme_raw

        # Specialization: text inside brackets
        spec_match = re.search(
            r"\[(.+?)\]",
            programme_raw,
        )
        if spec_match:
            result["specialization"] = spec_match.group(1).strip()

        # Section
        sec_match = re.search(
            r"\(Section\s+([A-Z])\)",
            programme_raw,
            re.IGNORECASE,
        )
        if sec_match:
            result["section"] = sec_match.group(1).strip()

    # Semester extraction
    sem_match = re.search(
        r"(\d+\w*\s+(?:year\s+)?\d+\w*\s+semester)",
        heading,
        re.IGNORECASE,
    )
    if sem_match:
        result["semester"] = sem_match.group(1).strip()

    return result


# --------------------------------------------------
# FRONT MATTER
# --------------------------------------------------

def _build_front_matter(meta: dict[str, str]) -> str:
    """Build YAML front matter string for a V2 Markdown file."""

    lines = ["---"]

    for key, value in meta.items():
        lines.append(f"{key}: {value}")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------
# EXTRACTION
# --------------------------------------------------

def extract_pdf(pdf_path: Path) -> str:
    """Extract a PDF to Markdown using pymupdf4llm."""

    return pymupdf4llm.to_markdown(str(pdf_path))


def extract_class_routine(pdf_path: Path) -> str:
    """Extract only B.Tech CSE pages from class_routine.pdf.

    Each CSE page is extracted individually and tagged with
    structural metadata (program, specialization, semester, section).
    """

    doc = pymupdf.open(str(pdf_path))

    cse_sections: list[str] = []

    for page_num in range(doc.page_count):

        page = doc[page_num]
        page_text = page.get_text("text")

        if not _is_cse_page(page_text):
            continue

        # Extract this single page to Markdown
        page_md = pymupdf4llm.to_markdown(
            str(pdf_path),
            pages=[page_num],
        )

        # Try to parse the heading for metadata
        heading_match = re.search(
            r"#\s*\*\*(.+?)\*\*",
            page_md,
        )

        if heading_match:
            heading = heading_match.group(1)
            routine_meta = _extract_cse_routine_metadata(heading)

            meta_comment = (
                f"<!-- program: {routine_meta['program']} | "
                f"specialization: {routine_meta['specialization']} | "
                f"semester: {routine_meta['semester']} | "
                f"section: {routine_meta['section']} -->"
            )

            cse_sections.append(
                f"{meta_comment}\n\n{page_md}"
            )
        else:
            cse_sections.append(page_md)

    doc.close()

    return "\n\n---\n\n".join(cse_sections)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main() -> None:

    import sys
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print("ERROR: No PDF files found in", INPUT_DIR)
        return

    for pdf_path in pdf_files:

        filename = pdf_path.name

        if filename not in SOURCE_META:
            print(f"SKIP: {filename} (no metadata configured)")
            continue

        meta = SOURCE_META[filename]

        print(f"Extracting: {filename}")

        # Special handling for class routine
        if filename == "class_routine.pdf":
            markdown_text = extract_class_routine(pdf_path)
        else:
            markdown_text = extract_pdf(pdf_path)

        # Build front matter
        front_matter = _build_front_matter(meta)

        # Write output
        output_name = pdf_path.stem + ".md"
        output_path = OUTPUT_DIR / output_name

        output_path.write_text(
            front_matter + markdown_text,
            encoding="utf-8",
        )

        char_count = len(markdown_text)
        token_est = max(1, char_count // 4)

        print(
            f"  → {output_path.name}: "
            f"{char_count:,} chars, "
            f"~{token_est:,} tokens (raw estimate)"
        )

    print()
    print("=" * 60)
    print("PDF EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
