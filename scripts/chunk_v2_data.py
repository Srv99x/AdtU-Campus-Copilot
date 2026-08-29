"""
Chunk V2 Markdown documents for the Knowledge Base V2 layer.

Reads:
    data/processed/knowledge_v2/markdown/*.md

Creates:
    data/processed/knowledge_v2/canonical_chunks.jsonl

Uses token-aware chunking (~450-token target, ~50-token overlap)
with structural/header-aware splitting and table preservation.

Does NOT modify:
    data/processed/chunks/chunks.jsonl  (V1 immutable corpus)
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "processed" / "knowledge_v2" / "markdown"

OUTPUT_DIR = ROOT / "data" / "processed" / "knowledge_v2"

OUTPUT_FILE = OUTPUT_DIR / "canonical_chunks.jsonl"


# --------------------------------------------------
# TOKEN ESTIMATION (matches embed.py logic)
# --------------------------------------------------

# Base estimate: 4 characters per token, no safety factor
# for chunking target (safety factor applied at embedding time).
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count matching the project's convention (4 chars/token)."""
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


# --------------------------------------------------
# CHUNK SETTINGS (token-based)
# --------------------------------------------------

# Approximate ~450-token target for normal prose
TARGET_CHUNK_TOKENS = 450
CHUNK_SIZE = TARGET_CHUNK_TOKENS * CHARS_PER_TOKEN  # 1800 chars

# Approximate ~50-token overlap
TARGET_OVERLAP_TOKENS = 50
CHUNK_OVERLAP = TARGET_OVERLAP_TOKENS * CHARS_PER_TOKEN  # 200 chars


# --------------------------------------------------
# QUALITY STATUS
# --------------------------------------------------

# Patterns indicating garbled/OCR-corrupted extraction
GARBLED_PATTERNS: list[re.Pattern[str]] = [
    # Words missing spaces (common in scholarships.pdf)
    re.compile(r"[A-Z]{3,}[a-z]{3,}[A-Z]{3,}", re.ASCII),
    # Multiple consecutive uppercase words jammed together
    re.compile(r"[A-Z]{20,}"),
    # Fragmented table cells (cells with 1-2 chars of actual content,
    # not empty cells which are normal for merged row spans)
    re.compile(r"\|[^|\s]{1,2}\|[^|\s]{1,2}\|[^|\s]{1,2}\|"),
]


def detect_quality_status(text: str, source_file: str) -> str:
    """Return quality_status for a chunk.

    Returns:
        'ok' — text appears clean
        'needs_manual_review' — garbled or structurally unreliable
    """

    # Scholarships and class_routine are known problematic sources
    known_problematic = {"scholarships.md", "class_routine.md"}

    garbled_hits = sum(
        1
        for pattern in GARBLED_PATTERNS
        if pattern.search(text)
    )

    if garbled_hits >= 2:
        return "needs_manual_review"

    if source_file in known_problematic and garbled_hits >= 1:
        return "needs_manual_review"

    return "ok"


# --------------------------------------------------
# FRONT MATTER PARSING
# --------------------------------------------------

def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Parse YAML front matter and return (metadata_dict, remaining_text)."""

    meta: dict[str, str] = {}

    if not text.startswith("---"):
        return meta, text

    end_match = re.search(
        r"\n---\s*\n",
        text[3:],
    )

    if not end_match:
        return meta, text

    front_matter_text = text[3:end_match.start() + 3]
    remaining = text[end_match.end() + 3:]

    for line in front_matter_text.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()

    return meta, remaining


# --------------------------------------------------
# CLASS ROUTINE METADATA EXTRACTION
# --------------------------------------------------

def extract_routine_metadata(text: str) -> dict[str, str]:
    """Extract program/specialization/semester/section from HTML comments."""

    result: dict[str, str] = {
        "program": "",
        "specialization": "",
        "semester": "",
        "section": "",
    }

    comment_match = re.search(
        r"<!--\s*program:\s*(.+?)\s*\|\s*"
        r"specialization:\s*(.+?)\s*\|\s*"
        r"semester:\s*(.+?)\s*\|\s*"
        r"section:\s*(.+?)\s*-->",
        text,
    )

    if comment_match:
        result["program"] = comment_match.group(1).strip()
        result["specialization"] = comment_match.group(2).strip()
        result["semester"] = comment_match.group(3).strip()
        result["section"] = comment_match.group(4).strip()

    return result


# --------------------------------------------------
# TABLE DETECTION
# --------------------------------------------------

def contains_markdown_table(text: str) -> bool:
    """Detect Markdown pipe tables."""

    lines = text.splitlines()

    for i in range(len(lines) - 1):

        if (
            "|" in lines[i]
            and re.match(
                r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$",
                lines[i + 1].strip(),
            )
        ):
            return True

    return False


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize line endings and excessive whitespace."""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# SECTION SPLITTING
# --------------------------------------------------

def split_into_sections(text: str) -> list[dict[str, str]]:
    """Split Markdown by H1/H2 headers, preserving class routine boundaries."""

    # For class routine, also split on horizontal rules
    parts = re.split(r"\n---\n", text)

    sections: list[dict[str, str]] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Extract routine metadata if present
        routine_meta = extract_routine_metadata(part)

        # Remove HTML comments
        clean_part = re.sub(r"<!--.*?-->", "", part, flags=re.DOTALL).strip()
        if not clean_part:
            continue

        lines = clean_part.splitlines(keepends=True)
        current_title: str | None = None
        current_lines: list[str] = []

        for line in lines:
            header_match = re.match(
                r"^(#{1,3})\s+(.+?)\s*$",
                line,
            )

            if header_match:
                if current_lines:
                    section_entry: dict[str, str] = {
                        "section": current_title or "",
                        "text": "".join(current_lines).strip(),
                    }
                    section_entry.update(routine_meta)
                    sections.append(section_entry)

                current_title = header_match.group(2).strip()
                # Remove markdown bold markers from title
                current_title = current_title.replace("**", "").strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            section_entry = {
                "section": current_title or "",
                "text": "".join(current_lines).strip(),
            }
            section_entry.update(routine_meta)
            sections.append(section_entry)

    return sections


# --------------------------------------------------
# CHUNK NON-TABLE TEXT (token-aware)
# --------------------------------------------------

def chunk_text(text: str) -> list[str]:
    """Chunk text using token-aware sizing (~450 tokens target)."""

    text = clean_text(text)

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + CHUNK_SIZE, text_length)

        # Try to break at paragraph boundary
        if end < text_length:
            boundary = text.rfind("\n\n", start, end)

            if boundary > (start + CHUNK_SIZE // 2):
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - CHUNK_OVERLAP

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# --------------------------------------------------
# TABLE-AWARE CHUNKING
# --------------------------------------------------

def chunk_table_section(text: str) -> list[dict[str, object]]:
    """Intelligently chunk a section containing tables.

    Preserves table header + separator with each chunk.
    If a table is small enough, keeps it as one chunk.
    If oversized, splits by rows while preserving headers.
    """

    tokens = estimate_tokens(text)

    # If the whole section fits comfortably, keep it whole
    if tokens <= TARGET_CHUNK_TOKENS * 2:
        return [{"text": text, "is_table": True}]

    lines = text.splitlines(keepends=True)
    chunks: list[dict[str, object]] = []

    # Find table boundaries
    table_header: list[str] = []
    table_rows: list[str] = []
    non_table_buffer: list[str] = []
    in_table = False
    header_captured = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect table separator row
        if (
            not in_table
            and re.match(
                r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$",
                stripped,
            )
            and i > 0
            and "|" in lines[i - 1]
        ):
            # Flush non-table content
            if len(non_table_buffer) > 1:
                nt_text = "".join(non_table_buffer[:-1]).strip()
                if nt_text:
                    for piece in chunk_text(nt_text):
                        chunks.append({"text": piece, "is_table": False})

            table_header = [lines[i - 1], line]
            in_table = True
            header_captured = True
            non_table_buffer = []
            continue

        if in_table:
            if "|" in stripped:
                table_rows.append(line)
            else:
                # End of table, flush
                _flush_table_rows(
                    table_header, table_rows, chunks
                )
                table_header = []
                table_rows = []
                in_table = False
                header_captured = False
                non_table_buffer = [line]
        else:
            if not header_captured:
                non_table_buffer.append(line)

    # Flush remaining
    if in_table and table_rows:
        _flush_table_rows(table_header, table_rows, chunks)

    if non_table_buffer:
        nt_text = "".join(non_table_buffer).strip()
        if nt_text:
            for piece in chunk_text(nt_text):
                chunks.append({"text": piece, "is_table": False})

    return chunks


def _flush_table_rows(
    header: list[str],
    rows: list[str],
    chunks: list[dict[str, object]],
) -> None:
    """Split table rows into chunks, each prefixed with the header."""

    if not rows:
        return

    header_text = "".join(header)
    full_table = header_text + "".join(rows)

    if estimate_tokens(full_table) <= TARGET_CHUNK_TOKENS * 2:
        chunks.append({"text": full_table.strip(), "is_table": True})
        return

    # Split by rows, keeping header with each chunk
    current_rows: list[str] = []
    header_tokens = estimate_tokens(header_text)

    for row in rows:
        candidate = header_text + "".join(current_rows) + row
        if (
            estimate_tokens(candidate) > TARGET_CHUNK_TOKENS * 1.5
            and current_rows
        ):
            chunk_text_val = (
                header_text + "".join(current_rows)
            ).strip()
            chunks.append({"text": chunk_text_val, "is_table": True})
            current_rows = [row]
        else:
            current_rows.append(row)

    if current_rows:
        chunk_text_val = (
            header_text + "".join(current_rows)
        ).strip()
        chunks.append({"text": chunk_text_val, "is_table": True})


# --------------------------------------------------
# PROCESS ONE DOCUMENT
# --------------------------------------------------

def process_document(file_path: Path) -> list[dict[str, object]]:
    """Process a single V2 Markdown file into chunk records."""

    text = file_path.read_text(encoding="utf-8", errors="ignore")

    # Parse front matter
    meta, body = parse_front_matter(text)

    sections = split_into_sections(body)

    records: list[dict[str, object]] = []

    for section_data in sections:

        section_title = (
            section_data.get("section")
            or file_path.stem
        ).strip()

        section_text = section_data.get("text", "")

        if not section_text:
            continue

        # TABLE SECTION
        if contains_markdown_table(section_text):
            table_chunks = chunk_table_section(section_text)
            for tc in table_chunks:
                entry: dict[str, object] = {
                    "text": tc["text"],
                    "section": section_title,
                    "is_table": tc["is_table"],
                }
                # Carry routine metadata if present
                for rkey in ("program", "specialization", "semester"):
                    val = section_data.get(rkey, "")
                    if val:
                        entry[rkey] = val
                sec_val = section_data.get("section", "")
                if section_data.get("section") != section_title:
                    pass
                # Use section from routine metadata if available
                routine_section = section_data.get("section", "")
                if routine_section and routine_section != section_title:
                    entry["routine_section"] = routine_section

                records.append(entry)
            continue

        # NORMAL PROSE
        pieces = chunk_text(section_text)

        for piece in pieces:
            entry = {
                "text": piece,
                "section": section_title,
                "is_table": False,
            }
            for rkey in ("program", "specialization", "semester"):
                val = section_data.get(rkey, "")
                if val:
                    entry[rkey] = val

            records.append(entry)

    # Build final chunk records with all metadata
    chunk_records: list[dict[str, object]] = []

    for index, record in enumerate(records):

        quality = detect_quality_status(
            str(record["text"]),
            file_path.name,
        )

        chunk_record: dict[str, object] = {
            "chunk_id": f"v2_{file_path.stem}_{index}",
            "document": file_path.name,
            "source_url": meta.get("source_url", ""),
            "intent_category": meta.get("intent_category", "facilities"),
            "category": meta.get("intent_category", "facilities"),
            "topic": meta.get("topic", file_path.stem),
            "section": record["section"],
            "last_updated": meta.get("last_updated", "unknown"),
            "is_table": record["is_table"],
            "chunk_index": index,
            "text": record["text"],
            "source_type": meta.get("source_type", "official_pdf"),
            "verification_status": "trusted_official_source",
            "effective_date": meta.get("effective_date", "unknown"),
            "published_date": meta.get("published_date", "unknown"),
            "quality_status": quality,
        }

        # Add routine metadata if present
        for rkey in ("program", "specialization", "semester"):
            val = record.get(rkey, "")
            if val:
                chunk_record[rkey] = val

        token_count = estimate_tokens(str(record["text"]))
        chunk_record["estimated_tokens"] = token_count

        chunk_records.append(chunk_record)

    return chunk_records


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main() -> None:

    import sys
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return

    markdown_files = sorted(INPUT_DIR.glob("*.md"))

    if not markdown_files:
        print("ERROR: No Markdown files found in", INPUT_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    total_tables = 0
    total_manual_review = 0
    oversized: list[str] = []

    # Gemini embedding 2 token limit
    MAX_INPUT_TOKENS = 8192

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:

        for file_path in markdown_files:

            print(f"Chunking: {file_path.name}")

            records = process_document(file_path)

            for record in records:

                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                if record.get("is_table"):
                    total_tables += 1

                if record.get("quality_status") == "needs_manual_review":
                    total_manual_review += 1

                est_tokens = record.get("estimated_tokens", 0)
                if isinstance(est_tokens, (int, float)) and est_tokens > MAX_INPUT_TOKENS:
                    oversized.append(str(record.get("chunk_id", "?")))

            total_chunks += len(records)

            print(f"  -> {len(records)} chunks created")

    print()
    print("=" * 60)
    print("V2 CHUNKING COMPLETE")
    print("=" * 60)
    print(f"Documents processed   : {len(markdown_files)}")
    print(f"Total V2 chunks       : {total_chunks}")
    print(f"Table chunks          : {total_tables}")
    print(f"Needs manual review   : {total_manual_review}")
    print(f"Oversized (>{MAX_INPUT_TOKENS} tokens): {len(oversized)}")

    if oversized:
        print(f"  Oversized chunk IDs : {oversized}")

    print(f"Output file           : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
