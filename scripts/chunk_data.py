"""
Chunk AdtU Markdown documents for RAG.

Reads:
    data/processed/markdown/*.md

Creates:
    data/processed/chunks/chunks.jsonl

Required metadata for every chunk:
    source_url
    category
    section
    last_updated
    is_table
"""

from __future__ import annotations

import json
import re
from pathlib import Path


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "processed" / "markdown"

OUTPUT_DIR = ROOT / "data" / "processed" / "chunks"

OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"


# --------------------------------------------------
# CHUNK SETTINGS
# --------------------------------------------------

# Approximate target size.
# We use characters here because we are not
# calling a tokenizer during preprocessing.

CHUNK_SIZE = 2325

# Approximate 50-token overlap.
CHUNK_OVERLAP = 400


# --------------------------------------------------
# CATEGORY MAPPING
# --------------------------------------------------

def determine_category(
    filename: str,
    existing_category: str = ""
) -> str:

    """
    Determine the required category.

    Allowed categories:
        admissions
        fees
        facilities
    """

    existing = existing_category.lower().strip()

    if existing in {
        "admissions",
        "fees",
        "facilities"
    }:
        return existing

    name = filename.lower()

    # Admissions-related documents
    if any(
        keyword in name
        for keyword in [
            "admission",
            "apply",
            "join",
            "ews",
            "programme",
            "program"
        ]
    ):
        return "admissions"

    # Fee-related documents
    if any(
        keyword in name
        for keyword in [
            "fee",
            "fees",
            "tuition",
            "hostel-fee"
        ]
    ):
        return "fees"

    # Default university facilities/resources
    return "facilities"


# --------------------------------------------------
# METADATA EXTRACTION
# --------------------------------------------------

def extract_metadata(
    text: str,
    filename: str
) -> dict:

    """
    Extract source URL, category and last_updated
    from Markdown/front matter.
    """

    source_url = ""

    category = ""

    last_updated = ""

    # ----------------------------------------------
    # Source URL
    # ----------------------------------------------

    match = re.search(
        r"source_url:\s*(.+)",
        text,
        flags=re.IGNORECASE
    )

    if match:
        source_url = match.group(1).strip()

    # Fallback source URL
    if not source_url:

        match = re.search(
            r"^Source:\s*(https?://\S+)",
            text,
            flags=re.IGNORECASE | re.MULTILINE
        )

        if match:
            source_url = match.group(1).strip()

    # ----------------------------------------------
    # Category
    # ----------------------------------------------

    match = re.search(
        r"category:\s*(.+)",
        text,
        flags=re.IGNORECASE
    )

    if match:
        category = match.group(1).strip()

    category = determine_category(
        filename,
        category
    )

    # ----------------------------------------------
    # Last updated
    # ----------------------------------------------

    possible_patterns = [

        r"last_updated:\s*(.+)",

        r"last updated:\s*(.+)",

        r"updated on:\s*(.+)",

        r"updated:\s*(.+)",

    ]

    for pattern in possible_patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            last_updated = (
                match.group(1)
                .strip()
            )

            break

    # If the official page doesn't provide a
    # last-updated date, keep it explicitly unknown.

    if not last_updated:

        last_updated = "unknown"

    return {

        "source_url": source_url,

        "category": category,

        "last_updated": last_updated,

    }


# --------------------------------------------------
# TABLE DETECTION
# --------------------------------------------------

def extract_tables(text: str) -> list[tuple[int, int]]:

    """
    Find Markdown pipe-table ranges.

    Returns:
        [(start_position, end_position), ...]
    """

    lines = text.splitlines(
        keepends=True
    )

    tables = []

    position = 0

    i = 0

    while i < len(lines) - 1:

        current = lines[i].strip()

        next_line = lines[i + 1].strip()

        # Markdown table normally has:
        #
        # | Header | Header |
        # |--------|--------|

        if (
            "|" in current
            and re.match(
                r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$",
                next_line
            )
        ):

            start = position

            i += 2

            while i < len(lines):

                line = lines[i].strip()

                if "|" not in line:

                    break

                i += 1

            end = sum(
                len(line)
                for line in lines[:i]
            )

            tables.append(
                (start, end)
            )

            position = end

            continue

        position += len(
            lines[i]
        )

        i += 1

    return tables


# --------------------------------------------------
# SECTION SPLITTING
# --------------------------------------------------

def split_into_sections(
    text: str
) -> list[dict]:

    """
    Split Markdown primarily using H1/H2 headers.
    """

    lines = text.splitlines(
        keepends=True
    )

    sections = []

    current_title = None

    current_lines = []

    for line in lines:

        header_match = re.match(
            r"^(#{1,2})\s+(.+?)\s*$",
            line
        )

        if header_match:

            if current_lines:

                sections.append({

                    "section": current_title,

                    "text": "".join(
                        current_lines
                    ).strip()

                })

            current_title = (
                header_match.group(2)
                .strip()
            )

            current_lines = [
                line
            ]

        else:

            current_lines.append(
                line
            )

    if current_lines:

        sections.append({

            "section": current_title,

            "text": "".join(
                current_lines
            ).strip()

        })

    return sections


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text: str) -> str:

    # Normalize line endings
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# --------------------------------------------------
# TABLE CHECK
# --------------------------------------------------

def contains_markdown_table(
    text: str
) -> bool:

    lines = text.splitlines()

    for i in range(
        len(lines) - 1
    ):

        if (
            "|" in lines[i]
            and re.match(
                r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$",
                lines[i + 1].strip()
            )
        ):

            return True

    return False


# --------------------------------------------------
# CHUNK NON-TABLE TEXT
# --------------------------------------------------

def chunk_text(
    text: str
) -> list[str]:

    text = clean_text(text)

    if not text:

        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(
            start + CHUNK_SIZE,
            text_length
        )

        # Try to end at a paragraph boundary.
        if end < text_length:

            boundary = text.rfind(
                "\n\n",
                start,
                end
            )

            if boundary > (
                start
                + CHUNK_SIZE // 2
            ):

                end = boundary

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        if end >= text_length:

            break

        next_start = (
            end
            - CHUNK_OVERLAP
        )

        if next_start <= start:

            next_start = end

        start = next_start

    return chunks


# --------------------------------------------------
# PROCESS ONE DOCUMENT
# --------------------------------------------------

def process_document(
    file_path: Path
) -> list[dict]:

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    metadata = extract_metadata(
        text,
        file_path.name
    )

    sections = split_into_sections(
        text
    )

    records = []

    for section_data in sections:

        section_title = (
        section_data.get("section")
        or file_path.stem
        ).strip()

        section_text = section_data["text"]

        if not section_text:

            continue

        # --------------------------------------------------
        # TABLE SECTION
        # --------------------------------------------------

        if contains_markdown_table(
            section_text
        ):

            # Keep the complete section/table
            # together.

            records.append({

                "text": section_text,

                "section": section_title,

                "is_table": True

            })

            continue

        # --------------------------------------------------
        # NORMAL PROSE
        # --------------------------------------------------

        pieces = chunk_text(
            section_text
        )

        for piece in pieces:

            records.append({

                "text": piece,

                "section": section_title,

                "is_table": False

            })

    return [
        {

            "chunk_id": (
                f"{file_path.stem}_{index}"
            ),

            "document": file_path.name,

            "source_url": metadata[
                "source_url"
            ],

            "category": metadata[
                "category"
            ],

            "section": record[
                "section"
            ],

            "last_updated": metadata[
                "last_updated"
            ],

            "is_table": record[
                "is_table"
            ],

            "chunk_index": index,

            "text": record[
                "text"
            ]

        }

        for index, record
        in enumerate(records)

    ]


# --------------------------------------------------
# MAIN PROCESS
# --------------------------------------------------

def process_documents():

    if not INPUT_DIR.exists():

        print(
            f"ERROR: Input directory not found:"
            f" {INPUT_DIR}"
        )

        return

    markdown_files = sorted(
        INPUT_DIR.glob("*.md")
    )

    if not markdown_files:

        print(
            "ERROR: No Markdown files found."
        )

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_chunks = 0

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as output:

        for file_path in markdown_files:

            print(
                f"Processing: "
                f"{file_path.name}"
            )

            records = process_document(
                file_path
            )

            for record in records:

                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

            total_chunks += len(
                records
            )

            print(
                f"  → {len(records)} chunks created"
            )

    print()

    print(
        "=" * 60
    )

    print(
        "CHUNKING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Documents processed : "
        f"{len(markdown_files)}"
    )

    print(
        f"Total chunks        : "
        f"{total_chunks}"
    )

    print(
        f"Output file         : "
        f"{OUTPUT_FILE}"
    )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    process_documents()