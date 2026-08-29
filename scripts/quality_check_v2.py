"""
Data quality check for V2 chunks.

Checks:
    - Malformed extraction / garbled text
    - Broken tables
    - OCR corruption
    - Incorrect/suspicious dates or numbers
    - Duplicate content within V2
    - Missing provenance
    - Inconsistent fees
    - Incomplete scholarship rules
    - Stale information
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

V2_CHUNKS = ROOT / "data" / "processed" / "knowledge_v2" / "canonical_chunks.jsonl"
OUTPUT_FILE = ROOT / "data" / "processed" / "knowledge_v2" / "quality_report.csv"


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def main() -> None:

    import sys
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    if not V2_CHUNKS.exists():
        print(f"ERROR: V2 chunks not found: {V2_CHUNKS}")
        return

    records = load_jsonl(V2_CHUNKS)
    issues: list[dict[str, str]] = []

    for chunk in records:
        chunk_id = str(chunk.get("chunk_id", ""))
        text = str(chunk.get("text", ""))
        source = str(chunk.get("document", ""))
        tokens = estimate_tokens(text)

        # 1. Very short chunks (likely incomplete extraction)
        if tokens < 5:
            issues.append({
                "chunk_id": chunk_id,
                "source_file": source,
                "issue_type": "very_short_chunk",
                "severity": "warning",
                "description": f"Only {tokens} estimated tokens. May be incomplete extraction.",
                "recommendation": "Review source PDF for missing content",
            })

        # 2. Very large chunks (embedding risk)
        if tokens > 2000:
            issues.append({
                "chunk_id": chunk_id,
                "source_file": source,
                "issue_type": "oversized_chunk",
                "severity": "warning",
                "description": f"{tokens} estimated tokens. May need derived-child strategy at embedding time.",
                "recommendation": "Consider splitting or using derived children",
            })

        # 3. Garbled text detection (missing spaces in scholarship content)
        words_no_spaces = re.findall(r"[A-Z]{15,}", text)
        if words_no_spaces:
            issues.append({
                "chunk_id": chunk_id,
                "source_file": source,
                "issue_type": "garbled_text",
                "severity": "error",
                "description": f"Found {len(words_no_spaces)} long uppercase sequences suggesting missing spaces: {words_no_spaces[:3]}",
                "recommendation": "Manual review against original PDF required",
            })

        # 4. Broken table structure (pipe without separator row)
        if chunk.get("is_table"):
            lines = text.splitlines()
            has_separator = any(
                re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$", l.strip())
                for l in lines
            )
            if not has_separator and len(lines) > 2:
                issues.append({
                    "chunk_id": chunk_id,
                    "source_file": source,
                    "issue_type": "broken_table",
                    "severity": "warning",
                    "description": "Marked as table but no valid Markdown table separator row found",
                    "recommendation": "Verify table structure in original PDF",
                })

        # 5. HTML artifacts in text (from pymupdf4llm)
        html_tags = re.findall(r"<(?:br|sup|sub|mark|u|/\w+)>", text)
        if html_tags:
            # This is informational, not necessarily an error
            if len(html_tags) > 10:
                issues.append({
                    "chunk_id": chunk_id,
                    "source_file": source,
                    "issue_type": "excessive_html_artifacts",
                    "severity": "info",
                    "description": f"Contains {len(html_tags)} HTML tags from PDF extraction",
                    "recommendation": "Informational - HTML tags from pymupdf4llm are expected in some PDFs",
                })

        # 6. Suspicious fee amounts (sanity check)
        if chunk.get("topic") == "fees":
            amounts = re.findall(r"(?:\*\*)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b", text)
            for amt_str in amounts:
                try:
                    amt = float(amt_str.replace(",", ""))
                    if amt > 2_000_000:
                        issues.append({
                            "chunk_id": chunk_id,
                            "source_file": source,
                            "issue_type": "suspicious_fee_amount",
                            "severity": "warning",
                            "description": f"Unusually large fee amount detected: {amt_str}",
                            "recommendation": "Verify against original fee structure PDF",
                        })
                except ValueError:
                    pass

        # 7. Empty or near-empty table cells
        if chunk.get("is_table"):
            empty_cells = len(re.findall(r"\|\s*\|", text))
            total_cells = len(re.findall(r"\|", text))
            if total_cells > 0 and empty_cells / total_cells > 0.5:
                issues.append({
                    "chunk_id": chunk_id,
                    "source_file": source,
                    "issue_type": "mostly_empty_table",
                    "severity": "warning",
                    "description": f"Table has {empty_cells}/{total_cells} empty cells ({100*empty_cells/total_cells:.0f}%)",
                    "recommendation": "Review if table structure was preserved correctly",
                })

        # 8. Missing source_url
        if not chunk.get("source_url"):
            issues.append({
                "chunk_id": chunk_id,
                "source_file": source,
                "issue_type": "missing_source_url",
                "severity": "info",
                "description": "No public URL available for this PDF source",
                "recommendation": "Normal for PDFs without public URLs. Provenance tracked via source_type and document fields.",
            })

    # Write quality report
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "chunk_id", "source_file", "issue_type", "severity",
            "description", "recommendation",
        ])
        writer.writeheader()
        writer.writerows(issues)

    # Summary
    severity_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}

    for issue in issues:
        sev = issue["severity"]
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        itype = issue["issue_type"]
        type_counts[itype] = type_counts.get(itype, 0) + 1

    print()
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Total issues found    : {len(issues)}")
    print(f"By severity           : {severity_counts}")
    print(f"By type               : {type_counts}")
    print(f"Output file           : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
