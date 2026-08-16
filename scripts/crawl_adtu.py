import asyncio
import json
import re
from pathlib import Path

from crawl4ai import AsyncWebCrawler


ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "sources" / "sources.json"
OUT_DIR = ROOT / "data" / "processed" / "markdown"


def safe_filename(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    if not slug:
        slug = "page"

    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", slug)

    return f"{slug}.md"


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SOURCES_PATH, "r", encoding="utf-8") as f:
        sources = json.load(f)

    async with AsyncWebCrawler() as crawler:

        for source in sources:
            url = source["url"]

            print(f"\nCrawling: {url}")

            try:
                result = await crawler.arun(url=url)

                if not result.success:
                    print(f"FAILED: {url}")
                    print(result.error_message)
                    continue

                markdown = result.markdown

                output_file = OUT_DIR / safe_filename(url)

                content = f"""---
source_url: {url}
category: {source.get("category", "facilities")}
---

# Source

{url}

# Content

{markdown}
"""

                output_file.write_text(content, encoding="utf-8")

                print(f"SUCCESS: {output_file}")

            except Exception as e:
                print(f"ERROR: {url}")
                print(e)


if __name__ == "__main__":
    asyncio.run(main())