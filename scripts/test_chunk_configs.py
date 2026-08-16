from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "scripts"))

import chunk_data


INPUT_DIR = ROOT / "data" / "processed" / "markdown"

TARGET = 842


def count_chunks(
    chunk_size: int,
    chunk_overlap: int
) -> int:

    chunk_data.CHUNK_SIZE = chunk_size
    chunk_data.CHUNK_OVERLAP = chunk_overlap

    total = 0

    for file_path in sorted(
        INPUT_DIR.glob("*.md")
    ):

        total += len(
            chunk_data.process_document(
                file_path
            )
        )

    return total


def main():

    results = []

    print("=" * 70)
    print("FINE-GRAINED CHUNK CONFIGURATION SEARCH")
    print("=" * 70)

    print(
        "Searching for an exact 842-chunk configuration..."
    )
    print()

    for size in range(
        2300,
        2501,
        25
    ):

        for overlap in range(
            400,
            601,
            25
        ):

            if overlap >= size:
                continue

            total = count_chunks(
                size,
                overlap
            )

            difference = abs(
                total - TARGET
            )

            results.append(
                (
                    difference,
                    total,
                    size,
                    overlap
                )
            )

    results.sort()

    print(
        "=" * 70
    )

    print(
        "10 CLOSEST CONFIGURATIONS"
    )

    print(
        "=" * 70
    )

    for (
        difference,
        total,
        size,
        overlap
    ) in results[:10]:

        print(
            f"CHUNK_SIZE={size}, "
            f"CHUNK_OVERLAP={overlap} "
            f"→ {total} chunks "
            f"(difference={difference})"
        )

    exact = [
        result
        for result in results
        if result[0] == 0
    ]

    print()

    if exact:

        print(
            "=" * 70
        )

        print(
            "EXACT 842 CONFIGURATION(S) FOUND"
        )

        print(
            "=" * 70
        )

        for (
            _,
            total,
            size,
            overlap
        ) in exact:

            print(
                f"CHUNK_SIZE={size}, "
                f"CHUNK_OVERLAP={overlap} "
                f"→ {total} chunks"
            )

    else:

        print(
            "No exact 842 configuration found."
        )


if __name__ == "__main__":
    main()