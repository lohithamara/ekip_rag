from collections import Counter
from pathlib import Path

from ingestion.serialization.document_serializer import (
    DocumentSerializer,
)

from rag.chunking.config import ChunkingConfig
from rag.chunking.routing.profiler import (
    DocumentProfiler,
)
from rag.chunking.tokenization.factory import (
    create_token_counter,
)


PROCESSED_CLEAN_ROOT = Path(
    "data/processed/clean"
)


def main():

    clean_files = sorted(
        PROCESSED_CLEAN_ROOT.rglob("*.json")
    )

    if not clean_files:

        raise RuntimeError(
            "No clean documents found."
        )

    config = ChunkingConfig(
        strategy="recursive",
        chunk_size=512,
        chunk_overlap=64,
        min_chunk_size=50,
        length_unit="tokens",
    )

    token_counter = create_token_counter(
        config
    )

    profiler = DocumentProfiler(
        token_counter=token_counter,
        short_document_threshold=512,
    )

    file_types = Counter()

    profiles = []

    for clean_path in clean_files:

        document = (
            DocumentSerializer.load_clean_document(
                clean_path
            )
        )

        profile = profiler.profile(document)

        profiles.append(
            (
                clean_path,
                profile,
            )
        )

        file_types[
            profile.file_type
        ] += 1

    print()
    print("===================================")
    print("DOCUMENT CORPUS PROFILE")
    print("===================================")

    print(f"Documents       : {len(profiles)}")

    print()
    print("FILE TYPES")
    print("-----------------------------------")

    for file_type, count in sorted(
        file_types.items()
    ):

        print(
            f"{file_type:<15}: {count}"
        )

    print()
    print("DOCUMENT CHARACTERISTICS")
    print("-----------------------------------")

    print(
        "Short documents      : "
        f"{sum(
            profile.is_short_document
            for _, profile in profiles
        )}"
    )

    print(
        "Documents with pages : "
        f"{sum(
            profile.has_pages
            for _, profile in profiles
        )}"
    )

    print(
        "Documents with tables: "
        f"{sum(
            profile.has_tables
            for _, profile in profiles
        )}"
    )

    print(
        "Structured documents : "
        f"{sum(
            profile.has_structure
            for _, profile in profiles
        )}"
    )

    print()
    print("STRUCTURE CONFIDENCE DISTRIBUTION")
    print("-----------------------------------")

    buckets = Counter()

    for _, profile in profiles:

        score = profile.structure_confidence

        if score < 0.25:
            bucket = "0.00 - 0.24"

        elif score < 0.50:
            bucket = "0.25 - 0.49"

        elif score < 0.75:
            bucket = "0.50 - 0.74"

        else:
            bucket = "0.75 - 1.00"

        buckets[bucket] += 1

    for bucket in (
        "0.00 - 0.24",
        "0.25 - 0.49",
        "0.50 - 0.74",
        "0.75 - 1.00",
    ):

        print(
            f"{bucket:<15}: "
            f"{buckets[bucket]}"
        )

    print()
    print("PER-DOCUMENT PROFILE")
    print("-----------------------------------")

    for clean_path, profile in profiles:

        print()

        print(f"File       : {clean_path}")
        print(f"Type       : {profile.file_type}")
        print(f"Tokens     : {profile.token_count}")
        print(f"Pages      : {profile.pages_with_text}")
        print(f"Tables     : {profile.tables_with_rows}")
        print(f"Headings   : {profile.heading_count}")
        print(f"Paragraphs : {profile.paragraph_count}")
        print(
            f"Structure  : "
            f"{profile.structure_confidence:.2f}"
        )


if __name__ == "__main__":
    main()