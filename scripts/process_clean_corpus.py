from pathlib import Path

from ingestion.cleaning.factory import (
    create_default_cleaning_pipeline,
)
from ingestion.serialization.document_serializer import (
    DocumentSerializer,
)
from ingestion.services.document_cleaning_service import (
    DocumentCleaningService,
)


PROCESSED_RAW_ROOT = Path(
    "data/processed/raw"
)

PROCESSED_CLEAN_ROOT = Path(
    "data/processed/clean"
)


def build_output_path(
    raw_path: Path,
) -> Path:

    relative_path = raw_path.relative_to(
        PROCESSED_RAW_ROOT
    )

    return (
        PROCESSED_CLEAN_ROOT
        / relative_path
    )


def main():

    pipeline = (
        create_default_cleaning_pipeline()
    )

    cleaning_service = (
        DocumentCleaningService(
            cleaning_pipeline=pipeline,
        )
    )

    raw_files = list(
        PROCESSED_RAW_ROOT.rglob("*.json")
    )

    cleaned = 0
    skipped = 0
    failed = 0

    print(
        f"Raw documents discovered: "
        f"{len(raw_files)}"
    )

    for raw_path in raw_files:

        output_path = build_output_path(
            raw_path
        )

        print("\n===================================")
        print(f"Cleaning: {raw_path}")

        try:

            if output_path.exists():

                print(
                    "[SKIPPED] Clean JSON already exists"
                )

                skipped += 1

                continue

            raw_document = (
                DocumentSerializer
                .load_raw_document(raw_path)
            )

            clean_document = (
                cleaning_service.clean_document(
                    raw_document
                )
            )

            DocumentSerializer.save_clean_document(
                document=clean_document,
                output_path=output_path,
            )

            print(
                f"[CLEANED] {output_path}"
            )

            cleaned += 1

        except Exception as exc:

            print(f"[FAILED] {raw_path}")
            print(f"Reason: {exc}")

            failed += 1

    print("\n===================================")
    print("CLEANING SUMMARY")
    print("===================================")

    print(f"Cleaned : {cleaned}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")


if __name__ == "__main__":
    main()