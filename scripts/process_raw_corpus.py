from pathlib import Path

from ingestion.parsers.factory import (
    create_default_parser_registry,
)
from ingestion.serialization.document_serializer import (
    DocumentSerializer,
)
from ingestion.services.ingestion_processor import (
    IngestionProcessor,
)
from ingestion.storage.s3_client import S3Client


PROCESSED_RAW_ROOT = Path("data/processed/raw")

S3_ORIGINAL_PREFIX = "original/"


def build_output_path(
    tenant_id: str,
    department: str,
    content_hash: str,
    filename: str,
) -> Path:

    return (
        PROCESSED_RAW_ROOT
        / tenant_id
        / department
        / content_hash
        / f"{filename}.json"
    )


def parse_s3_key(
    s3_key: str,
) -> tuple[str, str, str, str]:

    parts = s3_key.split("/")

    if len(parts) != 5:

        raise ValueError(
            f"Unexpected S3 key structure: {s3_key}"
        )

    prefix = parts[0]
    tenant_id = parts[1]
    department = parts[2]
    content_hash = parts[3]
    filename = parts[4]

    if prefix != "original":

        raise ValueError(
            f"Expected original/ prefix: {s3_key}"
        )

    return (
        tenant_id,
        department,
        content_hash,
        filename,
    )


def main():

    s3 = S3Client()

    registry = create_default_parser_registry()

    processor = IngestionProcessor(
        s3_client=s3,
        parser_registry=registry,
    )

    objects = s3.list_objects(
        prefix=S3_ORIGINAL_PREFIX
    )

    processed = 0
    skipped = 0
    unsupported = 0
    failed = 0

    print(f"Objects discovered: {len(objects)}")

    for object_info in objects:

        s3_key = object_info["Key"]

        # S3 console folder-marker objects can exist.
        if s3_key.endswith("/"):
            continue

        print("\n===================================")
        print(f"Processing: {s3_key}")

        try:

            (
                tenant_id,
                department,
                content_hash,
                filename,
            ) = parse_s3_key(s3_key)

            output_path = build_output_path(
                tenant_id=tenant_id,
                department=department,
                content_hash=content_hash,
                filename=filename,
            )

            if output_path.exists():

                print("[SKIPPED] Raw JSON already exists")

                skipped += 1

                continue

            if not registry.supports(Path(filename)):

                print(
                    f"[UNSUPPORTED] "
                    f"No parser for: {filename}"
                )

                unsupported += 1

                continue

            raw_document = processor.process(
                s3_key=s3_key,
                tenant_id=tenant_id,
                department=department,
                content_hash=content_hash,
            )

            DocumentSerializer.save_raw_document(
                document=raw_document,
                output_path=output_path,
            )

            print(f"[PROCESSED] {output_path}")

            processed += 1

        except Exception as exc:

            print(f"[FAILED] {s3_key}")

            print(f"Reason: {exc}")

            failed += 1

    print("\n===================================")
    print("RAW INGESTION SUMMARY")
    print("===================================")

    print(f"Processed   : {processed}")
    print(f"Skipped     : {skipped}")
    print(f"Unsupported : {unsupported}")
    print(f"Failed      : {failed}")


if __name__ == "__main__":
    main()