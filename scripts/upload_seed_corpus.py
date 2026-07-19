import hashlib
import mimetypes
from pathlib import Path

from ingestion.storage.s3_client import S3Client


DATA_ROOT = Path("data/raw_corpus")

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".html",
    ".htm",
    ".csv",
    ".xlsx",
    ".md",
    ".json",
}


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def discover_documents():
    for file_path in DATA_ROOT.rglob("*"):

        if file_path.is_file():
            yield file_path


def extract_path_metadata(
    file_path: Path,
) -> tuple[str, str]:

    relative_path = file_path.relative_to(DATA_ROOT)

    parts = relative_path.parts

    if len(parts) < 3:
        raise ValueError(
            f"Expected path: "
            f"data/raw_corpus/<tenant>/<department>/<file>, "
            f"got: {file_path}"
        )

    tenant_id = parts[0]
    department = parts[1]

    return tenant_id, department


def build_s3_key(
    file_path: Path,
    tenant_id: str,
    department: str,
    content_hash: str,
) -> str:

    return (
        f"original/"
        f"{tenant_id}/"
        f"{department}/"
        f"{content_hash}/"
        f"{file_path.name}"
    )


def main():

    s3 = S3Client()

    uploaded = 0
    skipped = 0
    unsupported = 0
    failed = 0

    for file_path in discover_documents():

        extension = file_path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:

            print(f"[UNSUPPORTED] {file_path}")

            unsupported += 1

            continue

        try:

            tenant_id, department = (
                extract_path_metadata(file_path)
            )

            content_hash = calculate_sha256(file_path)

            content_type, _ = mimetypes.guess_type(
                file_path.name
            )

            s3_key = build_s3_key(
                file_path=file_path,
                tenant_id=tenant_id,
                department=department,
                content_hash=content_hash,
            )

            print("\n-----------------------------------")

            print(f"File       : {file_path}")
            print(f"Tenant     : {tenant_id}")
            print(f"Department : {department}")
            print(f"SHA-256    : {content_hash}")
            print(f"S3 Key     : {s3_key}")

            if s3.file_exists(s3_key):

                print("[SKIPPED] Already exists")

                skipped += 1

                continue
            print(f"FINAL S3 KEY: {s3_key}")
            s3.upload_file(
                local_path=str(file_path),
                s3_key=s3_key,
                metadata={
                    "tenant_id": tenant_id,
                    "department": department,
                    "content_hash": content_hash,
                    "original_filename": file_path.name,
                },
                content_type=content_type,
            )

            print("[UPLOADED]")

            uploaded += 1

        except Exception as exc:

            print(f"[FAILED] {file_path}")

            print(f"Reason: {exc}")

            failed += 1

    print("\n===================================")
    print("UPLOAD SUMMARY")
    print("===================================")

    print(f"Uploaded    : {uploaded}")
    print(f"Skipped     : {skipped}")
    print(f"Unsupported : {unsupported}")
    print(f"Failed      : {failed}")


if __name__ == "__main__":
    main()