from pathlib import Path

from ingestion.storage.s3_client import S3Client


DATA_ROOT = Path("data")

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


def find_test_file() -> Path:
    for file_path in DATA_ROOT.rglob("*"):

        if (
            file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            return file_path

    raise FileNotFoundError(
        "No supported document found inside data/"
    )


def extract_path_metadata(
    file_path: Path,
) -> tuple[str, str]:

    relative_path = file_path.relative_to(DATA_ROOT)

    parts = relative_path.parts

    if len(parts) < 3:
        raise ValueError(
            f"Expected data/<tenant>/<department>/<file>, "
            f"got: {file_path}"
        )

    tenant_id = parts[0]
    department = parts[1]

    return tenant_id, department


def main():

    test_file = find_test_file()

    tenant_id, department = extract_path_metadata(
        test_file
    )

    s3 = S3Client()

    s3_key = (
        f"test/"
        f"{tenant_id}/"
        f"{department}/"
        f"{test_file.name}"
    )

    print("Selected test file:")
    print(test_file)

    print("\nDetected metadata:")
    print(f"Tenant     : {tenant_id}")
    print(f"Department : {department}")

    print("\nUploading...")

    s3.upload_file(
        local_path=str(test_file),
        s3_key=s3_key,
    )

    print("\nUpload successful")
    print(f"Bucket : {s3.bucket_name}")
    print(f"S3 Key : {s3_key}")


if __name__ == "__main__":
    main()