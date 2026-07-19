import json
from pathlib import Path


PROCESSED_RAW_ROOT = Path("data/processed/raw")


def main():

    failed_documents = []

    for json_path in PROCESSED_RAW_ROOT.rglob("*.json"):

        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            document = json.load(file)

        failed_pages = []

        for page in document.get("pages", []):

            page_metadata = page.get("metadata", {})

            warnings = page_metadata.get("warnings", [])

            if any(
                "OCR failed" in warning
                or "OCR returned no text" in warning
                for warning in warnings
            ):
                failed_pages.append(
                    page.get("page_number")
                )

        if failed_pages:
            failed_documents.append(
                (json_path, failed_pages)
            )

    print("FAILED OCR DOCUMENTS")
    print("====================")

    for json_path, pages in failed_documents:
        print(f"{json_path}")
        print(f"Pages: {pages}")

    print()
    print(
        f"Documents requiring OCR reprocessing: "
        f"{len(failed_documents)}"
    )


if __name__ == "__main__":
    main()