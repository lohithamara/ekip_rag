from collections import Counter, defaultdict
from pathlib import Path

from ingestion.validation.raw_validator import (
    RawDocumentValidator,
)


PROCESSED_RAW_ROOT = Path("data/processed/raw")


def main():

    validator = RawDocumentValidator()

    json_files = list(
        PROCESSED_RAW_ROOT.rglob("*.json")
    )

    print(f"Raw JSON files discovered: {len(json_files)}")

    results = []

    for json_path in json_files:

        result = validator.validate_file(json_path)

        results.append(result)

    error_documents = 0
    warning_documents = 0
    clean_documents = 0

    issue_counts = Counter()

    duplicate_hashes = defaultdict(list)

    for result in results:

        if result.content_hash:

            duplicate_hashes[
                result.content_hash
            ].append(result.file_path)

        for issue in result.issues:
            issue_counts[issue.code] += 1

        if result.has_errors:
            error_documents += 1

        elif result.has_warnings:
            warning_documents += 1

        else:
            clean_documents += 1

    actual_duplicates = {
        content_hash: paths
        for content_hash, paths in duplicate_hashes.items()
        if len(paths) > 1
    }

    print("\n===================================")
    print("RAW CORPUS VALIDATION SUMMARY")
    print("===================================")

    print(f"Documents checked       : {len(results)}")
    print(f"Documents with errors   : {error_documents}")
    print(f"Documents with warnings : {warning_documents}")
    print(f"Documents without issues: {clean_documents}")

    print("\nISSUE COUNTS")
    print("-----------------------------------")

    if issue_counts:

        for code, count in sorted(
            issue_counts.items()
        ):
            print(f"{code:<30}: {count}")

    else:
        print("No validation issues detected.")

    print("\nDUPLICATE CONTENT HASHES")
    print("-----------------------------------")

    print(
        f"Duplicate hashes detected: "
        f"{len(actual_duplicates)}"
    )

    for content_hash, paths in actual_duplicates.items():

        print(f"\nHash: {content_hash}")

        for path in paths:
            print(f"  {path}")

    print("\nDOCUMENTS REQUIRING ATTENTION")
    print("-----------------------------------")

    attention_count = 0

    for result in results:

        if not result.issues:
            continue

        attention_count += 1

        print(f"\nFILE: {result.file_path}")

        for issue in result.issues:

            location = ""

            if issue.page_number is not None:
                location += (
                    f" page={issue.page_number}"
                )

            if issue.table_id is not None:
                location += (
                    f" table={issue.table_id}"
                )

            print(
                f"  [{issue.severity}] "
                f"{issue.code}{location}: "
                f"{issue.message}"
            )

    print()
    print(
        f"Documents requiring attention: "
        f"{attention_count}"
    )


if __name__ == "__main__":
    main()