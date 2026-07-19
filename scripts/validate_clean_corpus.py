from collections import Counter
from pathlib import Path

from ingestion.validation.clean_validator import (
    CleanDocumentValidator,
)


PROCESSED_RAW_ROOT = Path(
    "data/processed/raw"
)

PROCESSED_CLEAN_ROOT = Path(
    "data/processed/clean"
)


def main():

    validator = CleanDocumentValidator()

    raw_files = sorted(
        PROCESSED_RAW_ROOT.rglob("*.json")
    )

    print(
        f"Raw documents discovered: "
        f"{len(raw_files)}"
    )

    results = []

    missing_clean_files = []

    for raw_path in raw_files:

        relative_path = raw_path.relative_to(
            PROCESSED_RAW_ROOT
        )

        clean_path = (
            PROCESSED_CLEAN_ROOT
            / relative_path
        )

        if not clean_path.exists():

            missing_clean_files.append(
                clean_path
            )

            continue

        result = validator.validate_pair(
            raw_path=raw_path,
            clean_path=clean_path,
        )

        results.append(result)

    error_documents = 0
    warning_documents = 0
    clean_documents = 0

    issue_counts = Counter()

    retention_ratios = []

    for result in results:

        for issue in result.issues:
            issue_counts[issue.code] += 1

        if result.retention_ratio is not None:
            retention_ratios.append(
                (
                    result.retention_ratio,
                    result.clean_path,
                )
            )

        if result.has_errors:
            error_documents += 1

        elif result.has_warnings:
            warning_documents += 1

        else:
            clean_documents += 1

    print("\n===================================")
    print("CLEAN CORPUS VALIDATION SUMMARY")
    print("===================================")

    print(
        f"Document pairs checked    : "
        f"{len(results)}"
    )

    print(
        f"Missing clean documents   : "
        f"{len(missing_clean_files)}"
    )

    print(
        f"Documents with errors     : "
        f"{error_documents}"
    )

    print(
        f"Documents with warnings   : "
        f"{warning_documents}"
    )

    print(
        f"Documents without issues  : "
        f"{clean_documents}"
    )

    print("\nISSUE COUNTS")
    print("-----------------------------------")

    if issue_counts:

        for code, count in sorted(
            issue_counts.items()
        ):
            print(f"{code:<35}: {count}")

    else:
        print("No validation issues detected.")

    print("\nLOWEST TEXT RETENTION RATIOS")
    print("-----------------------------------")

    for ratio, path in sorted(
        retention_ratios,
        key=lambda item: item[0],
    )[:10]:

        print(
            f"{ratio:.4f}  {path}"
        )

    print("\nHIGHEST TEXT RETENTION RATIOS")
    print("-----------------------------------")

    for ratio, path in sorted(
        retention_ratios,
        key=lambda item: item[0],
        reverse=True,
    )[:10]:

        print(
            f"{ratio:.4f}  {path}"
        )

    print("\nDOCUMENTS REQUIRING ATTENTION")
    print("-----------------------------------")

    attention_count = 0

    for result in results:

        if not result.issues:
            continue

        attention_count += 1

        print(
            f"\nFILE: {result.clean_path}"
        )

        print(
            f"Retention ratio: "
            f"{result.retention_ratio}"
        )

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

    if missing_clean_files:

        print("\nMISSING CLEAN FILES")
        print("-----------------------------------")

        for path in missing_clean_files:
            print(path)

    print()
    print(
        f"Documents requiring attention: "
        f"{attention_count}"
    )


if __name__ == "__main__":
    main()