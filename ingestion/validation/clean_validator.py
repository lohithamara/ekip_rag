import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CleanValidationIssue:
    severity: str
    code: str
    message: str

    page_number: int | None = None
    table_id: str | None = None


@dataclass
class CleanDocumentValidationResult:
    raw_path: Path
    clean_path: Path

    document_id: str | None
    tenant_id: str | None
    department: str | None
    content_hash: str | None

    raw_text_length: int = 0
    clean_text_length: int = 0
    retention_ratio: float | None = None

    issues: list[CleanValidationIssue] = field(
        default_factory=list
    )

    @property
    def has_errors(self) -> bool:
        return any(
            issue.severity == "ERROR"
            for issue in self.issues
        )

    @property
    def has_warnings(self) -> bool:
        return any(
            issue.severity == "WARNING"
            for issue in self.issues
        )


class CleanDocumentValidator:

    MIN_RETENTION_RATIO = 0.50

    MAX_EXPANSION_RATIO = 1.20

    REQUIRED_METADATA_FIELDS = {
        "document_id",
        "tenant_id",
        "department",
        "filename",
        "file_type",
        "original_s3_key",
        "content_hash",
    }

    def validate_pair(
        self,
        raw_path: Path,
        clean_path: Path,
    ) -> CleanDocumentValidationResult:

        try:
            raw_document = self._load_json(
                raw_path
            )

            clean_document = self._load_json(
                clean_path
            )

        except Exception as exc:

            return CleanDocumentValidationResult(
                raw_path=raw_path,
                clean_path=clean_path,
                document_id=None,
                tenant_id=None,
                department=None,
                content_hash=None,
                issues=[
                    CleanValidationIssue(
                        severity="ERROR",
                        code="INVALID_JSON",
                        message=str(exc),
                    )
                ],
            )

        raw_metadata = raw_document.get(
            "metadata",
            {},
        )

        clean_metadata = clean_document.get(
            "metadata",
            {},
        )

        result = CleanDocumentValidationResult(
            raw_path=raw_path,
            clean_path=clean_path,
            document_id=clean_metadata.get(
                "document_id"
            ),
            tenant_id=clean_metadata.get(
                "tenant_id"
            ),
            department=clean_metadata.get(
                "department"
            ),
            content_hash=clean_metadata.get(
                "content_hash"
            ),
        )

        self._validate_required_metadata(
            clean_metadata,
            result,
        )

        self._validate_document_identity(
            raw_metadata,
            clean_metadata,
            result,
        )

        self._validate_clean_text(
            raw_document,
            clean_document,
            result,
        )

        self._validate_pages(
            raw_document,
            clean_document,
            result,
        )

        self._validate_tables(
            raw_document,
            clean_document,
            result,
        )

        self._validate_cleaning_steps(
            clean_document,
            result,
        )

        return result

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict[str, Any]:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                f"JSON root must be an object: {path}"
            )

        return data

    def _validate_required_metadata(
        self,
        metadata: dict,
        result: CleanDocumentValidationResult,
    ) -> None:

        for field_name in self.REQUIRED_METADATA_FIELDS:

            value = metadata.get(field_name)

            if value is None or value == "":

                result.issues.append(
                    CleanValidationIssue(
                        severity="ERROR",
                        code="MISSING_METADATA",
                        message=(
                            f"Missing required metadata: "
                            f"{field_name}"
                        ),
                    )
                )

    @staticmethod
    def _validate_document_identity(
        raw_metadata: dict,
        clean_metadata: dict,
        result: CleanDocumentValidationResult,
    ) -> None:

        identity_fields = [
            "document_id",
            "tenant_id",
            "department",
            "filename",
            "file_type",
            "original_s3_key",
            "content_hash",
        ]

        for field_name in identity_fields:

            raw_value = raw_metadata.get(
                field_name
            )

            clean_value = clean_metadata.get(
                field_name
            )

            if raw_value != clean_value:

                result.issues.append(
                    CleanValidationIssue(
                        severity="ERROR",
                        code="DOCUMENT_IDENTITY_MISMATCH",
                        message=(
                            f"{field_name}: "
                            f"raw={raw_value}, "
                            f"clean={clean_value}"
                        ),
                    )
                )

    def _validate_clean_text(
        self,
        raw_document: dict,
        clean_document: dict,
        result: CleanDocumentValidationResult,
    ) -> None:

        raw_text = raw_document.get(
            "raw_text"
        )

        clean_text = clean_document.get(
            "clean_text"
        )

        if not isinstance(raw_text, str):

            result.issues.append(
                CleanValidationIssue(
                    severity="ERROR",
                    code="INVALID_RAW_TEXT",
                    message="raw_text must be a string.",
                )
            )

            return

        if not isinstance(clean_text, str):

            result.issues.append(
                CleanValidationIssue(
                    severity="ERROR",
                    code="INVALID_CLEAN_TEXT",
                    message="clean_text must be a string.",
                )
            )

            return

        raw_meaningful = self._meaningful_length(
            raw_text
        )

        clean_meaningful = self._meaningful_length(
            clean_text
        )

        result.raw_text_length = raw_meaningful

        result.clean_text_length = clean_meaningful

        if raw_meaningful == 0:

            result.retention_ratio = None

            return

        result.retention_ratio = (
            clean_meaningful / raw_meaningful
        )

        if clean_meaningful == 0:

            result.issues.append(
                CleanValidationIssue(
                    severity="ERROR",
                    code="EMPTY_CLEAN_TEXT",
                    message=(
                        "Cleaning removed all meaningful "
                        "document text."
                    ),
                )
            )

        elif (
            result.retention_ratio
            < self.MIN_RETENTION_RATIO
        ):

            result.issues.append(
                CleanValidationIssue(
                    severity="WARNING",
                    code="LOW_TEXT_RETENTION",
                    message=(
                        f"Text retention ratio is "
                        f"{result.retention_ratio:.3f}"
                    ),
                )
            )

        if (
            result.retention_ratio
            > self.MAX_EXPANSION_RATIO
        ):

            result.issues.append(
                CleanValidationIssue(
                    severity="WARNING",
                    code="UNEXPECTED_TEXT_EXPANSION",
                    message=(
                        f"Clean text expansion ratio is "
                        f"{result.retention_ratio:.3f}"
                    ),
                )
            )

    @staticmethod
    def _validate_pages(
        raw_document: dict,
        clean_document: dict,
        result: CleanDocumentValidationResult,
    ) -> None:

        raw_pages = raw_document.get(
            "pages",
            [],
        )

        clean_pages = clean_document.get(
            "pages",
            [],
        )

        if not isinstance(clean_pages, list):

            result.issues.append(
                CleanValidationIssue(
                    severity="ERROR",
                    code="INVALID_CLEAN_PAGES",
                    message="Clean pages must be a list.",
                )
            )

            return

        if len(raw_pages) != len(clean_pages):

            result.issues.append(
                CleanValidationIssue(
                    severity="ERROR",
                    code="PAGE_COUNT_MISMATCH",
                    message=(
                        f"raw={len(raw_pages)}, "
                        f"clean={len(clean_pages)}"
                    ),
                )
            )

            return

        for raw_page, clean_page in zip(
            raw_pages,
            clean_pages,
        ):

            raw_page_number = raw_page.get(
                "page_number"
            )

            clean_page_number = clean_page.get(
                "page_number"
            )

            if raw_page_number != clean_page_number:

                result.issues.append(
                    CleanValidationIssue(
                        severity="ERROR",
                        code="PAGE_NUMBER_MISMATCH",
                        message=(
                            f"raw={raw_page_number}, "
                            f"clean={clean_page_number}"
                        ),
                        page_number=clean_page_number,
                    )
                )

            if (
                raw_page.get("ocr_used")
                != clean_page.get("ocr_used")
            ):

                result.issues.append(
                    CleanValidationIssue(
                        severity="ERROR",
                        code="PAGE_PROVENANCE_MISMATCH",
                        message=(
                            "ocr_used changed during cleaning."
                        ),
                        page_number=clean_page_number,
                    )
                )

    @staticmethod
    def _validate_tables(
        raw_document: dict,
        clean_document: dict,
        result: CleanDocumentValidationResult,
    ) -> None:

        raw_tables = raw_document.get(
            "tables",
            [],
        )

        clean_tables = clean_document.get(
            "tables",
            [],
        )

        if not isinstance(clean_tables, list):

            result.issues.append(
                CleanValidationIssue(
                    severity="ERROR",
                    code="INVALID_CLEAN_TABLES",
                    message="Clean tables must be a list.",
                )
            )

            return

        if len(raw_tables) != len(clean_tables):

            result.issues.append(
                CleanValidationIssue(
                    severity="ERROR",
                    code="TABLE_COUNT_MISMATCH",
                    message=(
                        f"raw={len(raw_tables)}, "
                        f"clean={len(clean_tables)}"
                    ),
                )
            )

            return

        for raw_table, clean_table in zip(
            raw_tables,
            clean_tables,
        ):

            raw_table_id = raw_table.get(
                "table_id"
            )

            clean_table_id = clean_table.get(
                "table_id"
            )

            if raw_table_id != clean_table_id:

                result.issues.append(
                    CleanValidationIssue(
                        severity="ERROR",
                        code="TABLE_ID_MISMATCH",
                        message=(
                            f"raw={raw_table_id}, "
                            f"clean={clean_table_id}"
                        ),
                        table_id=clean_table_id,
                    )
                )

            if (
                raw_table.get("page_number")
                != clean_table.get("page_number")
            ):

                result.issues.append(
                    CleanValidationIssue(
                        severity="ERROR",
                        code="TABLE_PROVENANCE_MISMATCH",
                        message=(
                            "Table page_number changed "
                            "during cleaning."
                        ),
                        table_id=clean_table_id,
                    )
                )

            raw_rows = raw_table.get(
                "rows",
                [],
            )

            clean_rows = clean_table.get(
                "rows",
                [],
            )

            if len(raw_rows) != len(clean_rows):

                result.issues.append(
                    CleanValidationIssue(
                        severity="ERROR",
                        code="TABLE_ROW_COUNT_MISMATCH",
                        message=(
                            f"raw={len(raw_rows)}, "
                            f"clean={len(clean_rows)}"
                        ),
                        table_id=clean_table_id,
                    )
                )

    @staticmethod
    def _validate_cleaning_steps(
        clean_document: dict,
        result: CleanDocumentValidationResult,
    ) -> None:

        cleaning_steps = clean_document.get(
            "cleaning_steps"
        )

        if not isinstance(cleaning_steps, list):

            result.issues.append(
                CleanValidationIssue(
                    severity="ERROR",
                    code="INVALID_CLEANING_STEPS",
                    message=(
                        "cleaning_steps must be a list."
                    ),
                )
            )

            return

        if not cleaning_steps:

            result.issues.append(
                CleanValidationIssue(
                    severity="WARNING",
                    code="NO_CLEANING_STEPS_RECORDED",
                    message=(
                        "No cleaning steps were recorded."
                    ),
                )
            )

    @staticmethod
    def _meaningful_length(
        text: str,
    ) -> int:

        return sum(
            character.isalnum()
            for character in text
        )