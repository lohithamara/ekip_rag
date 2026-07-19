import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str

    page_number: int | None = None
    table_id: str | None = None


@dataclass
class DocumentValidationResult:
    file_path: Path
    document_id: str | None
    tenant_id: str | None
    department: str | None
    content_hash: str | None

    issues: list[ValidationIssue] = field(
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


class RawDocumentValidator:

    MIN_DOCUMENT_TEXT_CHARS = 50
    MIN_PAGE_TEXT_CHARS = 10

    REQUIRED_METADATA_FIELDS = {
        "document_id",
        "tenant_id",
        "department",
        "filename",
        "file_type",
        "original_s3_key",
        "content_hash",
    }

    def validate_file(
        self,
        json_path: Path,
    ) -> DocumentValidationResult:

        try:
            document = self._load_json(json_path)

        except Exception as exc:

            result = DocumentValidationResult(
                file_path=json_path,
                document_id=None,
                tenant_id=None,
                department=None,
                content_hash=None,
            )

            result.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="INVALID_JSON",
                    message=str(exc),
                )
            )

            return result

        metadata = document.get("metadata", {})

        result = DocumentValidationResult(
            file_path=json_path,
            document_id=metadata.get("document_id"),
            tenant_id=metadata.get("tenant_id"),
            department=metadata.get("department"),
            content_hash=metadata.get("content_hash"),
        )

        self._validate_required_metadata(
            metadata,
            result,
        )

        self._validate_path_consistency(
            json_path,
            metadata,
            result,
        )

        self._validate_raw_text(
            document,
            result,
        )

        self._validate_document_warnings(
            document,
            result,
        )

        self._validate_pages(
            document,
            result,
        )

        self._validate_tables(
            document,
            result,
        )

        return result

    @staticmethod
    def _load_json(
        json_path: Path,
    ) -> dict[str, Any]:

        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "Raw document JSON root must be an object."
            )

        return data

    def _validate_required_metadata(
        self,
        metadata: dict,
        result: DocumentValidationResult,
    ) -> None:

        for field_name in self.REQUIRED_METADATA_FIELDS:

            value = metadata.get(field_name)

            if value is None or value == "":

                result.issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="MISSING_METADATA",
                        message=(
                            f"Missing required metadata: "
                            f"{field_name}"
                        ),
                    )
                )

    def _validate_path_consistency(
        self,
        json_path: Path,
        metadata: dict,
        result: DocumentValidationResult,
    ) -> None:

        # Expected:
        # raw/<tenant>/<department>/<hash>/<filename>.json

        parts = json_path.parts

        try:
            raw_index = parts.index("raw")

        except ValueError:

            result.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="INVALID_OUTPUT_PATH",
                    message=(
                        "Could not locate raw directory "
                        "in processed file path."
                    ),
                )
            )

            return

        relative_parts = parts[raw_index + 1:]

        if len(relative_parts) < 4:

            result.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="INVALID_OUTPUT_PATH",
                    message=(
                        f"Unexpected processed path: "
                        f"{json_path}"
                    ),
                )
            )

            return

        path_tenant = relative_parts[0]
        path_department = relative_parts[1]
        path_hash = relative_parts[2]

        checks = [
            (
                "tenant_id",
                path_tenant,
                metadata.get("tenant_id"),
            ),
            (
                "department",
                path_department,
                metadata.get("department"),
            ),
            (
                "content_hash",
                path_hash,
                metadata.get("content_hash"),
            ),
        ]

        for field_name, path_value, metadata_value in checks:

            if path_value != metadata_value:

                result.issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="PATH_METADATA_MISMATCH",
                        message=(
                            f"{field_name}: path={path_value}, "
                            f"metadata={metadata_value}"
                        ),
                    )
                )

    def _validate_raw_text(
        self,
        document: dict,
        result: DocumentValidationResult,
    ) -> None:

        raw_text = document.get("raw_text")

        if not isinstance(raw_text, str):

            result.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="INVALID_RAW_TEXT",
                    message="raw_text must be a string.",
                )
            )

            return

        meaningful_text = "".join(
            character
            for character in raw_text
            if character.isalnum()
        )

        if not meaningful_text:

            result.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="EMPTY_DOCUMENT_TEXT",
                    message="Document contains no meaningful text.",
                )
            )

        elif (
            len(meaningful_text)
            < self.MIN_DOCUMENT_TEXT_CHARS
        ):

            result.issues.append(
                ValidationIssue(
                    severity="WARNING",
                    code="NEAR_EMPTY_DOCUMENT_TEXT",
                    message=(
                        f"Only {len(meaningful_text)} "
                        f"meaningful text characters extracted."
                    ),
                )
            )

    def _validate_document_warnings(
        self,
        document: dict,
        result: DocumentValidationResult,
    ) -> None:

        warnings = document.get("warnings", [])

        if not isinstance(warnings, list):

            result.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="INVALID_WARNINGS_FIELD",
                    message="warnings must be a list.",
                )
            )

            return

        for warning in warnings:

            result.issues.append(
                ValidationIssue(
                    severity="WARNING",
                    code="PARSER_WARNING",
                    message=str(warning),
                )
            )

    def _validate_pages(
        self,
        document: dict,
        result: DocumentValidationResult,
    ) -> None:

        pages = document.get("pages", [])

        if not isinstance(pages, list):

            result.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="INVALID_PAGES_FIELD",
                    message="pages must be a list.",
                )
            )

            return

        for page in pages:

            page_number = page.get("page_number")

            text = page.get("text", "")

            page_metadata = page.get("metadata", {})

            ocr_required = page_metadata.get(
                "ocr_required",
                False,
            )

            ocr_attempted = page_metadata.get(
                "ocr_attempted",
                False,
            )

            ocr_succeeded = page_metadata.get(
                "ocr_succeeded",
                False,
            )

            meaningful_text = "".join(
                character
                for character in text
                if character.isalnum()
            )

            if not meaningful_text:

                result.issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="EMPTY_PAGE_TEXT",
                        message="Page contains no meaningful text.",
                        page_number=page_number,
                    )
                )

            elif (
                len(meaningful_text)
                < self.MIN_PAGE_TEXT_CHARS
            ):

                result.issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        code="NEAR_EMPTY_PAGE_TEXT",
                        message=(
                            f"Page contains only "
                            f"{len(meaningful_text)} meaningful "
                            f"text characters."
                        ),
                        page_number=page_number,
                    )
                )

            if ocr_required and not ocr_attempted:

                result.issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="OCR_NOT_ATTEMPTED",
                        message=(
                            "OCR was required but was not attempted."
                        ),
                        page_number=page_number,
                    )
                )

            if (
                ocr_required
                and ocr_attempted
                and not ocr_succeeded
            ):

                result.issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="OCR_FAILED",
                        message=(
                            "OCR was required but did not succeed."
                        ),
                        page_number=page_number,
                    )
                )

    def _validate_tables(
        self,
        document: dict,
        result: DocumentValidationResult,
    ) -> None:

        tables = document.get("tables", [])

        if not isinstance(tables, list):

            result.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="INVALID_TABLES_FIELD",
                    message="tables must be a list.",
                )
            )

            return

        for table in tables:

            table_id = table.get("table_id")

            headers = table.get("headers", [])

            rows = table.get("rows", [])

            if not headers and not rows:

                result.issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        code="EMPTY_TABLE",
                        message=(
                            "Extracted table contains no "
                            "headers or rows."
                        ),
                        table_id=table_id,
                    )
                )

            if not isinstance(headers, list):

                result.issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="INVALID_TABLE_HEADERS",
                        message="Table headers must be a list.",
                        table_id=table_id,
                    )
                )

            if not isinstance(rows, list):

                result.issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="INVALID_TABLE_ROWS",
                        message="Table rows must be a list.",
                        table_id=table_id,
                    )
                )