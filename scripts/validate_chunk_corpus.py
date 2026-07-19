from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

from rag.chunking.storage.manifest import (
    ManifestSerializer,
)
from rag.chunking.storage.serializer import (
    ChunkSerializer,
)


CHUNK_STORAGE_ROOT = Path(
    "data/processed/chunks"
)

EXPECTED_CHUNK_SIZE = 512


@dataclass
class ValidationIssue:

    severity: str

    code: str

    path: str

    message: str


@dataclass
class CorpusValidationReport:

    document_directories: int = 0

    valid_documents: int = 0

    invalid_documents: int = 0

    total_chunks: int = 0

    text_chunks: int = 0

    table_chunks: int = 0

    global_chunk_ids: set[str] = field(
        default_factory=set
    )

    strategy_counts: Counter = field(
        default_factory=Counter
    )

    chunk_type_counts: Counter = field(
        default_factory=Counter
    )

    tenant_counts: Counter = field(
        default_factory=Counter
    )

    department_counts: Counter = field(
        default_factory=Counter
    )

    issues: list[ValidationIssue] = field(
        default_factory=list
    )


class ChunkCorpusValidator:

    CHUNKS_FILENAME = "chunks.jsonl"

    MANIFEST_FILENAME = "manifest.json"

    ALLOWED_CHUNK_TYPES = {
        "text",
        "table",
    }

    ALLOWED_STRATEGIES = {
        "fixed_size",
        "recursive",
        "sentence",
        "paragraph",
        "page_aware",
        "structure_aware",
        "table_aware",
        "json_aware",
    }

    def __init__(
        self,
        storage_root: Path,
        max_chunk_size: int,
    ):
        self.storage_root = Path(storage_root)

        self.max_chunk_size = max_chunk_size

    def validate(
        self,
    ) -> CorpusValidationReport:

        report = CorpusValidationReport()

        document_dirs = (
            self._discover_document_directories()
        )

        report.document_directories = len(
            document_dirs
        )

        for document_dir in document_dirs:

            issue_count_before = len(
                report.issues
            )

            self._validate_document_directory(
                document_dir=document_dir,
                report=report,
            )

            issue_count_after = len(
                report.issues
            )

            new_issues = report.issues[
                issue_count_before:
                issue_count_after
            ]

            has_error = any(
                issue.severity == "ERROR"
                for issue in new_issues
            )

            if has_error:

                report.invalid_documents += 1

            else:

                report.valid_documents += 1

        return report

    def _discover_document_directories(
        self,
    ) -> list[Path]:

        if not self.storage_root.exists():

            raise RuntimeError(
                f"Chunk storage root does not exist: "
                f"{self.storage_root}"
            )

        directories = set()

        for artifact_path in (
            self.storage_root.rglob("*")
        ):

            if not artifact_path.is_file():
                continue

            if artifact_path.name in {
                self.CHUNKS_FILENAME,
                self.MANIFEST_FILENAME,
            }:

                directories.add(
                    artifact_path.parent
                )

        return sorted(directories)

    def _validate_document_directory(
        self,
        document_dir: Path,
        report: CorpusValidationReport,
    ) -> None:

        chunks_path = (
            document_dir
            / self.CHUNKS_FILENAME
        )

        manifest_path = (
            document_dir
            / self.MANIFEST_FILENAME
        )

        if not chunks_path.is_file():

            self._add_issue(
                report=report,
                severity="ERROR",
                code="MISSING_CHUNKS",
                path=document_dir,
                message=(
                    "chunks.jsonl is missing."
                ),
            )

            return

        if not manifest_path.is_file():

            self._add_issue(
                report=report,
                severity="ERROR",
                code="MISSING_MANIFEST",
                path=document_dir,
                message=(
                    "manifest.json is missing."
                ),
            )

            return

        try:

            manifest = (
                ManifestSerializer.load_json(
                    manifest_path
                )
            )

        except Exception as exc:

            self._add_issue(
                report=report,
                severity="ERROR",
                code="INVALID_MANIFEST",
                path=manifest_path,
                message=str(exc),
            )

            return

        try:

            chunks = ChunkSerializer.load_jsonl(
                chunks_path
            )

        except Exception as exc:

            self._add_issue(
                report=report,
                severity="ERROR",
                code="INVALID_CHUNKS",
                path=chunks_path,
                message=str(exc),
            )

            return

        self._validate_directory_layout(
            document_dir=document_dir,
            manifest=manifest,
            report=report,
        )

        self._validate_manifest(
            manifest=manifest,
            chunks=chunks,
            manifest_path=manifest_path,
            report=report,
        )

        self._validate_chunks(
            manifest=manifest,
            chunks=chunks,
            chunks_path=chunks_path,
            report=report,
        )

        report.total_chunks += len(chunks)

        report.tenant_counts[
            manifest.tenant_id
        ] += 1

        report.department_counts[
            (
                manifest.tenant_id,
                manifest.department,
            )
        ] += 1

    def _validate_directory_layout(
        self,
        document_dir,
        manifest,
        report,
    ) -> None:

        try:

            relative = document_dir.relative_to(
                self.storage_root
            )

        except ValueError:

            self._add_issue(
                report=report,
                severity="ERROR",
                code="OUTSIDE_STORAGE_ROOT",
                path=document_dir,
                message=(
                    "Document directory is outside "
                    "the configured storage root."
                ),
            )

            return

        parts = relative.parts

        if len(parts) != 3:

            self._add_issue(
                report=report,
                severity="ERROR",
                code="INVALID_DIRECTORY_LAYOUT",
                path=document_dir,
                message=(
                    "Expected path layout "
                    "tenant/department/content_hash."
                ),
            )

            return

        tenant_id, department, content_hash = (
            parts
        )

        if tenant_id != manifest.tenant_id:

            self._add_issue(
                report=report,
                severity="ERROR",
                code="TENANT_PATH_MISMATCH",
                path=document_dir,
                message=(
                    "Directory tenant does not match "
                    "manifest tenant_id."
                ),
            )

        if department != manifest.department:

            self._add_issue(
                report=report,
                severity="ERROR",
                code="DEPARTMENT_PATH_MISMATCH",
                path=document_dir,
                message=(
                    "Directory department does not "
                    "match manifest department."
                ),
            )

        if content_hash != manifest.content_hash:

            self._add_issue(
                report=report,
                severity="ERROR",
                code="CONTENT_HASH_PATH_MISMATCH",
                path=document_dir,
                message=(
                    "Directory content hash does not "
                    "match manifest content_hash."
                ),
            )

    def _validate_manifest(
        self,
        manifest,
        chunks,
        manifest_path,
        report,
    ) -> None:

        if not manifest.document_id:

            self._add_issue(
                report,
                "ERROR",
                "EMPTY_DOCUMENT_ID",
                manifest_path,
                "Manifest document_id is empty.",
            )

        if not manifest.tenant_id:

            self._add_issue(
                report,
                "ERROR",
                "EMPTY_TENANT_ID",
                manifest_path,
                "Manifest tenant_id is empty.",
            )

        if not manifest.content_hash:

            self._add_issue(
                report,
                "ERROR",
                "EMPTY_CONTENT_HASH",
                manifest_path,
                "Manifest content_hash is empty.",
            )

        if manifest.total_chunks != len(chunks):

            self._add_issue(
                report,
                "ERROR",
                "MANIFEST_CHUNK_COUNT_MISMATCH",
                manifest_path,
                (
                    f"Manifest total_chunks="
                    f"{manifest.total_chunks}, "
                    f"actual={len(chunks)}."
                ),
            )

        actual_strategy_counts = Counter(
            chunk.strategy
            for chunk in chunks
        )

        actual_type_counts = Counter(
            chunk.chunk_type
            for chunk in chunks
        )

        if (
            dict(sorted(actual_strategy_counts.items()))
            != manifest.strategy_counts
        ):

            self._add_issue(
                report,
                "ERROR",
                "STRATEGY_COUNT_MISMATCH",
                manifest_path,
                "Manifest strategy counts are incorrect.",
            )

        if (
            dict(sorted(actual_type_counts.items()))
            != manifest.chunk_type_counts
        ):

            self._add_issue(
                report,
                "ERROR",
                "CHUNK_TYPE_COUNT_MISMATCH",
                manifest_path,
                "Manifest chunk type counts are incorrect.",
            )

        decision_strategies = {
            item.strategy
            for item in manifest.strategies
        }

        actual_strategies = set(
            actual_strategy_counts
        )

        if decision_strategies != actual_strategies:

            self._add_issue(
                report,
                "ERROR",
                "MANIFEST_STRATEGY_SET_MISMATCH",
                manifest_path,
                (
                    "Manifest strategy entries do not "
                    "match stored chunk strategies."
                ),
            )

        for strategy_manifest in (
            manifest.strategies
        ):

            actual_count = (
                actual_strategy_counts[
                    strategy_manifest.strategy
                ]
            )

            if (
                strategy_manifest.total_chunks
                != actual_count
            ):

                self._add_issue(
                    report,
                    "ERROR",
                    "STRATEGY_MANIFEST_COUNT_MISMATCH",
                    manifest_path,
                    (
                        f"Strategy "
                        f"{strategy_manifest.strategy}: "
                        f"manifest="
                        f"{strategy_manifest.total_chunks}, "
                        f"actual={actual_count}."
                    ),
                )

    def _validate_chunks(
        self,
        manifest,
        chunks,
        chunks_path,
        report,
    ) -> None:

        local_ids = set()

        strategy_indexes = set()

        for line_number, chunk in enumerate(
            chunks,
            start=1,
        ):

            report.strategy_counts[
                chunk.strategy
            ] += 1

            report.chunk_type_counts[
                chunk.chunk_type
            ] += 1

            if chunk.chunk_type == "text":

                report.text_chunks += 1

            elif chunk.chunk_type == "table":

                report.table_chunks += 1

            if not chunk.chunk_id:

                self._add_chunk_issue(
                    report,
                    "ERROR",
                    "EMPTY_CHUNK_ID",
                    chunks_path,
                    line_number,
                    "chunk_id is empty.",
                )

            if chunk.chunk_id in local_ids:

                self._add_chunk_issue(
                    report,
                    "ERROR",
                    "DUPLICATE_LOCAL_CHUNK_ID",
                    chunks_path,
                    line_number,
                    (
                        f"Duplicate chunk_id "
                        f"{chunk.chunk_id}."
                    ),
                )

            local_ids.add(
                chunk.chunk_id
            )

            if (
                chunk.chunk_id
                in report.global_chunk_ids
            ):

                self._add_chunk_issue(
                    report,
                    "ERROR",
                    "DUPLICATE_GLOBAL_CHUNK_ID",
                    chunks_path,
                    line_number,
                    (
                        f"chunk_id already exists "
                        f"elsewhere in corpus: "
                        f"{chunk.chunk_id}"
                    ),
                )

            report.global_chunk_ids.add(
                chunk.chunk_id
            )

            strategy_identity = (
                chunk.strategy,
                chunk.chunk_index,
            )

            if (
                strategy_identity
                in strategy_indexes
            ):

                self._add_chunk_issue(
                    report,
                    "ERROR",
                    "DUPLICATE_STRATEGY_INDEX",
                    chunks_path,
                    line_number,
                    (
                        "Duplicate "
                        "(strategy, chunk_index)."
                    ),
                )

            strategy_indexes.add(
                strategy_identity
            )

            if (
                chunk.document_id
                != manifest.document_id
            ):

                self._add_chunk_issue(
                    report,
                    "ERROR",
                    "CHUNK_DOCUMENT_MISMATCH",
                    chunks_path,
                    line_number,
                    (
                        "Chunk document_id does not "
                        "match manifest."
                    ),
                )

            if (
                chunk.content_hash
                != manifest.content_hash
            ):

                self._add_chunk_issue(
                    report,
                    "ERROR",
                    "CHUNK_CONTENT_HASH_MISMATCH",
                    chunks_path,
                    line_number,
                    (
                        "Chunk content_hash does not "
                        "match manifest."
                    ),
                )

            if (
                chunk.tenant_id
                != manifest.tenant_id
            ):

                self._add_chunk_issue(
                    report,
                    "ERROR",
                    "CHUNK_TENANT_MISMATCH",
                    chunks_path,
                    line_number,
                    (
                        "Chunk tenant_id does not "
                        "match manifest."
                    ),
                )

            if (
                chunk.department
                != manifest.department
            ):

                self._add_chunk_issue(
                    report,
                    "ERROR",
                    "CHUNK_DEPARTMENT_MISMATCH",
                    chunks_path,
                    line_number,
                    (
                        "Chunk department does not "
                        "match manifest."
                    ),
                )

            if (
                chunk.strategy
                not in self.ALLOWED_STRATEGIES
            ):

                self._add_chunk_issue(
                    report,
                    "ERROR",
                    "UNKNOWN_STRATEGY",
                    chunks_path,
                    line_number,
                    (
                        f"Unknown strategy: "
                        f"{chunk.strategy}"
                    ),
                )

            if (
                chunk.chunk_type
                not in self.ALLOWED_CHUNK_TYPES
            ):

                self._add_chunk_issue(
                    report,
                    "ERROR",
                    "UNKNOWN_CHUNK_TYPE",
                    chunks_path,
                    line_number,
                    (
                        f"Unknown chunk type: "
                        f"{chunk.chunk_type}"
                    ),
                )

            if not chunk.text.strip():

                self._add_chunk_issue(
                    report,
                    "ERROR",
                    "EMPTY_CHUNK_TEXT",
                    chunks_path,
                    line_number,
                    "Chunk text is empty.",
                )

            if chunk.token_count <= 0:

                self._add_chunk_issue(
                    report,
                    "ERROR",
                    "INVALID_TOKEN_COUNT",
                    chunks_path,
                    line_number,
                    (
                        f"Invalid token_count="
                        f"{chunk.token_count}."
                    ),
                )

            if (
                chunk.token_count
                > self.max_chunk_size
            ):

                self._add_chunk_issue(
                    report,
                    "ERROR",
                    "OVERSIZED_CHUNK",
                    chunks_path,
                    line_number,
                    (
                        f"token_count="
                        f"{chunk.token_count} "
                        f"exceeds max="
                        f"{self.max_chunk_size}."
                    ),
                )

            if chunk.chunk_index < 0:

                self._add_chunk_issue(
                    report,
                    "ERROR",
                    "NEGATIVE_CHUNK_INDEX",
                    chunks_path,
                    line_number,
                    (
                        f"chunk_index="
                        f"{chunk.chunk_index}."
                    ),
                )

            self._validate_provenance(
                chunk=chunk,
                chunks_path=chunks_path,
                line_number=line_number,
                report=report,
            )

    def _validate_provenance(
        self,
        chunk,
        chunks_path,
        line_number,
        report,
    ) -> None:

        # page_aware → page_numbers

        if (
            chunk.strategy == "page_aware"
            and not chunk.page_numbers
        ):

            self._add_chunk_issue(
                report=report,
                severity="ERROR",
                code="MISSING_PAGE_PROVENANCE",
                path=chunks_path,
                line_number=line_number,
                message=(
                    "page_aware chunk has no "
                    "page_numbers."
                ),
            )

        # table_aware → table_ids

        if (
            chunk.strategy == "table_aware"
            and not chunk.table_ids
        ):

            self._add_chunk_issue(
                report=report,
                severity="ERROR",
                code="MISSING_TABLE_PROVENANCE",
                path=chunks_path,
                line_number=line_number,
                message=(
                    "table_aware chunk has no "
                    "table_ids."
                ),
            )

        # structure_aware → section_path

        if (
            chunk.strategy == "structure_aware"
            and not chunk.section_path
        ):

            self._add_chunk_issue(
                report=report,
                severity="WARNING",
                code="MISSING_SECTION_PROVENANCE",
                path=chunks_path,
                line_number=line_number,
                message=(
                    "structure_aware chunk has no "
                    "section_path."
                ),
            )

    @staticmethod
    def _add_issue(
        report,
        severity,
        code,
        path,
        message,
    ) -> None:

        report.issues.append(
            ValidationIssue(
                severity=severity,
                code=code,
                path=str(path),
                message=message,
            )
        )

    @classmethod
    def _add_chunk_issue(
        cls,
        report,
        severity,
        code,
        path,
        line_number,
        message,
    ) -> None:

        cls._add_issue(
            report=report,
            severity=severity,
            code=code,
            path=path,
            message=(
                f"line={line_number}: {message}"
            ),
        )


def print_report(
    report: CorpusValidationReport,
) -> None:

    errors = [
        issue
        for issue in report.issues
        if issue.severity == "ERROR"
    ]

    warnings = [
        issue
        for issue in report.issues
        if issue.severity == "WARNING"
    ]

    print()
    print("===================================")
    print("CHUNK CORPUS VALIDATION SUMMARY")
    print("===================================")

    print(
        f"Document directories : "
        f"{report.document_directories}"
    )

    print(
        f"Valid documents       : "
        f"{report.valid_documents}"
    )

    print(
        f"Invalid documents     : "
        f"{report.invalid_documents}"
    )

    print(
        f"Total chunks          : "
        f"{report.total_chunks}"
    )

    print(
        f"Unique global IDs     : "
        f"{len(report.global_chunk_ids)}"
    )

    print(
        f"Text chunks           : "
        f"{report.text_chunks}"
    )

    print(
        f"Table chunks          : "
        f"{report.table_chunks}"
    )

    print(
        f"Errors                : "
        f"{len(errors)}"
    )

    print(
        f"Warnings              : "
        f"{len(warnings)}"
    )

    print()
    print("CHUNKS BY STRATEGY")
    print("-----------------------------------")

    for strategy, count in sorted(
        report.strategy_counts.items()
    ):

        print(
            f"{strategy:<20}: {count}"
        )

    print()
    print("ISSUES BY CODE")
    print("-----------------------------------")

    issue_counts = Counter(
        (
            issue.severity,
            issue.code,
        )
        for issue in report.issues
    )

    if not issue_counts:

        print("No issues detected.")

    else:

        for (
            severity,
            code,
        ), count in sorted(
            issue_counts.items()
        ):

            print(
                f"{severity:<10}"
                f"{code:<35}"
                f"{count}"
            )

    if report.issues:

        print()
        print("ISSUE DETAILS")
        print("-----------------------------------")

        for index, issue in enumerate(
            report.issues,
            start=1,
        ):

            print()
            print(f"ISSUE #{index}")
            print(f"Severity : {issue.severity}")
            print(f"Code     : {issue.code}")
            print(f"Path     : {issue.path}")
            print(f"Message  : {issue.message}")

    print()
    print("===================================")
    print("FINAL STATUS")
    print("===================================")

    if errors:

        print(
            f"FAIL: {len(errors)} validation "
            "errors detected."
        )

    elif warnings:

        print(
            "PASS WITH WARNINGS: "
            f"{len(warnings)} warnings detected."
        )

    else:

        print(
            "PASS: Persisted chunk corpus is valid."
        )


def main():

    validator = ChunkCorpusValidator(
        storage_root=CHUNK_STORAGE_ROOT,
        max_chunk_size=EXPECTED_CHUNK_SIZE,
    )

    report = validator.validate()

    print_report(report)


if __name__ == "__main__":
    main()