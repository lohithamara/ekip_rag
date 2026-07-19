from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentMetadata:
    document_id: str
    tenant_id: str
    department: str

    filename: str
    file_type: str
    content_type: str | None

    original_s3_key: str
    content_hash: str

    file_size_bytes: int | None = None
    page_count: int | None = None
    author: str | None = None
    title: str | None = None
    language: str | None = None

    ocr_used: bool = False

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedTable:
    table_id: str

    page_number: int | None

    headers: list[str]

    rows: list[list[Any]]

    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ExtractedPage:
    page_number: int
    text: str

    ocr_used: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class RawDocument:
    metadata: DocumentMetadata

    raw_text: str

    pages: list[ExtractedPage] = field(default_factory=list)

    tables: list[ExtractedTable] = field(default_factory=list)

    parser_name: str | None = None

    parser_version: str | None = None

    warnings: list[str] = field(default_factory=list)


@dataclass
class CleanDocument:
    metadata: DocumentMetadata

    clean_text: str

    pages: list[ExtractedPage] = field(default_factory=list)

    tables: list[ExtractedTable] = field(default_factory=list)

    cleaning_steps: list[str] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)
