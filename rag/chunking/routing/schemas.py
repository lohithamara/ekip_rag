from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentProfile:

    document_id: str

    file_type: str

    character_count: int

    token_count: int

    page_count: int

    pages_with_text: int

    table_count: int

    tables_with_rows: int

    heading_count: int

    paragraph_count: int

    average_paragraph_tokens: float

    has_pages: bool

    has_tables: bool

    has_structure: bool

    is_short_document: bool

    structure_confidence: float

    metadata: dict[str, Any] = field(
        default_factory=dict
    )