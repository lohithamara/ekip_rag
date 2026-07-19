from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:

    # Identity
    chunk_id: str
    document_id: str
    content_hash: str

    # Multi-tenant isolation
    tenant_id: str
    department: str

    # Content
    text: str
    chunk_index: int

    # Chunking information
    strategy: str
    chunking_version: str

    character_count: int
    token_count: int

    # Source provenance
    source_filename: str
    source_file_type: str
    source_s3_key: str

    # Chunk classification
    chunk_type: str

    # Fields with defaults start here
    page_numbers: list[int] = field(default_factory=list)
    table_ids: list[str] = field(default_factory=list)
    section_path: list[str] = field(default_factory=list)

    start_char: int | None = None
    end_char: int | None = None

    parent_chunk_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ChunkingResult:

    document_id: str

    tenant_id: str

    strategy: str

    chunking_version: str

    chunks: list[Chunk]

    total_chunks: int

    total_characters: int

    total_tokens: int

    metadata: dict[str, Any] = field(
        default_factory=dict
    )