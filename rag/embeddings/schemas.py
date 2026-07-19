from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EmbeddingRecord:
    chunk_id: str
    document_id: str
    tenant_id: str
    department: str

    vector: tuple[float, ...]

    model_name: str
    embedding_version: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class EmbeddingResult:
    document_id: str
    tenant_id: str

    model_name: str
    embedding_version: str

    dimension: int

    records: tuple[EmbeddingRecord, ...]

    total_embeddings: int