from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StrategyManifest:

    strategy: str

    content_scope: str

    reason: str

    priority: int

    total_chunks: int

    chunk_type_counts: dict[str, int]

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ChunkManifest:

    manifest_version: str

    document_id: str

    content_hash: str

    tenant_id: str

    department: str

    source_filename: str

    source_file_type: str

    source_s3_key: str

    router_version: str

    chunking_version: str

    total_chunks: int

    chunk_type_counts: dict[str, int]

    strategy_counts: dict[str, int]

    strategies: tuple[
        StrategyManifest,
        ...
    ]

    chunking_config: dict[str, Any]

    generated_at: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )