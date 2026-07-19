from dataclasses import dataclass

from rag.chunking.routing.router import (
    RoutingResult,
)
from rag.chunking.schemas import (
    Chunk,
    ChunkingResult,
)


@dataclass(frozen=True)
class RoutedChunkingResult:

    document_id: str

    tenant_id: str

    router_version: str

    routing_result: RoutingResult

    strategy_results: tuple[
        ChunkingResult,
        ...
    ]

    chunks: tuple[
        Chunk,
        ...
    ]

    total_chunks: int