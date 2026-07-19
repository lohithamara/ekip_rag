from dataclasses import dataclass
from dataclasses import field

from rag.reranking.schemas import (
    RerankedResult,
)


@dataclass(slots=True)
class ContextRequest:
    query: str
    reranked_results: list[RerankedResult]


@dataclass(slots=True)
class ContextChunk:
    """
    Chunk included in the final LLM context.
    """

    chunk_id: str

    document_id: str

    source_filename: str

    text: str

    token_count: int

    metadata: dict = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ContextResponse:
    """
    Final context sent to the prompt builder.
    """

    chunks: list[ContextChunk]

    total_chunks: int

    total_tokens: int