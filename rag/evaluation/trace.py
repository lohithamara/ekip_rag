from dataclasses import dataclass

from rag.generation.schemas import (
    ContextResponse,
)

from rag.reranking.schemas import (
    RerankResponse,
)

from rag.retrieval.schemas import (
    RetrievalResult,
)


@dataclass(frozen=True, slots=True)
class RAGTrace:

    original_query: str

    retrieval_query: str

    retrieved_results: tuple[
        RetrievalResult,
        ...,
    ]

    reranked_response: RerankResponse

    context: ContextResponse

    generated_answer: str

    validated_sources: tuple[
        str,
        ...,
    ]