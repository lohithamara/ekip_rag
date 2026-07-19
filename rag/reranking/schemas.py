from dataclasses import dataclass, field

from rag.retrieval.schemas import RetrievalResult


@dataclass(slots=True)
class RerankRequest:
    """
    Request passed to a reranker.
    """

    query: str
    results: list[RetrievalResult]
    top_k: int


@dataclass(slots=True)
class RerankedResult:
    """
    A retrieved chunk after reranking.
    """

    retrieval_result: RetrievalResult

    # Score produced by the reranker
    rerank_score: float


@dataclass(slots=True)
class RerankResponse:
    """
    Output returned by a reranker.
    """

    results: list[RerankedResult]

    model_name: str

    total_candidates: int

    returned_candidates: int

    elapsed_seconds: float

    metadata: dict = field(
        default_factory=dict
    )