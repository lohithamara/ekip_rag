from rag.reranking.base import BaseReranker
from rag.reranking.schemas import (
    RerankRequest,
    RerankResponse,
)


class RerankingService:

    def __init__(
        self,
        reranker: BaseReranker,
    ):
        self.reranker = reranker

    def rerank(
        self,
        request: RerankRequest,
    ) -> RerankResponse:

        if not request.query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if request.top_k <= 0:
            raise ValueError(
                "Top-k must be greater than zero."
            )

        if request.top_k > len(request.results):
            request.top_k = len(request.results)

        return self.reranker.rerank(request)