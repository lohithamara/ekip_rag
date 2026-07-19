from rag.retrieval.dense_service import (
    DenseRetrievalService,
)
from rag.retrieval.rrf import (
    reciprocal_rank_fusion,
)
from rag.retrieval.schemas import RetrievalRequest
from rag.retrieval.sparse_service import (
    SparseRetrievalService,
)


class HybridRetrievalService:

    def __init__(
        self,
        dense_service: DenseRetrievalService,
        sparse_service: SparseRetrievalService,
    ):
        self.dense_service = dense_service
        self.sparse_service = sparse_service

    def retrieve(
        self,
        request: RetrievalRequest,
    ):

        dense_results = (
            self.dense_service.retrieve(
                request
            )
        )

        sparse_results = (
            self.sparse_service.retrieve(
                request
            )
        )

        fused_results = (
            reciprocal_rank_fusion(
                result_lists=[
                    dense_results,
                    sparse_results,
                ],
                limit=request.limit,
                weights=[
                    1.0,
                    1.0,
                ],
            )
        )

        return fused_results