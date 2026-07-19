from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.model import EmbeddingModel
from rag.retrieval.dense_service import DenseRetrievalService
from rag.retrieval.hybrid_service import HybridRetrievalService
from rag.retrieval.sparse_service import SparseRetrievalService
from rag.retrieval.schemas import RetrievalRequest
from rag.sparse_retrieval.bm25_index import BM25Index
from rag.sparse_retrieval.config import SparseRetrievalConfig
from rag.vector_store.config import VectorDBConfig
from rag.vector_store.qdrant_store import QdrantVectorStore

from rag.reranking.config import RerankerConfig
from rag.reranking.factory import create_reranker
from rag.reranking.schemas import RerankRequest
from rag.reranking.service import RerankingService

from security.authorization.scope import (
    DepartmentScope,
)

from rag.retrieval.schemas import (
    RetrievalRequest,
)

def main():

    vector_store = QdrantVectorStore(
        VectorDBConfig()
    )

    embedding_model = EmbeddingModel(
        EmbeddingConfig()
    )

    dense_service = DenseRetrievalService(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    sparse_service = SparseRetrievalService(
        BM25Index.load(
            SparseRetrievalConfig()
        )
    )

    hybrid_service = HybridRetrievalService(
        dense_service=dense_service,
        sparse_service=sparse_service,
    )

    reranker = create_reranker(
        RerankerConfig()
    )

    reranking_service = RerankingService(
        reranker
    )

    query = (
        "What are the approval limits "
        "for the finance team?"
    )

    retrieved = hybrid_service.retrieve(

        RetrievalRequest(

            query=query,

            scope=DepartmentScope(

                tenant_id="tenant_1",

                departments=(
                    "finance",
                ),
            ),

            limit=10,
        )
    )

    print("=" * 70)
    print("HYBRID RETRIEVAL")
    print("=" * 70)

    for index, result in enumerate(
        retrieved,
        start=1,
    ):
        print(
            f"{index:2d}. "
            f"{result.metadata.get('source_filename')} "
            f"({result.score:.4f})"
        )

    request = RerankRequest(
        query=query,
        results=retrieved,
        top_k=5,
    )

    response = reranking_service.rerank(
        request
    )

    print()
    print("=" * 70)
    print("AFTER RERANKING")
    print("=" * 70)

    pairs = list(
        zip(
            retrieved,
            reranker.model.predict(
                [(query, r.text) for r in retrieved]
            ),
            strict=True,
        )
    )

    pairs.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    print("\nALL RERANK SCORES\n")

    for i, (result, score) in enumerate(pairs, 1):
        print(
            f"{i:2d}. "
            f"{score:8.4f} "
            f"{result.metadata.get('source_filename')}"
        )
    vector_store.close()


if __name__ == "__main__":
    main()