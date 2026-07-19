import json
from pathlib import Path

from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.model import EmbeddingModel
from rag.generation.context_builder import ContextBuilder
from rag.generation.config import ContextBuilderConfig
from rag.generation.schemas import ContextRequest
from rag.reranking.config import RerankerConfig
from rag.reranking.factory import create_reranker
from rag.reranking.schemas import RerankRequest
from rag.reranking.service import RerankingService
from rag.retrieval.dense_service import DenseRetrievalService
from rag.retrieval.hybrid_service import HybridRetrievalService
from rag.retrieval.sparse_service import SparseRetrievalService
from rag.sparse_retrieval.bm25_index import BM25Index
from rag.sparse_retrieval.config import SparseRetrievalConfig
from rag.vector_store.config import VectorDBConfig
from rag.vector_store.qdrant_store import QdrantVectorStore


EVALUATION_FILE = Path(
    "data/evaluation/retrieval_queries.json"
)


def source_filenames(chunks):

    return {
        chunk.source_filename
        for chunk in chunks
        if chunk.source_filename
    }


def main():

    if not EVALUATION_FILE.is_file():
        raise RuntimeError(
            "Evaluation dataset does not exist."
        )

    queries = json.loads(
        EVALUATION_FILE.read_text(
            encoding="utf-8"
        )
    )

    if not queries:
        raise RuntimeError(
            "Evaluation dataset is empty."
        )

    vector_store = QdrantVectorStore(
        VectorDBConfig()
    )

    try:

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

        reranking_service = RerankingService(
            create_reranker(
                RerankerConfig()
            )
        )

        context_builder = ContextBuilder(
            ContextBuilderConfig()
        )

        total_before_tokens = 0
        total_after_tokens = 0

        retained_count = 0

        print()
        print("=" * 100)
        print("CONTEXT BUILDER EVALUATION")
        print("=" * 100)

        for index, item in enumerate(
            queries,
            start=1,
        ):

            retrieved = hybrid_service.retrieve(
                query=item["query"],
                tenant_id=item["tenant_id"],
                department=item["department"],
                limit=10,
            )

            reranked = reranking_service.rerank(
                RerankRequest(
                    query=item["query"],
                    results=retrieved,
                    top_k=5,
                )
            )

            before_tokens = sum(
                result.retrieval_result.metadata.get(
                    "token_count",
                    0,
                )
                for result in reranked.results
            )

            context = context_builder.build(
                ContextRequest(
                    query=item["query"],
                    reranked_results=reranked.results,
                )
            )

            after_tokens = context.total_tokens

            relevant_documents = set(
                item["relevant_documents"]
            )

            retained_documents = (
                source_filenames(
                    context.chunks
                )
            )

            retained = bool(
                relevant_documents
                & retained_documents
            )

            if retained:
                retained_count += 1
            else:
                print()
                print("FAILURE DETAILS")
                print(f"Query    : {item['query']}")
                print(
                    f"Expected : "
                    f"{sorted(relevant_documents)}"
                )
                print(
                    f"Retained : "
                    f"{sorted(retained_documents)}"
                )
                print()
            total_before_tokens += (
                before_tokens
            )

            total_after_tokens += (
                after_tokens
            )

            if before_tokens > 0:
                reduction = (
                    (
                        before_tokens
                        - after_tokens
                    )
                    / before_tokens
                    * 100
                )
            else:
                reduction = 0.0

            status = (
                "PASS"
                if retained
                else "FAIL"
            )

            print(
                f"[{index:2d}/{len(queries)}] "
                f"{status:<4} "
                f"before={before_tokens:5d} "
                f"after={after_tokens:5d} "
                f"reduction={reduction:6.1f}% "
                f"{item['query']}"
            )

        query_count = len(queries)

        retention_rate = (
            retained_count
            / query_count
        )

        if total_before_tokens > 0:
            overall_reduction = (
                (
                    total_before_tokens
                    - total_after_tokens
                )
                / total_before_tokens
                * 100
            )
        else:
            overall_reduction = 0.0

        average_before = (
            total_before_tokens
            / query_count
        )

        average_after = (
            total_after_tokens
            / query_count
        )

        print()
        print("=" * 100)
        print("CONTEXT EVALUATION SUMMARY")
        print("=" * 100)

        print(
            f"Queries                  : "
            f"{query_count}"
        )

        print(
            f"Relevant source retained : "
            f"{retained_count}/{query_count}"
        )

        print(
            f"Retention rate           : "
            f"{retention_rate:.3f}"
        )

        print(
            f"Average tokens before    : "
            f"{average_before:.1f}"
        )

        print(
            f"Average tokens after     : "
            f"{average_after:.1f}"
        )

        print(
            f"Overall token reduction  : "
            f"{overall_reduction:.1f}%"
        )

    finally:

        vector_store.close()


if __name__ == "__main__":
    main()