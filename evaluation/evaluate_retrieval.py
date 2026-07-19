import json
from pathlib import Path

from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.model import EmbeddingModel
from rag.evaluation.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from rag.retrieval.dense_service import DenseRetrievalService
from rag.retrieval.hybrid_service import HybridRetrievalService
from rag.retrieval.sparse_service import SparseRetrievalService
from rag.sparse_retrieval.bm25_index import BM25Index
from rag.sparse_retrieval.config import SparseRetrievalConfig
from rag.vector_store.config import VectorDBConfig
from rag.vector_store.qdrant_store import QdrantVectorStore

from rag.reranking.config import RerankerConfig
from rag.reranking.factory import create_reranker
from rag.reranking.schemas import RerankRequest
from rag.reranking.service import RerankingService

EVALUATION_FILE = Path(
    "data/evaluation/retrieval_queries.json"
)


def document_ranking(results):

    documents = []
    seen = set()

    for result in results:

        filename = result.metadata.get(
            "source_filename"
        )

        if not filename:
            continue

        if filename in seen:
            continue

        seen.add(filename)
        documents.append(filename)

    return documents

def evaluate_reranker(
    hybrid_service,
    reranking_service,
    queries,
    candidate_limit=10,
    top_k=5,
):

    recall_1 = []
    recall_3 = []
    recall_5 = []
    reciprocal_ranks = []

    print()
    print("=" * 60)
    print("Hybrid + Reranker")
    print("=" * 60)

    for index, item in enumerate(
        queries,
        start=1,
    ):

        retrieved = hybrid_service.retrieve(
            query=item["query"],
            tenant_id=item["tenant_id"],
            department=item["department"],
            limit=candidate_limit,
        )

        request = RerankRequest(
            query=item["query"],
            results=retrieved,
            top_k=top_k,
        )

        response = reranking_service.rerank(
            request
        )

        documents = [
            r.retrieval_result.metadata.get(
                "source_filename"
            )
            for r in response.results
        ]

        relevant = item["relevant_documents"]

        r1 = recall_at_k(documents, relevant, 1)
        r3 = recall_at_k(documents, relevant, 3)
        r5 = recall_at_k(documents, relevant, 5)
        rr = reciprocal_rank(
            documents,
            relevant,
        )

        recall_1.append(r1)
        recall_3.append(r3)
        recall_5.append(r5)
        reciprocal_ranks.append(rr)

        status = (
            "PASS"
            if rr > 0
            else "MISS"
        )

        print(
            f"[{index}/{len(queries)}] "
            f"{status} "
            f"RR={rr:.3f} "
            f"{item['query']}"
        )

    query_count = len(queries)

    return {
        "retriever": "Hybrid+Reranker",
        "queries": query_count,
        "recall_at_1": sum(recall_1) / query_count,
        "recall_at_3": sum(recall_3) / query_count,
        "recall_at_5": sum(recall_5) / query_count,
        "mrr": mean_reciprocal_rank(
            reciprocal_ranks
        ),
    }

def evaluate_service(
    name,
    service,
    queries,
    limit=5,
):

    recall_1 = []
    recall_3 = []
    recall_5 = []
    reciprocal_ranks = []

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    for index, item in enumerate(
        queries,
        start=1,
    ):

        results = service.retrieve(
            query=item["query"],
            tenant_id=item["tenant_id"],
            department=item["department"],
            limit=limit,
        )

        documents = document_ranking(
            results
        )

        relevant = item[
            "relevant_documents"
        ]

        r1 = recall_at_k(
            documents,
            relevant,
            1,
        )

        r3 = recall_at_k(
            documents,
            relevant,
            3,
        )

        r5 = recall_at_k(
            documents,
            relevant,
            5,
        )

        rr = reciprocal_rank(
            documents,
            relevant,
        )

        recall_1.append(r1)
        recall_3.append(r3)
        recall_5.append(r5)
        reciprocal_ranks.append(rr)

        status = (
            "PASS"
            if rr > 0
            else "MISS"
        )

        print(
            f"[{index}/{len(queries)}] "
            f"{status} "
            f"RR={rr:.3f} "
            f"{item['query']}"
        )

    query_count = len(queries)

    return {
        "retriever": name,
        "queries": query_count,
        "recall_at_1": (
            sum(recall_1) / query_count
        ),
        "recall_at_3": (
            sum(recall_3) / query_count
        ),
        "recall_at_5": (
            sum(recall_5) / query_count
        ),
        "mrr": mean_reciprocal_rank(
            reciprocal_ranks
        ),
    }


def print_summary(metrics):

    print()
    print("=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"{'Retriever':<12}"
        f"{'R@1':>10}"
        f"{'R@3':>10}"
        f"{'R@5':>10}"
        f"{'MRR':>10}"
    )

    print("-" * 52)

    for item in metrics:

        print(
            f"{item['retriever']:<12}"
            f"{item['recall_at_1']:>10.3f}"
            f"{item['recall_at_3']:>10.3f}"
            f"{item['recall_at_5']:>10.3f}"
            f"{item['mrr']:>10.3f}"
        )


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

    sparse_config = SparseRetrievalConfig()

    if not Path(
        sparse_config.index_path
    ).is_file():
        raise RuntimeError(
            "Sparse index does not exist."
        )

    vector_store = QdrantVectorStore(
        VectorDBConfig()
    )

    if not vector_store.collection_exists():

        vector_store.close()

        raise RuntimeError(
            "Vector collection does not exist."
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
                sparse_config
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
        metrics = []

        metrics.append(
            evaluate_service(
                name="Dense",
                service=dense_service,
                queries=queries,
            )
        )

        metrics.append(
            evaluate_service(
                name="Sparse",
                service=sparse_service,
                queries=queries,
            )
        )

        metrics.append(
            evaluate_service(
                name="Hybrid",
                service=hybrid_service,
                queries=queries,
            )
        )
        metrics.append(
            evaluate_reranker(
                hybrid_service=hybrid_service,
                reranking_service=reranking_service,
                queries=queries,
            )
        )
        print_summary(metrics)

    finally:

        vector_store.close()


if __name__ == "__main__":
    main()