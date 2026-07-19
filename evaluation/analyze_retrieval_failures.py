import json
from pathlib import Path

from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.model import EmbeddingModel
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

OUTPUT_FILE = Path(
    "data/evaluation/retrieval_failure_report.json"
)

CANDIDATE_LIMIT = 20
FINAL_LIMIT = 5


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


def relevant_rank(
    documents,
    relevant_documents,
):

    relevant = set(relevant_documents)

    for rank, document in enumerate(
        documents,
        start=1,
    ):
        if document in relevant:
            return rank

    return None


def categorize_failure(
    dense_rank,
    sparse_rank,
    hybrid_rank,
):

    if hybrid_rank is not None:
        if hybrid_rank == 1:
            return "success_rank_1"

        return "success_low_rank"

    if (
        dense_rank is None
        and sparse_rank is None
    ):
        return "both_retrievers_miss"

    if (
        dense_rank is None
        and sparse_rank is not None
    ):
        return "dense_miss_fusion_failure"

    if (
        dense_rank is not None
        and sparse_rank is None
    ):
        return "sparse_miss_fusion_failure"

    return "fusion_ranking_failure"


def main():

    queries = json.loads(
        EVALUATION_FILE.read_text(
            encoding="utf-8"
        )
    )

    sparse_config = SparseRetrievalConfig()

    vector_store = QdrantVectorStore(
        VectorDBConfig()
    )

    if not vector_store.collection_exists():

        vector_store.close()

        raise RuntimeError(
            "Vector collection does not exist."
        )

    try:

        dense_service = DenseRetrievalService(
            embedding_model=EmbeddingModel(
                EmbeddingConfig()
            ),
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

        report = []

        for index, item in enumerate(
            queries,
            start=1,
        ):

            query = item["query"]
            tenant_id = item["tenant_id"]
            department = item["department"]

            relevant = item[
                "relevant_documents"
            ]

            dense_results = dense_service.retrieve(
                query=query,
                tenant_id=tenant_id,
                department=department,
                limit=CANDIDATE_LIMIT,
            )

            sparse_results = sparse_service.retrieve(
                query=query,
                tenant_id=tenant_id,
                department=department,
                limit=CANDIDATE_LIMIT,
            )

            hybrid_results = hybrid_service.retrieve(
                query=query,
                tenant_id=tenant_id,
                department=department,
                limit=FINAL_LIMIT,
                candidate_limit=CANDIDATE_LIMIT,
            )

            dense_documents = document_ranking(
                dense_results
            )

            sparse_documents = document_ranking(
                sparse_results
            )

            hybrid_documents = document_ranking(
                hybrid_results
            )

            dense_rank = relevant_rank(
                dense_documents,
                relevant,
            )

            sparse_rank = relevant_rank(
                sparse_documents,
                relevant,
            )

            hybrid_rank = relevant_rank(
                hybrid_documents,
                relevant,
            )

            category = categorize_failure(
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
                hybrid_rank=hybrid_rank,
            )

            report.append(
                {
                    "query_number": index,
                    "query": query,
                    "tenant_id": tenant_id,
                    "department": department,
                    "relevant_documents": relevant,
                    "dense_rank": dense_rank,
                    "sparse_rank": sparse_rank,
                    "hybrid_rank": hybrid_rank,
                    "category": category,
                    "dense_top_5": (
                        dense_documents[:5]
                    ),
                    "sparse_top_5": (
                        sparse_documents[:5]
                    ),
                    "hybrid_top_5": (
                        hybrid_documents[:5]
                    ),
                }
            )

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUTPUT_FILE.write_text(
            json.dumps(
                report,
                indent=2,
            ),
            encoding="utf-8",
        )

        categories = {}

        for item in report:

            category = item["category"]

            categories[category] = (
                categories.get(category, 0)
                + 1
            )

        print()
        print("=" * 60)
        print("RETRIEVAL FAILURE ANALYSIS")
        print("=" * 60)

        print(
            f"Queries analyzed      : {len(report)}"
        )

        print(
            f"Rank 1 successes      : "
            f"{categories.get('success_rank_1', 0)}"
        )

        print(
            f"Low-rank successes    : "
            f"{categories.get('success_low_rank', 0)}"
        )

        print(
            f"Both retrievers miss  : "
            f"{categories.get('both_retrievers_miss', 0)}"
        )

        print(
            f"Dense miss + fusion   : "
            f"{categories.get('dense_miss_fusion_failure', 0)}"
        )

        print(
            f"Sparse miss + fusion  : "
            f"{categories.get('sparse_miss_fusion_failure', 0)}"
        )

        print(
            f"Fusion ranking failure: "
            f"{categories.get('fusion_ranking_failure', 0)}"
        )

        print()
        print("FAILURES / LOW-RANK RESULTS")
        print("-" * 60)

        for item in report:

            if item["category"] == "success_rank_1":
                continue

            print(
                f"Q{item['query_number']}: "
                f"{item['category']}"
            )

            print(
                f"Query  : {item['query']}"
            )

            print(
                f"Dense  : {item['dense_rank']}"
            )

            print(
                f"Sparse : {item['sparse_rank']}"
            )

            print(
                f"Hybrid : {item['hybrid_rank']}"
            )

            print()

        print(
            f"Report saved          : {OUTPUT_FILE}"
        )

    finally:

        vector_store.close()


if __name__ == "__main__":
    main()