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

QUERY_NUMBERS = {
    1,
    2,
    9,
    16,
    18,
    20,
}


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


def find_rank(
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


def print_ranking(
    name,
    documents,
    relevant_documents,
):

    rank = find_rank(
        documents,
        relevant_documents,
    )

    print()
    print(f"{name} | relevant rank: {rank}")
    print("-" * 60)

    for index, document in enumerate(
        documents,
        start=1,
    ):

        marker = (
            "*"
            if document in relevant_documents
            else " "
        )

        print(
            f"{marker} {index}. {document}"
        )


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

        for query_number in sorted(
            QUERY_NUMBERS
        ):

            item = queries[
                query_number - 1
            ]

            query = item["query"]
            tenant_id = item["tenant_id"]
            department = item["department"]

            relevant = item[
                "relevant_documents"
            ]

            dense_results = (
                dense_service.retrieve(
                    query=query,
                    tenant_id=tenant_id,
                    department=department,
                    limit=20,
                )
            )

            sparse_results = (
                sparse_service.retrieve(
                    query=query,
                    tenant_id=tenant_id,
                    department=department,
                    limit=20,
                )
            )

            hybrid_results = (
                hybrid_service.retrieve(
                    query=query,
                    tenant_id=tenant_id,
                    department=department,
                    limit=10,
                    candidate_limit=20,
                )
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

            overlap = (
                set(dense_documents)
                & set(sparse_documents)
            )

            print()
            print("=" * 70)
            print(f"QUERY {query_number}: {query}")
            print(f"RELEVANT: {relevant}")
            print("=" * 70)

            print_ranking(
                "DENSE",
                dense_documents,
                relevant,
            )

            print_ranking(
                "SPARSE",
                sparse_documents,
                relevant,
            )

            print_ranking(
                "HYBRID",
                hybrid_documents,
                relevant,
            )

            print()
            print(
                "Dense/Sparse document overlap:",
                len(overlap),
            )

    finally:

        vector_store.close()


if __name__ == "__main__":
    main()