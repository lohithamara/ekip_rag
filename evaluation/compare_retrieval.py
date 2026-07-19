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


QUERIES = [
    (
        "What is the refund policy for standard customers?",
        "support",
    ),
    (
        "How do employees enroll in benefits?",
        "hr",
    ),
    (
        "Which API endpoint handles document ingestion?",
        "engineering",
    ),
]


def print_results(name, results):

    print()
    print(name)
    print("-" * 60)

    if not results:
        print("No results.")
        return

    for rank, result in enumerate(results, start=1):

        print(
            f"{rank}. "
            f"{result.metadata.get('source_filename')} "
            f"| score={result.score:.4f}"
        )

        print(
            result.text[:180].replace("\n", " ")
        )

        print()


def main():

    sparse_config = SparseRetrievalConfig()

    if not Path(sparse_config.index_path).is_file():
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
            BM25Index.load(sparse_config)
        )

        hybrid_service = HybridRetrievalService(
            dense_service=dense_service,
            sparse_service=sparse_service,
        )

        for query, department in QUERIES:

            print()
            print("=" * 60)
            print(f"QUERY: {query}")
            print(f"DEPARTMENT: {department}")
            print("=" * 60)

            dense_results = dense_service.retrieve(
                query=query,
                tenant_id="tenant_1",
                department=department,
                limit=5,
            )

            sparse_results = sparse_service.retrieve(
                query=query,
                tenant_id="tenant_1",
                department=department,
                limit=5,
            )

            hybrid_results = hybrid_service.retrieve(
                query=query,
                tenant_id="tenant_1",
                department=department,
                limit=5,
                candidate_limit=20,
            )

            print_results(
                "DENSE RESULTS",
                dense_results,
            )

            print_results(
                "SPARSE RESULTS",
                sparse_results,
            )

            print_results(
                "HYBRID RESULTS",
                hybrid_results,
            )

    finally:

        vector_store.close()


if __name__ == "__main__":
    main()