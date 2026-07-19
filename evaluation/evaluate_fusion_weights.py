import json
from pathlib import Path

from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.model import EmbeddingModel
from rag.evaluation.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from rag.retrieval.hybrid_service import (
    HybridRetrievalService,
)
from rag.sparse_retrieval.bm25_index import BM25Index
from rag.sparse_retrieval.config import (
    SparseRetrievalConfig,
)
from rag.retrieval.dense_service import (
    DenseRetrievalService,
)
from rag.retrieval.sparse_service import (
    SparseRetrievalService,
)
from rag.vector_store.config import VectorDBConfig
from rag.vector_store.qdrant_store import (
    QdrantVectorStore,
)


EVALUATION_FILE = Path(
    "data/evaluation/retrieval_queries.json"
)


WEIGHT_CONFIGS = [
    (1.0, 1.0),
    (1.0, 1.1),
    (1.0, 1.2),
    (1.0, 1.3),
    (1.0, 1.4),
    (1.0, 1.5),
    (1.1, 1.0),
    (1.2, 1.0),
]


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


def evaluate_weights(
    hybrid_service,
    queries,
    dense_weight,
    sparse_weight,
    limit=5,
):

    recall_1 = []
    recall_3 = []
    recall_5 = []
    reciprocal_ranks = []

    misses = []

    for index, item in enumerate(
        queries,
        start=1,
    ):

        results = hybrid_service.retrieve(
            query=item["query"],
            tenant_id=item["tenant_id"],
            department=item["department"],
            limit=limit,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
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

        if rr == 0:
            misses.append(index)

    query_count = len(queries)

    return {
        "dense_weight": dense_weight,
        "sparse_weight": sparse_weight,
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
        "misses": misses,
    }


def print_summary(results):

    print()
    print("=" * 88)
    print("FUSION WEIGHT EVALUATION")
    print("=" * 88)

    print(
        f"{'Dense':>8}"
        f"{'Sparse':>10}"
        f"{'R@1':>10}"
        f"{'R@3':>10}"
        f"{'R@5':>10}"
        f"{'MRR':>10}"
        f"{'Misses':>20}"
    )

    print("-" * 88)

    for item in results:

        misses = (
            ",".join(
                str(query_number)
                for query_number
                in item["misses"]
            )
            or "-"
        )

        print(
            f"{item['dense_weight']:>8.1f}"
            f"{item['sparse_weight']:>10.1f}"
            f"{item['recall_at_1']:>10.3f}"
            f"{item['recall_at_3']:>10.3f}"
            f"{item['recall_at_5']:>10.3f}"
            f"{item['mrr']:>10.3f}"
            f"{misses:>20}"
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

        results = []

        for (
            dense_weight,
            sparse_weight,
        ) in WEIGHT_CONFIGS:

            print(
                "Evaluating "
                f"dense={dense_weight:.1f}, "
                f"sparse={sparse_weight:.1f}"
            )

            metrics = evaluate_weights(
                hybrid_service=hybrid_service,
                queries=queries,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )

            results.append(metrics)

        print_summary(results)

        best = max(
            results,
            key=lambda item: (
                item["recall_at_5"],
                item["mrr"],
                item["recall_at_1"],
                item["recall_at_3"],
            ),
        )

        print()
        print("=" * 88)
        print("BEST CONFIGURATION")
        print("=" * 88)

        print(
            f"Dense weight  : "
            f"{best['dense_weight']:.1f}"
        )

        print(
            f"Sparse weight : "
            f"{best['sparse_weight']:.1f}"
        )

        print(
            f"R@1          : "
            f"{best['recall_at_1']:.3f}"
        )

        print(
            f"R@3          : "
            f"{best['recall_at_3']:.3f}"
        )

        print(
            f"R@5          : "
            f"{best['recall_at_5']:.3f}"
        )

        print(
            f"MRR          : "
            f"{best['mrr']:.3f}"
        )

        print(
            "Misses       : "
            f"{best['misses']}"
        )

    finally:

        vector_store.close()


if __name__ == "__main__":
    main()