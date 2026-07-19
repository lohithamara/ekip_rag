from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.model import EmbeddingModel
from rag.vector_store.config import VectorDBConfig
from rag.vector_store.qdrant_store import (
    QdrantVectorStore,
)


def main():

    queries = [
        "What is annual recurring revenue?",
        "How do employees enroll in benefits?",
        "What is the refund policy for standard customers?",
        "Which API endpoint handles document ingestion?",
    ]

    model = EmbeddingModel(
        EmbeddingConfig()
    )

    store = QdrantVectorStore(
        VectorDBConfig()
    )

    try:

        for query in queries:

            query_vector = model.encode(
                [query]
            )[0]

            results = store.search(
                query_vector=query_vector,
                tenant_id="1",
                department="finance",
                limit=5,
            )

            print()
            print("===================================")
            print("VECTOR SEARCH RESULTS")
            print("===================================")
            print(f"Query: {query}")
            print()

            for rank, result in enumerate(
                results,
                start=1,
            ):

                payload = result.payload or {}

                print(f"Rank       : {rank}")
                print(
                    f"Score      : "
                    f"{result.score:.4f}"
                )
                print(
                    f"Document   : "
                    f"{payload.get('source_filename')}"
                )
                print(
                    f"Department : "
                    f"{payload.get('department')}"
                )
                print(
                    f"Text       : "
                    f"{payload.get('text', '')[:300]}"
                )
                print("-----------------------------------")

    finally:

        store.close()
if __name__ == "__main__":
    main()