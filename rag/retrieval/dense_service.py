from rag.embeddings.model import EmbeddingModel
from rag.retrieval.schemas import RetrievalRequest, RetrievalResult
from rag.vector_store.qdrant_store import (
    QdrantVectorStore,
)


class DenseRetrievalService:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: QdrantVectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        request: RetrievalRequest,
    ):

        embedding = (
            self.embedding_model.encode(
                [request.query]
            )[0]
        )

        all_results = []

        # print("\nDENSE RETRIEVAL")
        # print("Tenant :", request.scope.tenant_id)
        # print("Departments :", request.scope.departments)

        # ------------------------------------------
        # Admin / Knowledge Admin -> Search all departments
        # ------------------------------------------
        if "*" in request.scope.departments:

            # print("\nSearching ALL departments")

            results = self.vector_store.search(
                query_vector=embedding,
                tenant_id=request.scope.tenant_id,
                department=None,
                limit=request.limit,
            )

            # for point in results[:3]:
            #     print("=" * 80)
            #     print(point.payload.keys())
            #     print(point.payload)

            # print("Returned:", len(results))

            for point in results:

                all_results.append(
                    RetrievalResult(
                        chunk_id=point.payload["chunk_id"],
                        document_id=point.payload["document_id"],
                        text=point.payload["text"],
                        score=float(point.score),
                        metadata=dict(point.payload),
                    )
                )

        # ------------------------------------------
        # Department-wise search
        # ------------------------------------------
        else:

            for department in request.scope.departments:

                # print(f"\nSearching department: {department}")

                results = self.vector_store.search(
                    query_vector=embedding,
                    tenant_id=request.scope.tenant_id,
                    department=department,
                    limit=request.limit,
                )
                # for point in results[:3]:
                #     print("=" * 80)
                #     print(point.payload.keys())
                #     print(point.payload)

                for point in results:

                    all_results.append(
                        RetrievalResult(
                            chunk_id=point.payload["chunk_id"],
                            document_id=point.payload["document_id"],
                            text=point.payload["text"],
                            score=float(point.score),
                            metadata=dict(point.payload),
                        )
                    )

        all_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return all_results