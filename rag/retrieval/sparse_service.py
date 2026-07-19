from rag.retrieval.schemas import (
    RetrievalRequest,
)

from rag.sparse_retrieval.bm25_index import (
    BM25Index,
)

from rag.retrieval.schemas import (
    RetrievalResult, 
)

class SparseRetrievalService:

    def __init__(
        self,
        bm25_index: BM25Index,
    ):
        self.bm25_index = bm25_index

    def _append_results(
        self,
        all_results,
        results,
    ):

        for chunk, score in results:

            all_results.append(

                RetrievalResult(

                    chunk_id=chunk.chunk_id,

                    document_id=chunk.document_id,

                    text=chunk.text,

                    score=float(score),

                    metadata={
                        "tenant_id":
                            chunk.tenant_id,

                        "department":
                            chunk.department,

                        "source_filename":
                            chunk.source_filename,

                        "source_file_type":
                            chunk.source_file_type,

                        "chunk_index":
                            chunk.chunk_index,

                        "chunk_type":
                            chunk.chunk_type,

                        "strategy":
                            chunk.strategy,

                        "token_count":
                            chunk.token_count,

                        "page_numbers":
                            chunk.page_numbers,

                        "section_path":
                            chunk.section_path,

                        "table_ids":
                            chunk.table_ids,

                        "metadata":
                            chunk.metadata,
                    },
                )
            )

    def retrieve(
        self,
        request: RetrievalRequest,
    ):

        all_results = []

        # Admin / Knowledge Admin
        # Search all departments
        if "*" in request.scope.departments:

            results = self.bm25_index.search(
                query=request.query,
                tenant_id=(
                    request.scope.tenant_id
                ),
                department=None,
                limit=request.limit,
            )

            self._append_results(
                all_results,
                results,
            )

        # Department-restricted users
        else:

            for department in (
                request.scope.departments
            ):

                results = self.bm25_index.search(
                    query=request.query,
                    tenant_id=(
                        request.scope.tenant_id
                    ),
                    department=department,
                    limit=request.limit,
                )

                self._append_results(
                    all_results,
                    results,
                )

        all_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return all_results[:request.limit]