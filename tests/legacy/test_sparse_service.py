import pytest

from rag.chunking.schemas import Chunk
from rag.retrieval.sparse_service import (
    SparseRetrievalService,
)
from rag.sparse_retrieval.bm25_index import BM25Index
from rag.sparse_retrieval.config import (
    SparseRetrievalConfig,
)


def create_chunk(
    chunk_id: str,
    text: str,
) -> Chunk:

    return Chunk(
        chunk_id=chunk_id,
        document_id="document_1",
        tenant_id="tenant_1",
        department="engineering",
        content_hash="content_hash",
        source_filename="document.txt",
        source_file_type=".txt",
        source_s3_key=(
            "tenant_1/engineering/document.txt"
        ),
        text=text,
        chunk_index=0,
        chunk_type="text",
        strategy="recursive",
        chunking_version="1.0",
        character_count=len(text),
        token_count=len(text.split()),
        page_numbers=[],
        table_ids=[],
        metadata={},
    )


@pytest.fixture
def retrieval_service():

    index = BM25Index(
        SparseRetrievalConfig()
    )

    index.build(
        [
            create_chunk(
                "chunk_1",
                "refund policy standard customers",
            ),
            create_chunk(
                "chunk_2",
                "employee benefits enrollment guide",
            ),
            create_chunk(
                "chunk_3",
                "api endpoint document ingestion",
            ),
        ]
    )

    return SparseRetrievalService(index)


def test_retrieve_returns_results(
    retrieval_service,
):

    results = retrieval_service.retrieve(
        query="refund standard",
        tenant_id="tenant_1",
    )

    assert results

    assert results[0].chunk_id == "chunk_1"

    assert results[0].score != 0


def test_retrieve_requires_query(
    retrieval_service,
):

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        retrieval_service.retrieve(
            query=" ",
            tenant_id="tenant_1",
        )


def test_retrieve_requires_tenant(
    retrieval_service,
):

    with pytest.raises(
        ValueError,
        match="Tenant ID cannot be empty",
    ):
        retrieval_service.retrieve(
            query="refund",
            tenant_id=" ",
        )


def test_retrieve_passes_scope_and_limit():

    class FakeBM25Index:

        def __init__(self):
            self.search_args = None

        def search(self, **kwargs):
            self.search_args = kwargs
            return []

    index = FakeBM25Index()

    service = SparseRetrievalService(index)

    service.retrieve(
        query="refund policy",
        tenant_id="tenant_1",
        department="support",
        limit=10,
    )

    assert index.search_args == {
        "query": "refund policy",
        "tenant_id": "tenant_1",
        "department": "support",
        "limit": 10,
    }