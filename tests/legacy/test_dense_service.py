from types import SimpleNamespace

import pytest

from rag.retrieval.dense_service import (
    DenseRetrievalService,
)


class FakeEmbeddingModel:

    def encode(self, texts):

        return [
            (0.1,) * 384
            for _ in texts
        ]


class FakeVectorStore:

    def __init__(self):

        self.search_args = None

    def search(self, **kwargs):

        self.search_args = kwargs

        return [
            SimpleNamespace(
                score=0.85,
                payload={
                    "chunk_id": "chunk_1",
                    "document_id": "document_1",
                    "tenant_id": "tenant_1",
                    "department": "engineering",
                    "text": "Enterprise retrieval system",
                },
            )
        ]


@pytest.fixture
def retrieval_service():

    return DenseRetrievalService(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(),
    )


def test_retrieve_returns_results(
    retrieval_service,
):

    results = retrieval_service.retrieve(
        query="What is enterprise retrieval?",
        tenant_id="tenant_1",
    )

    assert len(results) == 1

    assert results[0].chunk_id == "chunk_1"

    assert results[0].score == 0.85


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
            query="retrieval",
            tenant_id=" ",
        )


def test_retrieve_passes_scope_to_vector_store(
    retrieval_service,
):

    retrieval_service.retrieve(
        query="retrieval",
        tenant_id="tenant_1",
        department="engineering",
        limit=10,
        score_threshold=0.3,
    )

    args = (
        retrieval_service
        .vector_store
        .search_args
    )

    assert args["tenant_id"] == "tenant_1"
    assert args["department"] == "engineering"
    assert args["limit"] == 10
    assert args["score_threshold"] == 0.3