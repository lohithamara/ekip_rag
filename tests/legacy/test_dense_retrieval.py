import pytest

from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.model import EmbeddingModel
from rag.retrieval.dense_service import (
    DenseRetrievalService,
)
from rag.vector_store.config import VectorDBConfig
from rag.vector_store.qdrant_store import (
    QdrantVectorStore,
)


@pytest.fixture
def retrieval_service():

    store = QdrantVectorStore(
        VectorDBConfig()
    )

    if not store.collection_exists():
        store.close()

        pytest.skip(
            "Vector collection does not exist."
        )

    model = EmbeddingModel(
        EmbeddingConfig()
    )

    service = DenseRetrievalService(
        embedding_model=model,
        vector_store=store,
    )

    yield service

    store.close()


def test_real_dense_retrieval_returns_results(
    retrieval_service,
):

    results = retrieval_service.retrieve(
        query=(
            "What is the refund policy "
            "for standard customers?"
        ),
        tenant_id="tenant_1",
        department="support",
        limit=5,
    )

    assert results

    assert len(results) <= 5

    assert all(
        result.tenant_id == "tenant_1"
        for result in results
    )

    assert all(
        result.department == "support"
        for result in results
    )

    assert all(
        result.text.strip()
        for result in results
    )


def test_real_dense_retrieval_is_ranked(
    retrieval_service,
):

    results = retrieval_service.retrieve(
        query="How do employees enroll in benefits?",
        tenant_id="tenant_1",
        department="hr",
        limit=5,
    )

    assert results

    scores = [
        result.score
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_real_dense_retrieval_respects_wrong_scope(
    retrieval_service,
):

    results = retrieval_service.retrieve(
        query="What is the refund policy?",
        tenant_id="tenant_that_does_not_exist",
        limit=5,
    )

    assert results == []