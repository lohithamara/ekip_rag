from pathlib import Path

import pytest

from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.model import EmbeddingModel
from rag.retrieval.dense_service import DenseRetrievalService
from rag.retrieval.hybrid_service import HybridRetrievalService
from rag.retrieval.sparse_service import SparseRetrievalService
from rag.sparse_retrieval.bm25_index import BM25Index
from rag.sparse_retrieval.config import SparseRetrievalConfig
from rag.vector_store.config import VectorDBConfig
from rag.vector_store.qdrant_store import QdrantVectorStore


@pytest.fixture
def retrieval_service():

    sparse_config = SparseRetrievalConfig()

    if not Path(sparse_config.index_path).is_file():
        pytest.skip("Sparse index does not exist.")

    vector_store = QdrantVectorStore(
        VectorDBConfig()
    )

    if not vector_store.collection_exists():
        vector_store.close()
        pytest.skip("Vector collection does not exist.")

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

    service = HybridRetrievalService(
        dense_service=dense_service,
        sparse_service=sparse_service,
    )

    yield service

    vector_store.close()


def test_real_hybrid_retrieval_returns_results(
    retrieval_service,
):

    results = retrieval_service.retrieve(
        query="refund policy standard customers",
        tenant_id="tenant_1",
        department="support",
        limit=5,
        candidate_limit=20,
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


def test_real_hybrid_retrieval_returns_ranked_results(
    retrieval_service,
):

    results = retrieval_service.retrieve(
        query="employee benefits enrollment",
        tenant_id="tenant_1",
        department="hr",
        limit=5,
        candidate_limit=20,
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


def test_real_hybrid_retrieval_finds_exact_term_document(
    retrieval_service,
):

    results = retrieval_service.retrieve(
        query="API endpoint document ingestion",
        tenant_id="tenant_1",
        department="engineering",
        limit=5,
        candidate_limit=20,
    )

    assert results

    filenames = [
        result.metadata.get("source_filename")
        for result in results
    ]

    assert "api_endpoint_inventory.csv" in filenames


def test_real_hybrid_retrieval_respects_wrong_scope(
    retrieval_service,
):

    results = retrieval_service.retrieve(
        query="refund policy",
        tenant_id="tenant_that_does_not_exist",
        limit=5,
        candidate_limit=20,
    )

    assert results == []