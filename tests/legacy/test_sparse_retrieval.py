from pathlib import Path

import pytest

from rag.retrieval.sparse_service import (
    SparseRetrievalService,
)
from rag.sparse_retrieval.bm25_index import (
    BM25Index,
)
from rag.sparse_retrieval.config import (
    SparseRetrievalConfig,
)


@pytest.fixture
def retrieval_service():

    config = SparseRetrievalConfig()

    if not Path(config.index_path).is_file():
        pytest.skip(
            "Sparse index does not exist."
        )

    index = BM25Index.load(config)

    return SparseRetrievalService(index)


def test_real_sparse_retrieval_returns_results(
    retrieval_service,
):

    results = retrieval_service.retrieve(
        query=(
            "refund policy standard customers"
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


def test_real_sparse_retrieval_returns_ranked_results(
    retrieval_service,
):

    results = retrieval_service.retrieve(
        query=(
            "employee benefits enrollment"
        ),
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


def test_real_sparse_retrieval_finds_exact_term_document(
    retrieval_service,
):

    results = retrieval_service.retrieve(
        query=(
            "API endpoint document ingestion"
        ),
        tenant_id="tenant_1",
        department="engineering",
        limit=5,
    )

    assert results

    filenames = [
        result.metadata["source_filename"]
        for result in results
    ]

    assert (
        "api_endpoint_inventory.csv"
        in filenames
    )


def test_real_sparse_retrieval_respects_wrong_scope(
    retrieval_service,
):

    results = retrieval_service.retrieve(
        query="refund policy",
        tenant_id="tenant_that_does_not_exist",
        limit=5,
    )

    assert results == []