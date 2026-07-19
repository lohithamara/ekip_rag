import pytest

from rag.retrieval.hybrid_service import (
    HybridRetrievalService,
)
from rag.retrieval.schemas import RetrievalResult


def create_result(
    chunk_id: str,
    document_id: str | None = None,
) -> RetrievalResult:

    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=(
            document_id
            or f"document_{chunk_id}"
        ),
        text=f"Text for {chunk_id}",
        score=0.5,
        tenant_id="tenant_1",
        department="engineering",
        metadata={},
    )


class FakeRetrievalService:

    def __init__(self, results):

        self.results = results
        self.retrieve_args = None

    def retrieve(self, **kwargs):

        self.retrieve_args = kwargs

        return self.results


@pytest.fixture
def hybrid_service():

    dense_service = FakeRetrievalService(
        [
            create_result("a"),
            create_result("b"),
        ]
    )

    sparse_service = FakeRetrievalService(
        [
            create_result("b"),
            create_result("c"),
        ]
    )

    return HybridRetrievalService(
        dense_service=dense_service,
        sparse_service=sparse_service,
    )


def test_retrieve_returns_fused_results(
    hybrid_service,
):

    results = hybrid_service.retrieve(
        query="enterprise retrieval",
        tenant_id="tenant_1",
    )

    assert len(results) == 3
    assert results[0].chunk_id == "b"


def test_retrieve_passes_scope_to_services(
    hybrid_service,
):

    hybrid_service.retrieve(
        query="refund policy",
        tenant_id="tenant_1",
        department="support",
        limit=5,
        candidate_limit=10,
    )

    dense_args = (
        hybrid_service
        .dense_service
        .retrieve_args
    )

    sparse_args = (
        hybrid_service
        .sparse_service
        .retrieve_args
    )

    assert dense_args == {
        "query": "refund policy",
        "tenant_id": "tenant_1",
        "department": "support",
        "limit": 10,
    }

    assert sparse_args == dense_args


def test_retrieve_respects_final_limit(
    hybrid_service,
):

    results = hybrid_service.retrieve(
        query="retrieval",
        tenant_id="tenant_1",
        limit=2,
    )

    assert len(results) == 2


def test_retrieve_requires_query(
    hybrid_service,
):

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        hybrid_service.retrieve(
            query=" ",
            tenant_id="tenant_1",
        )


def test_retrieve_requires_tenant(
    hybrid_service,
):

    with pytest.raises(
        ValueError,
        match="Tenant ID cannot be empty",
    ):
        hybrid_service.retrieve(
            query="retrieval",
            tenant_id=" ",
        )


def test_retrieve_requires_positive_limit(
    hybrid_service,
):

    with pytest.raises(
        ValueError,
        match="Limit must be greater than zero",
    ):
        hybrid_service.retrieve(
            query="retrieval",
            tenant_id="tenant_1",
            limit=0,
        )


def test_retrieve_requires_enough_candidates(
    hybrid_service,
):

    with pytest.raises(
        ValueError,
        match="Candidate limit cannot be smaller",
    ):
        hybrid_service.retrieve(
            query="retrieval",
            tenant_id="tenant_1",
            limit=10,
            candidate_limit=5,
        )