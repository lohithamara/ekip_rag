import pytest

from rag.retrieval.rrf import reciprocal_rank_fusion
from rag.retrieval.schemas import RetrievalResult


def create_result(
    chunk_id: str,
    score: float = 0.5,
) -> RetrievalResult:

    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="document_1",
        text=f"Text for {chunk_id}",
        score=score,
        tenant_id="tenant_1",
        department="engineering",
        metadata={},
    )


def test_rrf_combines_result_lists():

    dense = [
        create_result("a"),
        create_result("b"),
    ]

    sparse = [
        create_result("b"),
        create_result("c"),
    ]

    results = reciprocal_rank_fusion(
        [dense, sparse]
    )

    assert len(results) == 3

    assert results[0].chunk_id == "b"


def test_rrf_removes_duplicate_chunks():

    dense = [
        create_result("a"),
    ]

    sparse = [
        create_result("a"),
    ]

    results = reciprocal_rank_fusion(
        [dense, sparse]
    )

    assert len(results) == 1


def test_rrf_respects_limit():

    results = reciprocal_rank_fusion(
        [
            [
                create_result("a"),
                create_result("b"),
                create_result("c"),
            ]
        ],
        limit=2,
    )

    assert len(results) == 2


def test_rrf_returns_empty_for_empty_lists():

    results = reciprocal_rank_fusion(
        []
    )

    assert results == []


def test_rrf_requires_positive_limit():

    with pytest.raises(
        ValueError,
        match="Limit must be greater than zero",
    ):
        reciprocal_rank_fusion(
            [],
            limit=0,
        )


def test_rrf_requires_positive_k():

    with pytest.raises(
        ValueError,
        match="RRF k must be greater than zero",
    ):
        reciprocal_rank_fusion(
            [],
            k=0,
        )


def test_rrf_supports_weights():

    dense = [
        create_result("a"),
    ]

    sparse = [
        create_result("b"),
    ]

    results = reciprocal_rank_fusion(
        [dense, sparse],
        weights=[1.0, 2.0],
    )

    assert results[0].chunk_id == "b"


def test_rrf_requires_matching_weights():

    with pytest.raises(
        ValueError,
        match="Weights must match",
    ):
        reciprocal_rank_fusion(
            [
                [create_result("a")],
                [create_result("b")],
            ],
            weights=[1.0],
        )