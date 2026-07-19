import pytest

from rag.evaluation.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_finds_relevant_document():

    score = recall_at_k(
        retrieved_documents=[
            "wrong.pdf",
            "correct.pdf",
        ],
        relevant_documents=[
            "correct.pdf",
        ],
        k=2,
    )

    assert score == 1.0


def test_recall_at_k_misses_relevant_document():

    score = recall_at_k(
        retrieved_documents=[
            "wrong.pdf",
            "correct.pdf",
        ],
        relevant_documents=[
            "correct.pdf",
        ],
        k=1,
    )

    assert score == 0.0


def test_recall_at_k_handles_multiple_relevant_documents():

    score = recall_at_k(
        retrieved_documents=[
            "a.pdf",
            "wrong.pdf",
            "b.pdf",
        ],
        relevant_documents=[
            "a.pdf",
            "b.pdf",
        ],
        k=2,
    )

    assert score == 0.5


def test_reciprocal_rank_returns_first_relevant_rank():

    score = reciprocal_rank(
        retrieved_documents=[
            "wrong.pdf",
            "correct.pdf",
            "correct.pdf",
        ],
        relevant_documents=[
            "correct.pdf",
        ],
    )

    assert score == 0.5


def test_reciprocal_rank_returns_zero_for_no_match():

    score = reciprocal_rank(
        retrieved_documents=[
            "wrong.pdf",
        ],
        relevant_documents=[
            "correct.pdf",
        ],
    )

    assert score == 0.0


def test_mean_reciprocal_rank():

    score = mean_reciprocal_rank(
        [1.0, 0.5, 0.0]
    )

    assert score == 0.5


def test_mean_reciprocal_rank_handles_empty_input():

    assert mean_reciprocal_rank([]) == 0.0


def test_recall_at_k_requires_positive_k():

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        recall_at_k(
            retrieved_documents=[],
            relevant_documents=[],
            k=0,
        )