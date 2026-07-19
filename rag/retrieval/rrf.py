from dataclasses import replace

from rag.retrieval.schemas import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    limit: int = 5,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievalResult]:

    if limit <= 0:
        raise ValueError(
            "Limit must be greater than zero."
        )

    if k <= 0:
        raise ValueError(
            "RRF k must be greater than zero."
        )

    if weights is None:
        weights = [1.0] * len(result_lists)

    if len(weights) != len(result_lists):
        raise ValueError(
            "Weights must match result lists."
        )

    results = {}
    scores = {}

    for result_list, weight in zip(
        result_lists,
        weights,
    ):

        for rank, result in enumerate(
            result_list,
            start=1,
        ):

            results[result.chunk_id] = result

            scores[result.chunk_id] = (
                scores.get(
                    result.chunk_id,
                    0.0,
                )
                + weight / (k + rank)
            )

    ranked = sorted(
        results.values(),
        key=lambda result: scores[
            result.chunk_id
        ],
        reverse=True,
    )

    return [
        replace(
            result,
            score=scores[result.chunk_id],
        )
        for result in ranked[:limit]
    ]