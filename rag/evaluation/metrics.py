def recall_at_k(
    retrieved_documents: list[str],
    relevant_documents: list[str],
    k: int,
) -> float:

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    if not relevant_documents:
        return 0.0

    retrieved = set(
        retrieved_documents[:k]
    )

    relevant = set(
        relevant_documents
    )

    return len(
        retrieved & relevant
    ) / len(relevant)


def reciprocal_rank(
    retrieved_documents: list[str],
    relevant_documents: list[str],
) -> float:

    relevant = set(relevant_documents)

    for rank, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        if document in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    reciprocal_ranks: list[float],
) -> float:

    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(
        reciprocal_ranks
    )

def normalized_text_contains(
    text: str,
    expected_text: str,
) -> bool:

    normalized_text = " ".join(
        text.lower().split()
    )

    normalized_expected = " ".join(
        expected_text.lower().split()
    )

    return (
        normalized_expected
        in normalized_text
    )

def source_recall(
    validated_sources: list[str]
    | tuple[str, ...],
    expected_sources: list[str]
    | tuple[str, ...],
) -> float:

    if not expected_sources:
        return 1.0

    validated = set(
        validated_sources
    )

    expected = set(
        expected_sources
    )

    return len(
        validated & expected
    ) / len(expected)


def is_abstention(
    answer: str,
) -> bool:

    normalized = " ".join(
        answer.lower().split()
    )

    return (
        "i do not know based on the "
        "provided context"
        in normalized
    )


def abstention_correct(
    answer: str,
    expected_behavior: str,
) -> bool:

    abstained = is_abstention(answer)

    if expected_behavior == "unknown":
        return abstained

    if expected_behavior == "answer":
        return not abstained

    raise ValueError(
        "expected_behavior must be "
        "'answer' or 'unknown'."
    )

def fact_group_matches(
    answer: str,
    fact_group: tuple[str, ...],
) -> bool:

    if not fact_group:
        return False

    return any(
        normalized_text_contains(
            answer,
            phrase,
        )
        for phrase in fact_group
    )


def fact_recall(
    answer: str,
    expected_facts: tuple[
        tuple[str, ...],
        ...,
    ],
) -> float:

    if not expected_facts:
        return 1.0

    matched = sum(
        fact_group_matches(
            answer,
            fact_group,
        )
        for fact_group in expected_facts
    )

    return matched / len(
        expected_facts
    )


def matched_facts(
    answer: str,
    expected_facts: tuple[
        tuple[str, ...],
        ...,
    ],
) -> tuple[
    tuple[str, ...],
    ...,
]:

    return tuple(
        fact_group
        for fact_group in expected_facts
        if fact_group_matches(
            answer,
            fact_group,
        )
    )


def missing_facts(
    answer: str,
    expected_facts: tuple[
        tuple[str, ...],
        ...,
    ],
) -> tuple[
    tuple[str, ...],
    ...,
]:

    return tuple(
        fact_group
        for fact_group in expected_facts
        if not fact_group_matches(
            answer,
            fact_group,
        )
    )