from rag.evaluation.metrics import (
    abstention_correct,
    fact_recall,
    is_abstention,
    normalized_text_contains,
    source_recall,
)


def main():

    print()
    print("=" * 80)
    print("RAG QUALITY METRICS TEST")
    print("=" * 80)
    print()

    assert normalized_text_contains(
        "Go to Settings > Security.",
        "settings > security",
    )

    print(
        "Normalized containment       : PASS"
    )

    assert fact_recall(
        answer=(
            "Go to Settings > Security "
            "and select Reset Password."
        ),
        expected_facts=(
            "Settings > Security",
            "Reset Password",
        ),
    ) == 1.0

    print(
        "Fact recall                  : PASS"
    )

    assert source_recall(
        validated_sources=(
            "a.pdf",
            "b.pdf",
        ),
        expected_sources=(
            "a.pdf",
            "c.pdf",
        ),
    ) == 0.5

    print(
        "Source recall                : PASS"
    )

    assert is_abstention(
        "I do not know based on the "
        "provided context."
    )

    print(
        "Abstention detection         : PASS"
    )

    assert abstention_correct(
        answer=(
            "I do not know based on the "
            "provided context."
        ),
        expected_behavior="unknown",
    )

    assert abstention_correct(
        answer="The answer is 42.",
        expected_behavior="answer",
    )

    print(
        "Abstention correctness       : PASS"
    )

    print()
    print("=" * 80)
    print("FINAL STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()