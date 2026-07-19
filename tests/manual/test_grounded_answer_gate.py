from rag.generation.grounded_answer_gate import (
    GroundedAnswerGate,
)


def main():

    gate = GroundedAnswerGate()

    # ------------------------------------------------------------
    # Valid grounded answer
    # ------------------------------------------------------------

    answer = gate.apply(
        answer=(
            "Employees receive 20 days of "
            "annual leave [leave_policy.pdf]."
        ),
        validated_sources=(
            "leave_policy.pdf",
        ),
    )

    assert (
        answer
        == (
            "Employees receive 20 days of "
            "annual leave [leave_policy.pdf]."
        )
    )

    print(
        "Grounded answer allowed       : PASS"
    )

    # ------------------------------------------------------------
    # Unsupported factual answer
    # ------------------------------------------------------------

    answer = gate.apply(
        answer=(
            "Employees receive 50 days "
            "of annual leave."
        ),
        validated_sources=(),
    )

    assert (
        answer
        == gate.ABSTENTION_TEXT
    )

    print(
        "Ungrounded answer blocked     : PASS"
    )

    # ------------------------------------------------------------
    # Correct abstention
    # ------------------------------------------------------------

    answer = gate.apply(
        answer=gate.ABSTENTION_TEXT,
        validated_sources=(),
    )

    assert (
        answer
        == gate.ABSTENTION_TEXT
    )

    print(
        "Abstention preserved          : PASS"
    )

    # ------------------------------------------------------------
    # Empty answer
    # ------------------------------------------------------------

    answer = gate.apply(
        answer="   ",
        validated_sources=(),
    )

    assert (
        answer
        == gate.ABSTENTION_TEXT
    )

    print(
        "Empty answer blocked          : PASS"
    )

    # ------------------------------------------------------------
    # Invalid citation already removed by validator
    # ------------------------------------------------------------

    answer = gate.apply(
        answer=(
            "The CEO prefers Python "
            "[fake_source.pdf]."
        ),
        validated_sources=(),
    )

    assert (
        answer
        == gate.ABSTENTION_TEXT
    )

    print(
        "Invalid citation blocked      : PASS"
    )

    print()
    print("=" * 80)
    print("GROUNDED ANSWER GATE TEST")
    print("=" * 80)
    print("FINAL STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()