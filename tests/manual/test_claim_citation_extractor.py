from rag.evaluation.citation_faithfulness import (
    ClaimCitationExtractor,
)


def main():

    extractor = ClaimCitationExtractor()

    print()
    print("=" * 80)
    print("CLAIM CITATION EXTRACTOR TEST")
    print("=" * 80)
    print()

    single = extractor.extract(
        answer=(
            "Password reset is available "
            "under Settings [faq.md]."
        ),
        allowed_sources={
            "faq.md",
        },
    )

    assert len(single) == 1

    assert single[0].claim == (
        "Password reset is available "
        "under Settings ."
    )

    assert single[0].citations == (
        "faq.md",
    )

    print(
        "Single claim extraction      : PASS"
    )

    multiple = extractor.extract(
        answer=(
            "The limit is 6 days "
            "[a.pdf] [b.pdf]."
        ),
        allowed_sources={
            "a.pdf",
            "b.pdf",
        },
    )

    assert len(multiple) == 1

    assert multiple[0].citations == (
        "a.pdf",
        "b.pdf",
    )

    print(
        "Multiple citations           : PASS"
    )

    invalid = extractor.extract(
        answer=(
            "The limit is 6 days "
            "[invented.pdf]."
        ),
        allowed_sources={
            "a.pdf",
        },
    )

    assert not invalid

    print(
        "Invalid citation rejected    : PASS"
    )

    separate = extractor.extract(
        answer=(
            "The limit is 6 days. "
            "[a.pdf]"
        ),
        allowed_sources={
            "a.pdf",
        },
    )

    # The claim itself has no inline citation.
    #
    # Our extractor intentionally does not
    # create uncited claims. A citation-only
    # unit cannot create a new pair when no
    # previous cited pair exists.

    assert not separate

    print(
        "Orphan citation ignored      : PASS"
    )

    introductory = extractor.extract(
        answer=(
            "The limits are as follows:\n"
            "- 6 days [a.pdf].\n"
            "- 11 days [b.pdf]."
        ),
        allowed_sources={
            "a.pdf",
            "b.pdf",
        },
    )

    assert len(introductory) == 2

    assert introductory[0].claim == (
        "6 days ."
    )

    assert introductory[1].claim == (
        "11 days ."
    )

    print(
        "Introductory unit ignored    : PASS"
    )

    mixed = extractor.extract(
        answer=(
            "The limit is 6 days "
            "[a.pdf] [invented.pdf]."
        ),
        allowed_sources={
            "a.pdf",
        },
    )

    assert len(mixed) == 1

    assert mixed[0].citations == (
        "a.pdf",
    )

    print(
        "Mixed citations handled      : PASS"
    )

    abstention = extractor.extract(
        answer=(
            "I do not know based on the "
            "provided context."
        ),
        allowed_sources=set(),
    )

    assert not abstention

    print(
        "Abstention handling          : PASS"
    )

    print()
    print("=" * 80)
    print("FINAL STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()