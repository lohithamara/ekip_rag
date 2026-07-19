from rag.citations.validator import (
    CitationValidator,
)


def main():

    validator = CitationValidator()

    answer = (
        "The limit is 6 days "
        "[policy.docx]. "
        "Another source says 10 days "
        "[controls.pdf]. "
        "This citation is invalid "
        "[fake_source.txt]. "
        "The first policy is cited again "
        "[policy.docx]."
    )

    allowed_sources = {
        "policy.docx",
        "controls.pdf",
    }

    sources = validator.validate(
        answer=answer,
        allowed_sources=allowed_sources,
    )

    print("VALIDATED SOURCES")

    for source in sources:
        print(f"- {source}")

    assert sources == [
        "policy.docx",
        "controls.pdf",
    ]

    print()
    print("FINAL STATUS: PASS")


if __name__ == "__main__":
    main()