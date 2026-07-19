from rag.evaluation.citation_completeness import (
    CitationCompletenessEvaluator,
)


def main():

    evaluator = (
        CitationCompletenessEvaluator()
    )

    print()
    print("=" * 80)
    print("CITATION COMPLETENESS TEST")
    print("=" * 80)
    print()

    # ============================================================
    # TEST 1
    # Fully cited answer
    # ============================================================

    complete = evaluator.evaluate(
        answer=(
            "The reset option is under "
            "Settings > Security [faq.md]. "
            "A confirmation link is sent "
            "by email [faq.md]."
        ),
        allowed_sources={
            "faq.md",
        },
    )

    assert (
        complete.factual_units
        == 2
    )

    assert (
        complete.cited_units
        == 2
    )

    assert (
        complete.completeness_score
        == 1.0
    )

    assert not complete.uncited_units

    print(
        "Fully cited answer           : PASS"
    )

    # ============================================================
    # TEST 2
    # Partially cited answer
    # ============================================================

    partial = evaluator.evaluate(
        answer=(
            "The reset option is under "
            "Settings > Security [faq.md]. "
            "A confirmation link is sent "
            "by email."
        ),
        allowed_sources={
            "faq.md",
        },
    )

    assert (
        partial.factual_units
        == 2
    )

    assert (
        partial.cited_units
        == 1
    )

    assert (
        partial.completeness_score
        == 0.5
    )

    assert (
        len(partial.uncited_units)
        == 1
    )

    print(
        "Partially cited answer       : PASS"
    )

    # ============================================================
    # TEST 3
    # Invalid citation must not count
    # ============================================================

    invalid = evaluator.evaluate(
        answer=(
            "The reset option is under "
            "Settings > Security "
            "[invented.pdf]."
        ),
        allowed_sources={
            "faq.md",
        },
    )

    assert (
        invalid.factual_units
        == 1
    )

    assert (
        invalid.cited_units
        == 0
    )

    assert (
        invalid.completeness_score
        == 0.0
    )

    assert (
        len(invalid.uncited_units)
        == 1
    )

    print(
        "Invalid citation rejected    : PASS"
    )

    # ============================================================
    # TEST 4
    # Citation-only unit attaches only to previous claim
    # ============================================================

    separate_citation = (
        evaluator.evaluate(
            answer=(
                "The reset option is under "
                "Settings > Security. "
                "A confirmation link is sent "
                "by email. "
                "[faq.md]"
            ),
            allowed_sources={
                "faq.md",
            },
        )
    )

    assert (
        separate_citation.factual_units
        == 2
    )

    assert (
        separate_citation.cited_units
        == 1
    )

    assert (
        separate_citation
        .completeness_score
        == 0.5
    )

    assert (
        len(
            separate_citation
            .uncited_units
        )
        == 1
    )

    print(
        "Separate citation association: PASS"
    )

    # ============================================================
    # TEST 5
    # Introductory colon-ending unit is ignored
    # ============================================================

    introductory = evaluator.evaluate(
        answer=(
            "The limits are as follows:\n"
            "- 6 business days [a.pdf].\n"
            "- 11 business days [b.pdf]."
        ),
        allowed_sources={
            "a.pdf",
            "b.pdf",
        },
    )

    assert (
        introductory.factual_units
        == 2
    )

    assert (
        introductory.cited_units
        == 2
    )

    assert (
        introductory
        .completeness_score
        == 1.0
    )

    assert not (
        introductory.uncited_units
    )

    print(
        "Introductory unit ignored    : PASS"
    )

    # ============================================================
    # TEST 6
    # Multiple valid citations in one factual unit
    # ============================================================

    multiple_valid = evaluator.evaluate(
        answer=(
            "The approval limit is "
            "6 business days "
            "[a.pdf] [b.pdf]."
        ),
        allowed_sources={
            "a.pdf",
            "b.pdf",
        },
    )

    assert (
        multiple_valid.factual_units
        == 1
    )

    assert (
        multiple_valid.cited_units
        == 1
    )

    assert (
        multiple_valid
        .completeness_score
        == 1.0
    )

    assert not (
        multiple_valid.uncited_units
    )

    print(
        "Multiple valid citations     : PASS"
    )

    # ============================================================
    # TEST 7
    # Mixed valid and invalid citations
    # ============================================================

    mixed_citations = evaluator.evaluate(
        answer=(
            "The approval limit is "
            "6 business days "
            "[a.pdf] [invented.pdf]."
        ),
        allowed_sources={
            "a.pdf",
        },
    )

    assert (
        mixed_citations.factual_units
        == 1
    )

    assert (
        mixed_citations.cited_units
        == 1
    )

    assert (
        mixed_citations
        .completeness_score
        == 1.0
    )

    assert not (
        mixed_citations.uncited_units
    )

    print(
        "Mixed citation handling      : PASS"
    )

    # ============================================================
    # TEST 8
    # Only invalid citations
    # ============================================================

    only_invalid = evaluator.evaluate(
        answer=(
            "The approval limit is "
            "6 business days "
            "[invented.pdf] "
            "[fake.docx]."
        ),
        allowed_sources={
            "a.pdf",
        },
    )

    assert (
        only_invalid.factual_units
        == 1
    )

    assert (
        only_invalid.cited_units
        == 0
    )

    assert (
        only_invalid
        .completeness_score
        == 0.0
    )

    assert (
        len(
            only_invalid.uncited_units
        )
        == 1
    )

    print(
        "Invalid-only citations       : PASS"
    )

    # ============================================================
    # TEST 9
    # Empty answer
    # ============================================================

    empty = evaluator.evaluate(
        answer="",
        allowed_sources=set(),
    )

    assert (
        empty.factual_units
        == 0
    )

    assert (
        empty.cited_units
        == 0
    )

    assert (
        empty.completeness_score
        == 1.0
    )

    assert not empty.uncited_units

    print(
        "Empty answer handling        : PASS"
    )

    # ============================================================
    # TEST 10
    # Abstention answer
    # ============================================================

    abstention = evaluator.evaluate(
        answer=(
            "I do not know based on the "
            "provided context."
        ),
        allowed_sources=set(),
    )

    assert (
        abstention.factual_units
        == 0
    )

    assert (
        abstention.cited_units
        == 0
    )

    assert (
        abstention.completeness_score
        == 1.0
    )

    assert not (
        abstention.uncited_units
    )

    print(
        "Abstention handling          : PASS"
    )

    # ============================================================
    # TEST 11
    # Citation-only answer
    #
    # No factual claim exists.
    # Citation should not become a factual unit.
    # ============================================================

    citation_only = evaluator.evaluate(
        answer="[faq.md]",
        allowed_sources={
            "faq.md",
        },
    )

    assert (
        citation_only.factual_units
        == 0
    )

    assert (
        citation_only.cited_units
        == 0
    )

    assert (
        citation_only
        .completeness_score
        == 1.0
    )

    assert not (
        citation_only.uncited_units
    )

    print(
        "Citation-only answer         : PASS"
    )

    print()
    print("=" * 80)
    print("FINAL STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()