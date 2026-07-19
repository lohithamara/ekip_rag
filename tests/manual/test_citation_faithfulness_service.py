from rag.evaluation.citation_faithfulness_service import (
    CitationFaithfulnessEvaluationService,
)

from rag.evaluation.citation_judge import (
    CitationFaithfulnessJudge,
)

from rag.generation.schemas import (
    ContextChunk,
)

from rag.llm.schemas import (
    LLMResponse,
)


class FakeLLMService:

    def __init__(
        self,
        responses,
    ):
        self.responses = list(
            responses
        )

    def generate(
        self,
        request,
    ):

        if not self.responses:
            raise RuntimeError(
                "No fake responses remain."
            )

        answer = self.responses.pop(0)

        return LLMResponse(
            answer=answer,
            model_name="fake-judge",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            finish_reason="stop",
            metadata={},
        )


def make_chunk(
    chunk_id: str,
    source_filename: str,
    text: str,
) -> ContextChunk:

    return ContextChunk(
        chunk_id=chunk_id,
        document_id=(
            f"document_{chunk_id}"
        ),
        source_filename=(
            source_filename
        ),
        text=text,
        token_count=len(
            text.split()
        ),
        metadata={},
    )


def main():

    print()
    print("=" * 80)
    print(
        "CITATION FAITHFULNESS "
        "SERVICE TEST"
    )
    print("=" * 80)
    print()

    # ============================================================
    # TEST 1
    # Fully supported answer
    # ============================================================

    judge = CitationFaithfulnessJudge(
        llm_service=FakeLLMService(
            responses=[
                (
                    '{"verdict":"SUPPORTED",'
                    '"reason":"Direct support."}'
                ),
                (
                    '{"verdict":"SUPPORTED",'
                    '"reason":"Direct support."}'
                ),
            ]
        )
    )

    service = (
        CitationFaithfulnessEvaluationService(
            judge=judge
        )
    )

    result = service.evaluate(
        answer=(
            "The limit is 6 days [a.pdf]. "
            "Manager approval is required "
            "[b.pdf]."
        ),
        allowed_sources={
            "a.pdf",
            "b.pdf",
        },
        context_chunks=(
            make_chunk(
                "1",
                "a.pdf",
                "The limit is 6 days.",
            ),
            make_chunk(
                "2",
                "b.pdf",
                "Manager approval is required.",
            ),
        ),
    )

    assert result.total_claims == 2

    assert (
        result.supported_claims
        == 2
    )

    assert (
        result.partially_supported_claims
        == 0
    )

    assert (
        result.unsupported_claims
        == 0
    )

    assert (
        result.faithfulness_score
        == 1.0
    )

    assert (
        result.prompt_tokens
        == 200
    )

    assert (
        result.completion_tokens
        == 40
    )

    assert (
        result.total_tokens
        == 240
    )

    print(
        "Fully supported answer       : PASS"
    )

    # ============================================================
    # TEST 2
    # Mixed claim verdicts
    # ============================================================

    judge = CitationFaithfulnessJudge(
        llm_service=FakeLLMService(
            responses=[
                (
                    '{"verdict":"SUPPORTED",'
                    '"reason":"Direct support."}'
                ),
                (
                    '{"verdict":'
                    '"PARTIALLY_SUPPORTED",'
                    '"reason":"Partial support."}'
                ),
                (
                    '{"verdict":"UNSUPPORTED",'
                    '"reason":"No support."}'
                ),
            ]
        )
    )

    service = (
        CitationFaithfulnessEvaluationService(
            judge=judge
        )
    )

    result = service.evaluate(
        answer=(
            "Claim A [a.pdf]. "
            "Claim B [b.pdf]. "
            "Claim C [c.pdf]."
        ),
        allowed_sources={
            "a.pdf",
            "b.pdf",
            "c.pdf",
        },
        context_chunks=(
            make_chunk(
                "1",
                "a.pdf",
                "Evidence A.",
            ),
            make_chunk(
                "2",
                "b.pdf",
                "Evidence B.",
            ),
            make_chunk(
                "3",
                "c.pdf",
                "Evidence C.",
            ),
        ),
    )

    assert result.total_claims == 3

    assert (
        result.supported_claims
        == 1
    )

    assert (
        result.partially_supported_claims
        == 1
    )

    assert (
        result.unsupported_claims
        == 1
    )

    expected_score = (
        1.0
        + 0.5
        + 0.0
    ) / 3

    assert (
        result.faithfulness_score
        == expected_score
    )

    assert (
        result.total_tokens
        == 360
    )

    print(
        "Mixed verdict aggregation    : PASS"
    )

    # ============================================================
    # TEST 3
    # Abstention
    # ============================================================

    judge = CitationFaithfulnessJudge(
        llm_service=FakeLLMService(
            responses=[]
        )
    )

    service = (
        CitationFaithfulnessEvaluationService(
            judge=judge
        )
    )

    result = service.evaluate(
        answer=(
            "I do not know based on the "
            "provided context."
        ),
        allowed_sources=set(),
        context_chunks=(),
    )

    assert result.total_claims == 0

    assert (
        result.supported_claims
        == 0
    )

    assert (
        result.partially_supported_claims
        == 0
    )

    assert (
        result.unsupported_claims
        == 0
    )

    assert (
        result.faithfulness_score
        == 1.0
    )

    assert result.total_tokens == 0

    print(
        "Abstention handling          : PASS"
    )

    # ============================================================
    # TEST 4
    # Citation source missing from context
    # ============================================================

    judge = CitationFaithfulnessJudge(
        llm_service=FakeLLMService(
            responses=[]
        )
    )

    service = (
        CitationFaithfulnessEvaluationService(
            judge=judge
        )
    )

    result = service.evaluate(
        answer=(
            "The limit is 6 days "
            "[missing.pdf]."
        ),
        allowed_sources={
            "missing.pdf",
        },
        context_chunks=(),
    )

    assert result.total_claims == 1

    assert (
        result.supported_claims
        == 0
    )

    assert (
        result.partially_supported_claims
        == 0
    )

    assert (
        result.unsupported_claims
        == 1
    )

    assert (
        result.faithfulness_score
        == 0.0
    )

    assert result.total_tokens == 0

    assert (
        result.claim_results[0]
        .metadata["missing_sources"]
        == (
            "missing.pdf",
        )
    )
    # ============================================================
    # TEST 5
    # Finance list items inherit shared context
    # ============================================================

    judge = CitationFaithfulnessJudge(
        llm_service=FakeLLMService(
            responses=[
                (
                    '{"verdict":"SUPPORTED",'
                    '"reason":"Complete claim supported."}'
                ),
                (
                    '{"verdict":"SUPPORTED",'
                    '"reason":"Complete claim supported."}'
                ),
                (
                    '{"verdict":"SUPPORTED",'
                    '"reason":"Complete claim supported."}'
                ),
            ]
        )
    )

    service = (
        CitationFaithfulnessEvaluationService(
            judge=judge
        )
    )

    result = service.evaluate(
        answer=(
            "Escalation to the Finance Business "
            "Partner is required for requests "
            "exceeding a certain number of "
            "consecutive business days, but the "
            "exact number of days is conflicting:\n"
            "- 6 consecutive business days "
            "[a.pdf],\n"
            "- 11 consecutive business days "
            "[b.pdf],\n"
            "- 15 consecutive business days "
            "[c.pdf]."
        ),
        allowed_sources={
            "a.pdf",
            "b.pdf",
            "c.pdf",
        },
        context_chunks=(
            make_chunk(
                "1",
                "a.pdf",
                (
                    "Escalation to the Finance "
                    "Business Partner is required "
                    "after 6 consecutive business "
                    "days."
                ),
            ),
            make_chunk(
                "2",
                "b.pdf",
                (
                    "Escalation to the Finance "
                    "Business Partner is required "
                    "after 11 consecutive business "
                    "days."
                ),
            ),
            make_chunk(
                "3",
                "c.pdf",
                (
                    "Escalation to the Finance "
                    "Business Partner is required "
                    "after 15 consecutive business "
                    "days."
                ),
            ),
        ),
    )

    assert result.total_claims == 3

    assert (
        result.supported_claims
        == 3
    )

    assert (
        result.faithfulness_score
        == 1.0
    )

    for claim_result in (
        result.claim_results
    ):

        assert (
            "Finance Business Partner"
            in claim_result.claim
        )

    assert (
        "6 consecutive business days"
        in result.claim_results[0].claim
    )

    assert (
        "11 consecutive business days"
        in result.claim_results[1].claim
    )

    assert (
        "15 consecutive business days"
        in result.claim_results[2].claim
    )

    print(
        "Finance list context          : PASS"
    )
    
    print(
        "Missing evidence handling    : PASS"
    )

    print()
    print("=" * 80)
    print("FINAL STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()