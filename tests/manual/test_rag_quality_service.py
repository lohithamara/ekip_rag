from rag.evaluation.schemas import (
    RAGQualityCase,
)

from rag.evaluation.service import (
    RAGQualityEvaluationService,
)

from rag.evaluation.trace import (
    RAGTrace,
)

from rag.generation.schemas import (
    ContextChunk,
    ContextResponse,
)

from rag.reranking.schemas import (
    RerankResponse,
)

from rag.evaluation.citation_completeness import (
    CitationCompletenessEvaluator,
)

from rag.evaluation.citation_faithfulness_service import (
    CitationFaithfulnessEvaluationService,
)

from rag.evaluation.citation_judge import (
    CitationFaithfulnessJudge,
)

from rag.llm.schemas import (
    LLMResponse,
)


class FakeLLMService:

    def generate(
        self,
        request,
    ) -> LLMResponse:

        return LLMResponse(
            answer=(
                '{"verdict":"SUPPORTED",'
                '"reason":"Evidence supports '
                'the claim."}'
            ),
            model_name="fake-judge",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            finish_reason="stop",
            metadata={},
        )


def main():

    # ============================================================
    # EVALUATION CASE
    # ============================================================

    case = RAGQualityCase(
        name="password_reset",
        query=(
            "How do I reset my password?"
        ),
        tenant_id="tenant_1",
        department="support",
        expected_behavior="answer",

        # IMPORTANT:
        #
        # expected_facts is:
        #
        # tuple[FactGroup, ...]
        #
        # Each FactGroup is also a tuple.
        expected_facts=(
            (
                "Settings > Security",
            ),
            (
                "Reset Password",
            ),
        ),

        expected_sources=(
            "faq.md",
        ),
    )

    # ============================================================
    # RERANK RESPONSE
    # ============================================================

    reranked_response = RerankResponse(
        results=[],
        model_name="test-reranker",
        total_candidates=0,
        returned_candidates=0,
        elapsed_seconds=0.0,
        metadata={},
    )

    # ============================================================
    # CONTEXT
    #
    # Faithfulness evaluation needs the actual cited evidence.
    # Therefore faq.md must exist in context.
    # ============================================================

    context_chunk = ContextChunk(
        chunk_id="chunk_1",
        document_id="document_1",
        source_filename="faq.md",
        text=(
            "Users can reset their password "
            "by navigating to "
            "Settings > Security and selecting "
            "Reset Password."
        ),
        token_count=18,
        metadata={},
    )

    context = ContextResponse(
        chunks=[
            context_chunk,
        ],
        total_chunks=1,
        total_tokens=18,
    )

    # ============================================================
    # TRACE
    # ============================================================

    trace = RAGTrace(
        original_query=(
            "How do I reset my password?"
        ),
        retrieval_query=(
            "How do I reset my password?"
        ),
        retrieved_results=(),
        reranked_response=(
            reranked_response
        ),
        context=context,
        generated_answer=(
            "Go to Settings > Security and "
            "select Reset Password [faq.md]."
        ),
        validated_sources=(
            "faq.md",
        ),
    )

    # ============================================================
    # CITATION COMPLETENESS EVALUATOR
    # ============================================================

    citation_completeness_evaluator = (
        CitationCompletenessEvaluator()
    )

    # ============================================================
    # CITATION FAITHFULNESS JUDGE
    #
    # Uses FakeLLMService.
    # No external API call is made.
    # ============================================================

    citation_faithfulness_judge = (
        CitationFaithfulnessJudge(
            llm_service=(
                FakeLLMService()
            )
        )
    )

    # ============================================================
    # CITATION FAITHFULNESS SERVICE
    # ============================================================

    citation_faithfulness_service = (
        CitationFaithfulnessEvaluationService(
            judge=(
                citation_faithfulness_judge
            )
        )
    )

    # ============================================================
    # QUALITY EVALUATION SERVICE
    # ============================================================

    service = (
        RAGQualityEvaluationService(
            citation_completeness_evaluator=(
                citation_completeness_evaluator
            ),
            citation_faithfulness_service=(
                citation_faithfulness_service
            ),
        )
    )

    # ============================================================
    # EVALUATE
    # ============================================================

    result = service.evaluate(
        case=case,
        trace=trace,
    )

    # ============================================================
    # BASIC METRIC ASSERTIONS
    # ============================================================

    assert result.answer_present

    assert (
        result.abstention_correct
    )

    assert (
        result.fact_recall
        == 1.0
    )

    assert (
        result.source_recall
        == 1.0
    )

    # ============================================================
    # CITATION COMPLETENESS ASSERTIONS
    # ============================================================

    assert (
        result.citation_factual_units
        == 1
    )

    assert (
        result.citation_cited_units
        == 1
    )

    assert (
        result.citation_completeness
        == 1.0
    )

    assert not (
        result.uncited_units
    )

    # ============================================================
    # CITATION FAITHFULNESS ASSERTIONS
    # ============================================================

    assert (
        result.citation_claims
        == 1
    )

    assert (
        result.citation_supported_claims
        == 1
    )

    assert (
        result
        .citation_partially_supported_claims
        == 0
    )

    assert (
        result.citation_unsupported_claims
        == 0
    )

    assert (
        result.citation_faithfulness
        == 1.0
    )

    assert (
        len(
            result
            .citation_faithfulness_results
        )
        == 1
    )

    assert (
        result
        .citation_faithfulness_results[0]
        .verdict
        == "SUPPORTED"
    )

    # ============================================================
    # FAITHFULNESS TOKEN ASSERTIONS
    # ============================================================

    assert (
        result
        .citation_faithfulness_prompt_tokens
        == 100
    )

    assert (
        result
        .citation_faithfulness_completion_tokens
        == 20
    )

    assert (
        result
        .citation_faithfulness_total_tokens
        == 120
    )

    # ============================================================
    # TRACE METADATA ASSERTIONS
    # ============================================================

    assert (
        result.metadata[
            "retrieved_count"
        ]
        == 0
    )

    assert (
        result.metadata[
            "reranked_count"
        ]
        == 0
    )

    assert (
        result.metadata[
            "context_chunks"
        ]
        == 1
    )

    assert (
        result.metadata[
            "context_tokens"
        ]
        == 18
    )

    assert (
        result.metadata[
            "citation_extracted_pairs"
        ]
        == 1
    )

    assert (
        result.metadata[
            "citation_resolved_claims"
        ]
        == 1
    )

    # ============================================================
    # OUTPUT
    # ============================================================

    print()
    print("=" * 80)
    print("RAG QUALITY SERVICE TEST")
    print("=" * 80)
    print()

    print(
        "Answer presence              : PASS"
    )

    print(
        "Abstention correctness       : PASS"
    )

    print(
        "Fact recall                  : PASS"
    )

    print(
        "Source recall                : PASS"
    )

    print(
        "Citation completeness        : PASS"
    )

    print(
        "Citation faithfulness        : PASS"
    )

    print(
        "Faithfulness token tracking  : PASS"
    )

    print(
        "Trace metadata               : PASS"
    )

    print()
    print("=" * 80)
    print("FINAL STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()