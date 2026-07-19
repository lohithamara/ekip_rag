from rag.evaluation.metrics import (
    abstention_correct,
    fact_recall,
    matched_facts,
    missing_facts,
    source_recall,
)

from rag.evaluation.schemas import (
    RAGQualityCase,
    RAGQualityResult,
)

from rag.evaluation.trace import (
    RAGTrace,
)

from rag.evaluation.citation_completeness import (
    CitationCompletenessEvaluator,
)

from rag.evaluation.citation_faithfulness_service import (
    CitationFaithfulnessEvaluationService,
)


class RAGQualityEvaluationService:

    def __init__(
        self,
        citation_completeness_evaluator:
        CitationCompletenessEvaluator,
        citation_faithfulness_service:
        CitationFaithfulnessEvaluationService,
    ):
        self.citation_completeness_evaluator = (
            citation_completeness_evaluator
        )

        self.citation_faithfulness_service = (
            citation_faithfulness_service
        )

    def evaluate(
        self,
        case: RAGQualityCase,
        trace: RAGTrace,
    ) -> RAGQualityResult:

        # ============================================================
        # VALIDATE TRACE
        # ============================================================

        if (
            trace.original_query.strip()
            != case.query.strip()
        ):
            raise ValueError(
                "Evaluation case and trace "
                "query mismatch."
            )

        answer = (
            trace.generated_answer
        )

        validated_sources = (
            trace.validated_sources
        )

        # ============================================================
        # BASIC ANSWER METRICS
        # ============================================================

        answer_present = bool(
            answer.strip()
        )

        abstention_is_correct = (
            abstention_correct(
                answer=answer,
                expected_behavior=(
                    case.expected_behavior
                ),
            )
        )

        # ============================================================
        # FACT RECALL
        # ============================================================

        answer_fact_recall = (
            fact_recall(
                answer=answer,
                expected_facts=(
                    case.expected_facts
                ),
            )
        )

        matched_expected_facts = (
            matched_facts(
                answer=answer,
                expected_facts=(
                    case.expected_facts
                ),
            )
        )

        missing_expected_facts = (
            missing_facts(
                answer=answer,
                expected_facts=(
                    case.expected_facts
                ),
            )
        )

        # ============================================================
        # SOURCE RECALL
        # ============================================================

        answer_source_recall = (
            source_recall(
                validated_sources=(
                    validated_sources
                ),
                expected_sources=(
                    case.expected_sources
                ),
            )
        )

        # ============================================================
        # CITATION COMPLETENESS
        #
        # Measures whether factual answer units contain citations
        # belonging to validated sources.
        # ============================================================

        citation_completeness_result = (
            self
            .citation_completeness_evaluator
            .evaluate(
                answer=answer,
                allowed_sources=set(
                    validated_sources
                ),
            )
        )

        # ============================================================
        # CITATION FAITHFULNESS
        #
        # Pipeline:
        #
        # answer
        #   ↓
        # claim-citation extraction
        #   ↓
        # cited-source evidence resolution
        #   ↓
        # semantic LLM judging
        #   ↓
        # claim-level faithfulness aggregation
        #
        # Only chunks from cited sources are passed to the judge.
        # ============================================================

        citation_faithfulness_result = (
            self
            .citation_faithfulness_service
            .evaluate(
                answer=answer,
                allowed_sources=set(
                    validated_sources
                ),
                context_chunks=(
                    trace.context.chunks
                ),
            )
        )

        # ============================================================
        # BUILD RESULT
        # ============================================================

        return RAGQualityResult(
            name=case.name,

            answer=answer,

            expected_behavior=(
                case.expected_behavior
            ),

            expected_facts=(
                case.expected_facts
            ),

            expected_sources=(
                case.expected_sources
            ),

            validated_sources=(
                validated_sources
            ),

            # --------------------------------------------------------
            # BASIC QUALITY METRICS
            # --------------------------------------------------------

            answer_present=(
                answer_present
            ),

            abstention_correct=(
                abstention_is_correct
            ),

            fact_recall=(
                answer_fact_recall
            ),

            source_recall=(
                answer_source_recall
            ),

            # --------------------------------------------------------
            # CITATION COMPLETENESS
            # --------------------------------------------------------

            citation_factual_units=(
                citation_completeness_result
                .factual_units
            ),

            citation_cited_units=(
                citation_completeness_result
                .cited_units
            ),

            citation_completeness=(
                citation_completeness_result
                .completeness_score
            ),

            uncited_units=(
                citation_completeness_result
                .uncited_units
            ),

            # --------------------------------------------------------
            # CITATION FAITHFULNESS
            # --------------------------------------------------------

            citation_claims=(
                citation_faithfulness_result
                .total_claims
            ),

            citation_supported_claims=(
                citation_faithfulness_result
                .supported_claims
            ),

            citation_partially_supported_claims=(
                citation_faithfulness_result
                .partially_supported_claims
            ),

            citation_unsupported_claims=(
                citation_faithfulness_result
                .unsupported_claims
            ),

            citation_faithfulness=(
                citation_faithfulness_result
                .faithfulness_score
            ),

            citation_faithfulness_results=(
                citation_faithfulness_result
                .claim_results
            ),

            citation_faithfulness_prompt_tokens=(
                citation_faithfulness_result
                .prompt_tokens
            ),

            citation_faithfulness_completion_tokens=(
                citation_faithfulness_result
                .completion_tokens
            ),

            citation_faithfulness_total_tokens=(
                citation_faithfulness_result
                .total_tokens
            ),

            # --------------------------------------------------------
            # TRACE / DEBUG METADATA
            # --------------------------------------------------------

            metadata={
                "retrieval_query": (
                    trace.retrieval_query
                ),

                "retrieved_count": len(
                    trace.retrieved_results
                ),

                "reranked_count": len(
                    trace
                    .reranked_response
                    .results
                ),

                "context_chunks": len(
                    trace.context.chunks
                ),

                "context_tokens": (
                    trace.context.total_tokens
                ),

                "matched_facts": (
                    matched_expected_facts
                ),

                "missing_facts": (
                    missing_expected_facts
                ),

                "citation_extracted_pairs": (
                    citation_faithfulness_result
                    .metadata.get(
                        "extracted_pairs",
                        0,
                    )
                ),

                "citation_resolved_claims": (
                    citation_faithfulness_result
                    .metadata.get(
                        "resolved_claims",
                        0,
                    )
                ),
            },
        )