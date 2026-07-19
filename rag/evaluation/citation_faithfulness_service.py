from dataclasses import dataclass, field

from rag.evaluation.citation_faithfulness import (
    ClaimCitationExtractor,
)

from rag.evaluation.citation_judge import (
    CitationFaithfulnessJudge,
    ClaimFaithfulnessResult,
)

from rag.evaluation.evidence import (
    SourceEvidenceResolver,
)

from rag.generation.schemas import (
    ContextChunk,
)


@dataclass(frozen=True, slots=True)
class CitationFaithfulnessEvaluationResult:

    total_claims: int

    supported_claims: int

    partially_supported_claims: int

    unsupported_claims: int

    faithfulness_score: float

    claim_results: tuple[
        ClaimFaithfulnessResult,
        ...,
    ]

    prompt_tokens: int

    completion_tokens: int

    total_tokens: int

    metadata: dict = field(
        default_factory=dict
    )


class CitationFaithfulnessEvaluationService:

    def __init__(
        self,
        judge: CitationFaithfulnessJudge,
        extractor: ClaimCitationExtractor
        | None = None,
        evidence_resolver: SourceEvidenceResolver
        | None = None,
    ):
        self.judge = judge

        self.extractor = (
            extractor
            or ClaimCitationExtractor()
        )

        self.evidence_resolver = (
            evidence_resolver
            or SourceEvidenceResolver()
        )

    def evaluate(
        self,
        answer: str,
        allowed_sources: set[str],
        context_chunks: list[
            ContextChunk
        ]
        | tuple[
            ContextChunk,
            ...,
        ],
    ) -> CitationFaithfulnessEvaluationResult:

        pairs = self.extractor.extract(
            answer=answer,
            allowed_sources=allowed_sources,
        )

        claim_evidence = (
            self.evidence_resolver
            .resolve_all(
                pairs=pairs,
                context_chunks=(
                    context_chunks
                ),
            )
        )

        claim_results = tuple(
            self.judge.evaluate(
                evidence
            )
            for evidence in claim_evidence
        )

        total_claims = len(
            claim_results
        )

        supported_claims = sum(
            result.verdict
            == "SUPPORTED"
            for result in claim_results
        )

        partially_supported_claims = sum(
            result.verdict
            == "PARTIALLY_SUPPORTED"
            for result in claim_results
        )

        unsupported_claims = sum(
            result.verdict
            == "UNSUPPORTED"
            for result in claim_results
        )

        faithfulness_score = (
            self._calculate_score(
                claim_results
            )
        )

        prompt_tokens = sum(
            result.prompt_tokens
            for result in claim_results
        )

        completion_tokens = sum(
            result.completion_tokens
            for result in claim_results
        )

        total_tokens = sum(
            result.total_tokens
            for result in claim_results
        )

        return (
            CitationFaithfulnessEvaluationResult(
                total_claims=total_claims,
                supported_claims=(
                    supported_claims
                ),
                partially_supported_claims=(
                    partially_supported_claims
                ),
                unsupported_claims=(
                    unsupported_claims
                ),
                faithfulness_score=(
                    faithfulness_score
                ),
                claim_results=(
                    claim_results
                ),
                prompt_tokens=(
                    prompt_tokens
                ),
                completion_tokens=(
                    completion_tokens
                ),
                total_tokens=total_tokens,
                metadata={
                    "extracted_pairs": len(
                        pairs
                    ),
                    "resolved_claims": len(
                        claim_evidence
                    ),
                },
            )
        )

    @staticmethod
    def _calculate_score(
        claim_results: tuple[
            ClaimFaithfulnessResult,
            ...,
        ],
    ) -> float:

        if not claim_results:
            return 1.0

        total_score = 0.0

        for result in claim_results:

            if (
                result.verdict
                == "SUPPORTED"
            ):
                total_score += 1.0

            elif (
                result.verdict
                == "PARTIALLY_SUPPORTED"
            ):
                total_score += 0.5

        return (
            total_score
            / len(claim_results)
        )