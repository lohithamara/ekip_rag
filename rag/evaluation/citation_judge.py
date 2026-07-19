import json
import re
from dataclasses import dataclass, field

from rag.evaluation.evidence import ClaimEvidence
from rag.llm.schemas import LLMRequest
from rag.llm.service import LLMService


@dataclass(frozen=True, slots=True)
class SourceFaithfulnessResult:

    source_filename: str

    verdict: str

    reason: str


@dataclass(frozen=True, slots=True)
class ClaimFaithfulnessResult:

    claim: str

    source_results: tuple[
        SourceFaithfulnessResult,
        ...,
    ]

    verdict: str

    supported_sources: tuple[
        str,
        ...,
    ]

    unsupported_sources: tuple[
        str,
        ...,
    ]

    prompt_tokens: int

    completion_tokens: int

    total_tokens: int

    metadata: dict = field(
        default_factory=dict
    )


class CitationFaithfulnessJudge:

    VALID_VERDICTS = {
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED",
    }

    def __init__(
        self,
        llm_service: LLMService,
        max_tokens: int = 256,
    ):
        self.llm_service = llm_service
        self.max_tokens = max_tokens

    def evaluate(
        self,
        evidence: ClaimEvidence,
    ) -> ClaimFaithfulnessResult:

        if not evidence.claim.strip():
            raise ValueError(
                "claim cannot be empty."
            )

        source_results = []

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        for source in (
            evidence.source_evidence
        ):

            source_result, usage = (
                self._evaluate_source(
                    claim=evidence.claim,
                    source_filename=(
                        source.source_filename
                    ),
                    evidence_text=(
                        self._build_evidence_text(
                            source.chunks
                        )
                    ),
                )
            )

            source_results.append(
                source_result
            )

            total_prompt_tokens += (
                usage["prompt_tokens"]
            )

            total_completion_tokens += (
                usage["completion_tokens"]
            )

            total_tokens += (
                usage["total_tokens"]
            )

        source_results_tuple = tuple(
            source_results
        )

        verdict = self._aggregate_verdict(
            source_results_tuple
        )

        supported_sources = tuple(
            result.source_filename
            for result in source_results_tuple
            if result.verdict
            in {
                "SUPPORTED",
                "PARTIALLY_SUPPORTED",
            }
        )

        unsupported_sources = tuple(
            result.source_filename
            for result in source_results_tuple
            if result.verdict
            == "UNSUPPORTED"
        )

        return ClaimFaithfulnessResult(
            claim=evidence.claim,
            source_results=(
                source_results_tuple
            ),
            verdict=verdict,
            supported_sources=(
                supported_sources
            ),
            unsupported_sources=(
                unsupported_sources
            ),
            prompt_tokens=(
                total_prompt_tokens
            ),
            completion_tokens=(
                total_completion_tokens
            ),
            total_tokens=total_tokens,
            metadata={
                "missing_sources": (
                    evidence.missing_sources
                ),
            },
        )

    def _evaluate_source(
        self,
        claim: str,
        source_filename: str,
        evidence_text: str,
    ) -> tuple[
        SourceFaithfulnessResult,
        dict,
    ]:

        system_prompt = (
            "You are a strict citation "
            "faithfulness evaluator. "
            "Determine whether the supplied "
            "source evidence supports the claim. "
            "Use only the supplied evidence. "
            "Do not use outside knowledge. "
            "Return valid JSON only."
        )

        user_prompt = (
            "<task>\n"
            "Evaluate whether the source evidence "
            "supports the claim.\n"
            "</task>\n\n"

            "<verdict_rules>\n"
            "SUPPORTED: The evidence directly "
            "supports the complete claim.\n"

            "PARTIALLY_SUPPORTED: The evidence "
            "supports part of the claim but not "
            "the complete claim.\n"

            "UNSUPPORTED: The evidence does not "
            "support the claim, contradicts it, "
            "or is unrelated.\n"
            "</verdict_rules>\n\n"

            "<output_format>\n"
            "{\"verdict\":\"SUPPORTED|"
            "PARTIALLY_SUPPORTED|UNSUPPORTED\","
            "\"reason\":\"brief explanation\"}\n"
            "</output_format>\n\n"

            "<claim>\n"
            f"{claim}\n"
            "</claim>\n\n"

            "<source_filename>\n"
            f"{source_filename}\n"
            "</source_filename>\n\n"

            "<source_evidence>\n"
            f"{evidence_text}\n"
            "</source_evidence>"
        )

        response = (
            self.llm_service.generate(
                LLMRequest(
                    system_prompt=(
                        system_prompt
                    ),
                    user_prompt=user_prompt,
                    temperature=0.0,
                    max_tokens=(
                        self.max_tokens
                    ),
                )
            )
        )

        parsed = self._parse_output(
            response.answer
        )

        result = (
            SourceFaithfulnessResult(
                source_filename=(
                    source_filename
                ),
                verdict=parsed["verdict"],
                reason=parsed["reason"],
            )
        )

        usage = {
            "prompt_tokens": (
                response.prompt_tokens
            ),
            "completion_tokens": (
                response.completion_tokens
            ),
            "total_tokens": (
                response.total_tokens
            ),
        }

        return result, usage

    def _build_evidence_text(
        self,
        chunks,
    ) -> str:

        parts = []

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            parts.append(
                (
                    f"<chunk id=\"{index}\">\n"
                    f"{chunk.text}\n"
                    f"</chunk>"
                )
            )

        return "\n\n".join(parts)

    def _parse_output(
        self,
        output: str,
    ) -> dict:

        cleaned = output.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        try:
            parsed = json.loads(
                cleaned
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Citation faithfulness judge "
                "returned invalid JSON."
            ) from exc

        verdict = str(
            parsed.get(
                "verdict",
                "",
            )
        ).strip().upper()

        reason = str(
            parsed.get(
                "reason",
                "",
            )
        ).strip()

        if verdict not in (
            self.VALID_VERDICTS
        ):
            raise ValueError(
                "Citation faithfulness judge "
                "returned invalid verdict."
            )

        if not reason:
            raise ValueError(
                "Citation faithfulness judge "
                "returned empty reason."
            )

        return {
            "verdict": verdict,
            "reason": reason,
        }

    @staticmethod
    def _aggregate_verdict(
        source_results: tuple[
            SourceFaithfulnessResult,
            ...,
        ],
    ) -> str:

        if not source_results:
            return "UNSUPPORTED"

        verdicts = {
            result.verdict
            for result in source_results
        }

        if verdicts == {
            "SUPPORTED",
        }:
            return "SUPPORTED"

        if (
            "SUPPORTED" in verdicts
            or "PARTIALLY_SUPPORTED"
            in verdicts
        ):
            return "PARTIALLY_SUPPORTED"

        return "UNSUPPORTED"