from rag.evaluation.citation_judge import (
    CitationFaithfulnessJudge,
)

from rag.evaluation.evidence import (
    ClaimEvidence,
    SourceEvidence,
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
):

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
    print("CITATION FAITHFULNESS JUDGE TEST")
    print("=" * 80)
    print()

    # TEST 1: supported source

    judge = CitationFaithfulnessJudge(
        llm_service=FakeLLMService(
            responses=[
                (
                    '{"verdict":"SUPPORTED",'
                    '"reason":"Evidence directly '
                    'supports the claim."}'
                )
            ]
        )
    )

    supported = judge.evaluate(
        ClaimEvidence(
            claim="The limit is 6 days.",
            citations=(
                "a.pdf",
            ),
            source_evidence=(
                SourceEvidence(
                    source_filename=(
                        "a.pdf"
                    ),
                    chunks=(
                        make_chunk(
                            "1",
                            "a.pdf",
                            (
                                "The approval "
                                "limit is 6 days."
                            ),
                        ),
                    ),
                ),
            ),
            missing_sources=(),
        )
    )

    assert (
        supported.verdict
        == "SUPPORTED"
    )

    assert (
        supported.supported_sources
        == (
            "a.pdf",
        )
    )

    assert not (
        supported.unsupported_sources
    )

    assert (
        supported.total_tokens
        == 120
    )

    print(
        "Supported source             : PASS"
    )

    # TEST 2: mixed support

    judge = CitationFaithfulnessJudge(
        llm_service=FakeLLMService(
            responses=[
                (
                    '{"verdict":"SUPPORTED",'
                    '"reason":"Supports claim."}'
                ),
                (
                    '{"verdict":"UNSUPPORTED",'
                    '"reason":"Unrelated evidence."}'
                ),
            ]
        )
    )

    mixed = judge.evaluate(
        ClaimEvidence(
            claim="The limit is 6 days.",
            citations=(
                "a.pdf",
                "b.pdf",
            ),
            source_evidence=(
                SourceEvidence(
                    source_filename="a.pdf",
                    chunks=(
                        make_chunk(
                            "1",
                            "a.pdf",
                            "Limit is 6 days.",
                        ),
                    ),
                ),
                SourceEvidence(
                    source_filename="b.pdf",
                    chunks=(
                        make_chunk(
                            "2",
                            "b.pdf",
                            "Unrelated information.",
                        ),
                    ),
                ),
            ),
            missing_sources=(),
        )
    )

    assert (
        mixed.verdict
        == "PARTIALLY_SUPPORTED"
    )

    assert (
        mixed.supported_sources
        == (
            "a.pdf",
        )
    )

    assert (
        mixed.unsupported_sources
        == (
            "b.pdf",
        )
    )

    assert (
        mixed.total_tokens
        == 240
    )

    print(
        "Mixed support aggregation    : PASS"
    )

    # TEST 3: no evidence

    judge = CitationFaithfulnessJudge(
        llm_service=FakeLLMService(
            responses=[]
        )
    )

    no_evidence = judge.evaluate(
        ClaimEvidence(
            claim="Unknown claim.",
            citations=(
                "missing.pdf",
            ),
            source_evidence=(),
            missing_sources=(
                "missing.pdf",
            ),
        )
    )

    assert (
        no_evidence.verdict
        == "UNSUPPORTED"
    )

    assert not (
        no_evidence.supported_sources
    )

    assert not (
        no_evidence.unsupported_sources
    )

    assert (
        no_evidence.metadata[
            "missing_sources"
        ]
        == (
            "missing.pdf",
        )
    )

    print(
        "Missing evidence handling    : PASS"
    )

    # TEST 4: markdown JSON cleaning

    judge = CitationFaithfulnessJudge(
        llm_service=FakeLLMService(
            responses=[
                (
                    "```json\n"
                    '{"verdict":'
                    '"PARTIALLY_SUPPORTED",'
                    '"reason":"Only part is '
                    'supported."}\n'
                    "```"
                )
            ]
        )
    )

    cleaned = judge.evaluate(
        ClaimEvidence(
            claim=(
                "The limit is 6 days "
                "and requires CEO approval."
            ),
            citations=(
                "a.pdf",
            ),
            source_evidence=(
                SourceEvidence(
                    source_filename="a.pdf",
                    chunks=(
                        make_chunk(
                            "1",
                            "a.pdf",
                            "Limit is 6 days.",
                        ),
                    ),
                ),
            ),
            missing_sources=(),
        )
    )

    assert (
        cleaned.verdict
        == "PARTIALLY_SUPPORTED"
    )

    print(
        "Markdown JSON cleaning       : PASS"
    )

    print()
    print("=" * 80)
    print("FINAL STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()