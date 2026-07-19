from dataclasses import dataclass, field


FactGroup = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RAGQualityCase:

    name: str
    query: str
    tenant_id: str
    department: str
    expected_behavior: str

    expected_facts: tuple[
        FactGroup,
        ...,
    ] = ()

    expected_sources: tuple[
        str,
        ...,
    ] = ()

@dataclass(frozen=True, slots=True)
class RAGQualityResult:

    name: str
    answer: str
    expected_behavior: str

    expected_facts: tuple[
        FactGroup,
        ...,
    ]

    expected_sources: tuple[
        str,
        ...,
    ]

    validated_sources: tuple[
        str,
        ...,
    ]

    answer_present: bool

    abstention_correct: bool

    fact_recall: float

    source_recall: float

    citation_factual_units: int

    citation_cited_units: int

    citation_completeness: float

    citation_claims: int

    citation_supported_claims: int

    citation_partially_supported_claims: int

    citation_unsupported_claims: int

    citation_faithfulness: float

    citation_faithfulness_results: tuple

    citation_faithfulness_prompt_tokens: int

    citation_faithfulness_completion_tokens: int

    citation_faithfulness_total_tokens: int

    uncited_units: tuple[
        str,
        ...,
    ]

    metadata: dict = field(
        default_factory=dict
    )