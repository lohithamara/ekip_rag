from dataclasses import dataclass, field


@dataclass(slots=True)
class LLMRequest:
    """
    Request sent to an LLM.
    """

    system_prompt: str

    user_prompt: str

    temperature: float = 0.0

    max_tokens: int = 512


@dataclass(slots=True)
class LLMResponse:
    """
    Response returned by an LLM.
    """

    answer: str

    model_name: str

    prompt_tokens: int

    completion_tokens: int

    total_tokens: int

    finish_reason: str

    metadata: dict = field(
        default_factory=dict
    )