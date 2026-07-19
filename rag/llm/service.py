from rag.llm.base import BaseLLM
from rag.llm.schemas import (
    LLMRequest,
    LLMResponse,
)


class LLMService:

    def __init__(
        self,
        llm: BaseLLM,
    ):
        self.llm = llm

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        if not request.user_prompt.strip():
            raise ValueError(
                "User prompt cannot be empty."
            )

        return self.llm.generate(
            request
        )