from abc import ABC
from abc import abstractmethod

from rag.llm.schemas import (
    LLMRequest,
    LLMResponse,
)


class BaseLLM(ABC):

    @abstractmethod
    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        """

        raise NotImplementedError