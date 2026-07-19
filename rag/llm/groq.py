from groq import Groq

from rag.llm.base import BaseLLM
from rag.llm.config import LLMConfig
from rag.llm.schemas import (
    LLMRequest,
    LLMResponse,
)


class GroqLLM(BaseLLM):

    def __init__(
        self,
        config: LLMConfig,
    ):

        self.config = config

        self.client = Groq(
            api_key=config.api_key,
            timeout=config.timeout,
        )

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        response = (
            self.client.chat.completions.create(

                model=self.config.model_name,

                messages=[
                    {
                        "role": "system",
                        "content": request.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": request.user_prompt,
                    },
                ],

                temperature=request.temperature,

                max_completion_tokens=request.max_tokens,
            )
        )

        message = response.choices[0].message

        usage = response.usage

        return LLMResponse(

            answer=message.content,

            model_name=response.model,

            prompt_tokens=usage.prompt_tokens,

            completion_tokens=usage.completion_tokens,

            total_tokens=usage.total_tokens,

            finish_reason=response.choices[
                0
            ].finish_reason,

        )