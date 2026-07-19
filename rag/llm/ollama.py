from ollama import Client

from rag.llm.base import BaseLLM
from rag.llm.config import LLMConfig
from rag.llm.schemas import (
    LLMRequest,
    LLMResponse,
)


class OllamaLLM(BaseLLM):

    def __init__(
        self,
        config: LLMConfig,
    ):

        self.config = config

        self.client = Client(
            host=config.base_url,
        )

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        response = self.client.chat(

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

            options={
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },

        )

        usage = response.get(
            "prompt_eval_count",
            0,
        ) + response.get(
            "eval_count",
            0,
        )

        return LLMResponse(

            answer=response["message"][
                "content"
            ],

            model_name=self.config.model_name,

            prompt_tokens=response.get(
                "prompt_eval_count",
                0,
            ),

            completion_tokens=response.get(
                "eval_count",
                0,
            ),

            total_tokens=usage,

            finish_reason=response.get(
                "done_reason",
                "stop",
            ),
        )