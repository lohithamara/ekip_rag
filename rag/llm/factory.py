from rag.llm.base import BaseLLM
from rag.llm.config import LLMConfig


def create_llm(
    config: LLMConfig,
) -> BaseLLM:

    if config.provider == "groq":
        from rag.llm.groq import GroqLLM
        return GroqLLM(config)

    if config.provider == "ollama":
        from rag.llm.ollama import OllamaLLM
        return OllamaLLM(config)

    raise ValueError(
        f"Unsupported provider: {config.provider}"
    )