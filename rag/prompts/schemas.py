from dataclasses import dataclass

from rag.generation.schemas import (
    ContextResponse,
)

from rag.memory.schemas import (
    MemoryResponse,
)


@dataclass(slots=True)
class PromptRequest:

    query: str

    context: ContextResponse

    memory: MemoryResponse | None = None


@dataclass(slots=True)
class PromptResponse:

    system_prompt: str

    user_prompt: str