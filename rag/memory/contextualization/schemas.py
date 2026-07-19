from dataclasses import dataclass

from rag.memory.schemas import (
    ConversationTurn,
)


@dataclass(frozen=True, slots=True)
class ContextualizationRequest:

    query: str

    turns: tuple[
        ConversationTurn,
        ...,
    ]


@dataclass(frozen=True, slots=True)
class ContextualizationResponse:

    original_query: str

    contextualized_query: str

    was_contextualized: bool

    model_name: str | None = None

    prompt_tokens: int = 0

    completion_tokens: int = 0

    total_tokens: int = 0