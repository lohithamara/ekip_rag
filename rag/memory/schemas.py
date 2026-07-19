from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationTurn:

    user_message: str

    assistant_message: str


@dataclass(frozen=True)
class MemoryRequest:

    tenant_id: str

    conversation_id: str


@dataclass(frozen=True)
class MemoryResponse:

    turns: tuple[ConversationTurn, ...]

    total_turns: int

    total_characters: int