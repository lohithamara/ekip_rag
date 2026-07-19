from rag.memory.config import (
    MemoryConfig,
)

from rag.memory.schemas import (
    ConversationTurn,
    MemoryRequest,
    MemoryResponse,
)

from rag.memory.store import (
    InMemoryConversationStore,
)


class ConversationMemoryService:

    def __init__(
        self,
        store: InMemoryConversationStore,
        config: MemoryConfig,
    ):
        self.store = store
        self.config = config

    def add_turn(
        self,
        tenant_id: str,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:

        if not self.config.enabled:
            return

        tenant_id = tenant_id.strip()
        conversation_id = (
            conversation_id.strip()
        )

        if not tenant_id:
            raise ValueError(
                "tenant_id cannot be empty."
            )

        if not conversation_id:
            raise ValueError(
                "conversation_id cannot be empty."
            )

        if not user_message.strip():
            raise ValueError(
                "user_message cannot be empty."
            )

        if not assistant_message.strip():
            raise ValueError(
                "assistant_message cannot be empty."
            )

        self.store.add_turn(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            turn=ConversationTurn(
                user_message=(
                    user_message.strip()
                ),
                assistant_message=(
                    assistant_message.strip()
                ),
            ),
        )

    def get_memory(
        self,
        request: MemoryRequest,
    ) -> MemoryResponse:

        if not self.config.enabled:
            return self._empty_response()

        if self.config.max_turns < 1:
            raise ValueError(
                "max_turns must be at least 1."
            )

        if self.config.max_characters < 1:
            raise ValueError(
                "max_characters must be "
                "at least 1."
            )

        tenant_id = request.tenant_id.strip()

        conversation_id = (
            request.conversation_id.strip()
        )

        if not tenant_id:
            raise ValueError(
                "tenant_id cannot be empty."
            )

        if not conversation_id:
            raise ValueError(
                "conversation_id cannot be empty."
            )

        turns = self.store.get_turns(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )

        selected = []
        total_characters = 0

        for turn in reversed(
            turns[-self.config.max_turns:]
        ):

            turn_characters = (
                len(turn.user_message)
                + len(turn.assistant_message)
            )

            if (
                total_characters
                + turn_characters
                > self.config.max_characters
            ):
                continue

            selected.append(turn)

            total_characters += (
                turn_characters
            )

        selected.reverse()

        return MemoryResponse(
            turns=tuple(selected),
            total_turns=len(selected),
            total_characters=total_characters,
        )

    def clear(
        self,
        tenant_id: str,
        conversation_id: str,
    ) -> None:

        self.store.clear(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )

    @staticmethod
    def _empty_response() -> MemoryResponse:

        return MemoryResponse(
            turns=(),
            total_turns=0,
            total_characters=0,
        )