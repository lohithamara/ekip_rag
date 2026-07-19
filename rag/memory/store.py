from collections import defaultdict

from rag.memory.schemas import (
    ConversationTurn,
)


class InMemoryConversationStore:

    def __init__(self):

        self._conversations = defaultdict(
            list
        )

    @staticmethod
    def _key(
        tenant_id: str,
        conversation_id: str,
    ) -> tuple[str, str]:

        return (
            tenant_id,
            conversation_id,
        )

    def add_turn(
        self,
        tenant_id: str,
        conversation_id: str,
        turn: ConversationTurn,
    ) -> None:

        key = self._key(
            tenant_id,
            conversation_id,
        )

        self._conversations[key].append(
            turn
        )

    def get_turns(
        self,
        tenant_id: str,
        conversation_id: str,
    ) -> tuple[ConversationTurn, ...]:

        key = self._key(
            tenant_id,
            conversation_id,
        )

        return tuple(
            self._conversations.get(
                key,
                ()
            )
        )

    def clear(
        self,
        tenant_id: str,
        conversation_id: str,
    ) -> None:

        key = self._key(
            tenant_id,
            conversation_id,
        )

        self._conversations.pop(
            key,
            None,
        )