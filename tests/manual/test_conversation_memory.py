from rag.memory.config import (
    MemoryConfig,
)
from rag.memory.schemas import (
    MemoryRequest,
)
from rag.memory.service import (
    ConversationMemoryService,
)
from rag.memory.store import (
    InMemoryConversationStore,
)


def create_service(
    enabled: bool = True,
    max_turns: int = 3,
    max_characters: int = 1000,
) -> ConversationMemoryService:

    return ConversationMemoryService(
        store=InMemoryConversationStore(),
        config=MemoryConfig(
            enabled=enabled,
            max_turns=max_turns,
            max_characters=max_characters,
        ),
    )


def test_basic_storage():

    service = create_service()

    service.add_turn(
        tenant_id="tenant_1",
        conversation_id="conversation_1",
        user_message="What is the refund policy?",
        assistant_message=(
            "Standard customers may request "
            "a refund within the allowed period."
        ),
    )

    memory = service.get_memory(
        MemoryRequest(
            tenant_id="tenant_1",
            conversation_id="conversation_1",
        )
    )

    assert memory.total_turns == 1

    assert (
        memory.turns[0].user_message
        == "What is the refund policy?"
    )

    print(
        "Basic storage              : PASS"
    )


def test_chronological_order():

    service = create_service(
        max_turns=5
    )

    for index in range(1, 4):

        service.add_turn(
            tenant_id="tenant_1",
            conversation_id="conversation_1",
            user_message=f"user_{index}",
            assistant_message=f"assistant_{index}",
        )

    memory = service.get_memory(
        MemoryRequest(
            tenant_id="tenant_1",
            conversation_id="conversation_1",
        )
    )

    messages = [
        turn.user_message
        for turn in memory.turns
    ]

    assert messages == [
        "user_1",
        "user_2",
        "user_3",
    ]

    print(
        "Chronological order        : PASS"
    )


def test_max_turns():

    service = create_service(
        max_turns=3
    )

    for index in range(1, 7):

        service.add_turn(
            tenant_id="tenant_1",
            conversation_id="conversation_1",
            user_message=f"user_{index}",
            assistant_message=f"assistant_{index}",
        )

    memory = service.get_memory(
        MemoryRequest(
            tenant_id="tenant_1",
            conversation_id="conversation_1",
        )
    )

    messages = [
        turn.user_message
        for turn in memory.turns
    ]

    assert memory.total_turns == 3

    assert messages == [
        "user_4",
        "user_5",
        "user_6",
    ]

    print(
        "Maximum turn limit         : PASS"
    )


def test_character_budget():

    service = create_service(
        max_turns=10,
        max_characters=20,
    )

    service.add_turn(
        tenant_id="tenant_1",
        conversation_id="conversation_1",
        user_message="11111",
        assistant_message="22222",
    )

    service.add_turn(
        tenant_id="tenant_1",
        conversation_id="conversation_1",
        user_message="33333",
        assistant_message="44444",
    )

    service.add_turn(
        tenant_id="tenant_1",
        conversation_id="conversation_1",
        user_message="55555",
        assistant_message="66666",
    )

    memory = service.get_memory(
        MemoryRequest(
            tenant_id="tenant_1",
            conversation_id="conversation_1",
        )
    )

    messages = [
        turn.user_message
        for turn in memory.turns
    ]

    assert memory.total_characters <= 20

    assert memory.total_turns == 2

    assert messages == [
        "33333",
        "55555",
    ]

    print(
        "Character budget           : PASS"
    )


def test_tenant_isolation():

    service = create_service()

    service.add_turn(
        tenant_id="tenant_1",
        conversation_id="conversation_1",
        user_message="tenant_1_question",
        assistant_message="tenant_1_answer",
    )

    service.add_turn(
        tenant_id="tenant_2",
        conversation_id="conversation_1",
        user_message="tenant_2_question",
        assistant_message="tenant_2_answer",
    )

    tenant_1_memory = service.get_memory(
        MemoryRequest(
            tenant_id="tenant_1",
            conversation_id="conversation_1",
        )
    )

    tenant_2_memory = service.get_memory(
        MemoryRequest(
            tenant_id="tenant_2",
            conversation_id="conversation_1",
        )
    )

    assert (
        tenant_1_memory.turns[0].user_message
        == "tenant_1_question"
    )

    assert (
        tenant_2_memory.turns[0].user_message
        == "tenant_2_question"
    )

    print(
        "Tenant isolation           : PASS"
    )


def test_conversation_isolation():

    service = create_service()

    service.add_turn(
        tenant_id="tenant_1",
        conversation_id="conversation_1",
        user_message="first_conversation",
        assistant_message="first_answer",
    )

    service.add_turn(
        tenant_id="tenant_1",
        conversation_id="conversation_2",
        user_message="second_conversation",
        assistant_message="second_answer",
    )

    memory_1 = service.get_memory(
        MemoryRequest(
            tenant_id="tenant_1",
            conversation_id="conversation_1",
        )
    )

    memory_2 = service.get_memory(
        MemoryRequest(
            tenant_id="tenant_1",
            conversation_id="conversation_2",
        )
    )

    assert (
        memory_1.turns[0].user_message
        == "first_conversation"
    )

    assert (
        memory_2.turns[0].user_message
        == "second_conversation"
    )

    print(
        "Conversation isolation     : PASS"
    )


def test_clear():

    service = create_service()

    service.add_turn(
        tenant_id="tenant_1",
        conversation_id="conversation_1",
        user_message="question",
        assistant_message="answer",
    )

    service.clear(
        tenant_id="tenant_1",
        conversation_id="conversation_1",
    )

    memory = service.get_memory(
        MemoryRequest(
            tenant_id="tenant_1",
            conversation_id="conversation_1",
        )
    )

    assert memory.total_turns == 0

    assert memory.turns == ()

    print(
        "Clear conversation         : PASS"
    )


def test_disabled_fallback():

    service = create_service(
        enabled=False
    )

    service.add_turn(
        tenant_id="tenant_1",
        conversation_id="conversation_1",
        user_message="question",
        assistant_message="answer",
    )

    memory = service.get_memory(
        MemoryRequest(
            tenant_id="tenant_1",
            conversation_id="conversation_1",
        )
    )

    assert memory.total_turns == 0

    assert memory.total_characters == 0

    assert memory.turns == ()

    print(
        "Disabled fallback          : PASS"
    )


def main():

    print()
    print("=" * 80)
    print("CONVERSATION MEMORY TEST")
    print("=" * 80)
    print()

    test_basic_storage()

    test_chronological_order()

    test_max_turns()

    test_character_budget()

    test_tenant_isolation()

    test_conversation_isolation()

    test_clear()

    test_disabled_fallback()

    print()
    print("=" * 80)
    print("FINAL STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()