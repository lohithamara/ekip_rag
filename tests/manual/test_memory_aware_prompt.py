from rag.generation.schemas import (
    ContextChunk,
    ContextResponse,
)

from rag.memory.schemas import (
    ConversationTurn,
    MemoryResponse,
)

from rag.prompts.builder import (
    PromptBuilder,
)

from rag.prompts.config import (
    PromptConfig,
)

from rag.prompts.schemas import (
    PromptRequest,
)


def main():

    builder = PromptBuilder(
        PromptConfig()
    )

    context = ContextResponse(
        chunks=[
            ContextChunk(
                chunk_id="chunk_1",
                document_id="document_1",
                source_filename=(
                    "refund_policy.md"
                ),
                text=(
                    "Standard customers may request "
                    "a refund within 30 days."
                ),
                token_count=10,
                metadata={},
            )
        ],
        total_chunks=1,
        total_tokens=10,
    )

    memory = MemoryResponse(
        turns=(
            ConversationTurn(
                user_message=(
                    "What is the refund policy "
                    "for enterprise customers?"
                ),
                assistant_message=(
                    "Enterprise customers follow "
                    "contract-specific refund terms."
                ),
            ),
        ),
        total_turns=1,
        total_characters=125,
    )

    response = builder.build(
        PromptRequest(
            query=(
                "What about standard customers?"
            ),
            context=context,
            memory=memory,
        )
    )

    prompt = response.user_prompt

    print()
    print("=" * 80)
    print("MEMORY-AWARE PROMPT TEST")
    print("=" * 80)
    print()
    print(prompt)

    assert (
        "<conversation_history>"
        in prompt
    )

    assert (
        "What is the refund policy "
        "for enterprise customers?"
        in prompt
    )

    assert (
        "Enterprise customers follow "
        "contract-specific refund terms."
        in prompt
    )

    assert (
        "Standard customers may request "
        "a refund within 30 days."
        in prompt
    )

    assert (
        "<current_question>\n"
        "What about standard customers?\n"
        "</current_question>"
        in prompt
    )

    assert (
        "Every factual sentence must contain "
        "at least one citation."
        in response.user_prompt
    )

    assert (
        "Every factual list item must contain "
        "at least one citation."
        in response.user_prompt
    )

    assert (
        "Never use one citation placed in a later "
        "sentence or list item to support an earlier "
        "factual sentence or list item."
        in response.user_prompt
    )

    memory_position = prompt.index(
        "<conversation_history>"
    )

    context_position = prompt.index(
        "<context>"
    )

    question_position = prompt.index(
        "<current_question>"
    )

    assert (
        memory_position
        < context_position
        < question_position
    )

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print("Conversation history present : PASS")
    print("Previous user turn present   : PASS")
    print("Previous answer present      : PASS")
    print("RAG context present          : PASS")
    print("Current question present     : PASS")
    print("Prompt section ordering      : PASS")

    print()
    print("FINAL STATUS: PASS")


if __name__ == "__main__":
    main()