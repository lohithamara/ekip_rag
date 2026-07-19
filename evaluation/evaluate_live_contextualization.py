import json
from pathlib import Path

from api.dependencies import create_container

from rag.memory.contextualization.config import (
    ContextualizerConfig,
)
from rag.memory.contextualization.schemas import (
    ContextualizationRequest,
)
from rag.memory.contextualization.service import (
    ConversationContextualizationService,
)
from rag.memory.schemas import (
    ConversationTurn,
)


EVALUATION_FILE = Path(
    "data/evaluation/"
    "conversational_memory_cases.json"
)


def main():

    if not EVALUATION_FILE.is_file():
        raise RuntimeError(
            "Conversational evaluation "
            "dataset does not exist."
        )

    cases = json.loads(
        EVALUATION_FILE.read_text(
            encoding="utf-8"
        )
    )

    if not cases:
        raise RuntimeError(
            "Conversational evaluation "
            "dataset is empty."
        )

    container = create_container()

    try:

        service = (
            ConversationContextualizationService(
                llm_service=(
                    container
                    .rag_service
                    .llm_service
                ),
                config=ContextualizerConfig(
                    enabled=True,
                    temperature=0.0,
                    max_tokens=128,
                    max_history_turns=3,
                    max_history_characters=4000,
                ),
            )
        )

        contextualized_count = 0

        unchanged_count = 0

        total_prompt_tokens = 0

        total_completion_tokens = 0

        print()
        print("=" * 100)
        print(
            "LIVE CONVERSATION "
            "CONTEXTUALIZATION EVALUATION"
        )
        print("=" * 100)

        for index, case in enumerate(
            cases,
            start=1,
        ):

            # For this evaluation we use a
            # minimal synthetic assistant answer.
            #
            # The contextualizer should mainly
            # depend on the previous user query
            # to resolve the follow-up.

            turns = (
                ConversationTurn(
                    user_message=(
                        case["first_query"]
                    ),
                    assistant_message=(
                        "The previous question "
                        "was answered using the "
                        "enterprise knowledge base."
                    ),
                ),
            )

            response = service.contextualize(
                ContextualizationRequest(
                    query=(
                        case["follow_up_query"]
                    ),
                    turns=turns,
                )
            )

            if response.was_contextualized:
                contextualized_count += 1

            else:
                unchanged_count += 1

            total_prompt_tokens += (
                response.prompt_tokens
            )

            total_completion_tokens += (
                response.completion_tokens
            )

            print()
            print("=" * 100)

            print(
                f"[{index}/{len(cases)}] "
                f"{case['name']}"
            )

            print("=" * 100)

            print("PREVIOUS QUERY")
            print(case["first_query"])

            print()

            print("FOLLOW-UP QUERY")
            print(case["follow_up_query"])

            print()

            print("CONTEXTUALIZED QUERY")
            print(
                response.contextualized_query
            )

            print()

            print(
                "WAS CONTEXTUALIZED:",
                response.was_contextualized,
            )

            print(
                "PROMPT TOKENS:",
                response.prompt_tokens,
            )

            print(
                "COMPLETION TOKENS:",
                response.completion_tokens,
            )

        case_count = len(cases)

        average_prompt_tokens = (
            total_prompt_tokens
            / case_count
        )

        average_completion_tokens = (
            total_completion_tokens
            / case_count
        )

        print()
        print("=" * 100)
        print(
            "LIVE CONTEXTUALIZATION "
            "SUMMARY"
        )
        print("=" * 100)

        print(
            f"Cases                     : "
            f"{case_count}"
        )

        print(
            f"Contextualized            : "
            f"{contextualized_count}"
        )

        print(
            f"Unchanged                 : "
            f"{unchanged_count}"
        )

        print(
            f"Average prompt tokens     : "
            f"{average_prompt_tokens:.1f}"
        )

        print(
            f"Average completion tokens : "
            f"{average_completion_tokens:.1f}"
        )

    finally:

        container.close()


if __name__ == "__main__":
    main()