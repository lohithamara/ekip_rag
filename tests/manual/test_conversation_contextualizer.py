from rag.llm.base import BaseLLM
from rag.llm.schemas import (
    LLMRequest,
    LLMResponse,
)
from rag.llm.service import LLMService

from rag.memory.contextualization.config import (
    ContextualizerConfig,
)
from rag.memory.contextualization.schemas import (
    ContextualizationRequest,
)
from rag.memory.contextualization.service import (
    ConversationContextualizationService,
)
from rag.memory.schemas import ConversationTurn


class FakeLLM(BaseLLM):

    def __init__(
        self,
        answer: str,
    ):
        self.answer = answer
        self.calls = 0

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        self.calls += 1

        return LLMResponse(
            answer=self.answer,
            model_name="fake-llm",
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            finish_reason="stop",
        )


def create_service(
    answer: str,
    enabled: bool = True,
    max_history_turns: int = 3,
    max_history_characters: int = 4000,
):

    fake_llm = FakeLLM(answer)

    service = (
        ConversationContextualizationService(
            llm_service=LLMService(
                fake_llm
            ),
            config=ContextualizerConfig(
                enabled=enabled,
                max_history_turns=(
                    max_history_turns
                ),
                max_history_characters=(
                    max_history_characters
                ),
            ),
        )
    )

    return service, fake_llm


def create_turn(
    user_message: str,
    assistant_message: str,
):

    return ConversationTurn(
        user_message=user_message,
        assistant_message=assistant_message,
    )


def test_contextualization():

    service, fake_llm = create_service(
        answer=(
            "What steps are involved in "
            "employee benefits enrollment?"
        )
    )

    response = service.contextualize(
        ContextualizationRequest(
            query="What steps are involved?",
            turns=(
                create_turn(
                    "How do employees enroll "
                    "in benefits?",
                    "Employees follow the "
                    "benefits enrollment process.",
                ),
            ),
        )
    )

    assert response.contextualized_query == (
        "What steps are involved in "
        "employee benefits enrollment?"
    )

    assert response.was_contextualized

    assert response.model_name == "fake-llm"

    assert response.prompt_tokens == 100

    assert response.completion_tokens == 10

    assert response.total_tokens == 110

    assert fake_llm.calls == 1

    print(
        "Follow-up contextualization    : PASS"
    )


def test_unchanged_query():

    query = (
        "What are the API rate "
        "limiting rules?"
    )

    service, fake_llm = create_service(
        answer=query
    )

    response = service.contextualize(
        ContextualizationRequest(
            query=query,
            turns=(
                create_turn(
                    "Which API endpoint handles "
                    "document ingestion?",
                    "The ingestion endpoint "
                    "handles documents.",
                ),
            ),
        )
    )

    assert (
        response.contextualized_query
        == query
    )

    assert not response.was_contextualized

    assert fake_llm.calls == 1

    print(
        "Standalone query unchanged     : PASS"
    )


def test_disabled_fallback():

    query = "What steps are involved?"

    service, fake_llm = create_service(
        answer="Should never be generated",
        enabled=False,
    )

    response = service.contextualize(
        ContextualizationRequest(
            query=query,
            turns=(
                create_turn(
                    "How do employees enroll "
                    "in benefits?",
                    "Answer",
                ),
            ),
        )
    )

    assert (
        response.contextualized_query
        == query
    )

    assert not response.was_contextualized

    assert fake_llm.calls == 0

    print(
        "Disabled fallback              : PASS"
    )


def test_no_history_fallback():

    query = "What steps are involved?"

    service, fake_llm = create_service(
        answer="Should never be generated"
    )

    response = service.contextualize(
        ContextualizationRequest(
            query=query,
            turns=(),
        )
    )

    assert (
        response.contextualized_query
        == query
    )

    assert not response.was_contextualized

    assert fake_llm.calls == 0

    print(
        "No-history fallback            : PASS"
    )


def test_output_cleaning():

    service, _ = create_service(
        answer=(
            "\"Standalone query: "
            "Who approves employee "
            "remote work requests?\""
        )
    )

    response = service.contextualize(
        ContextualizationRequest(
            query="Who approves it?",
            turns=(
                create_turn(
                    "What are the rules for "
                    "working remotely?",
                    "Remote work follows the "
                    "company policy.",
                ),
            ),
        )
    )

    assert response.contextualized_query == (
        "Who approves employee remote "
        "work requests?"
    )

    print(
        "Output cleaning                : PASS"
    )


def test_invalid_output_fallback():

    query = "What steps are involved?"

    service, fake_llm = create_service(
        answer=""
    )

    response = service.contextualize(
        ContextualizationRequest(
            query=query,
            turns=(
                create_turn(
                    "How do employees enroll "
                    "in benefits?",
                    "Answer",
                ),
            ),
        )
    )

    assert (
        response.contextualized_query
        == query
    )

    assert not response.was_contextualized

    assert fake_llm.calls == 1

    assert response.total_tokens == 110

    print(
        "Invalid-output fallback        : PASS"
    )


def test_history_turn_limit():

    service, fake_llm = create_service(
        answer=(
            "What are the approval limits "
            "for the finance team?"
        ),
        max_history_turns=2,
    )

    turns = (
        create_turn(
            "old_user_1",
            "old_answer_1",
        ),
        create_turn(
            "old_user_2",
            "old_answer_2",
        ),
        create_turn(
            "recent_user_1",
            "recent_answer_1",
        ),
        create_turn(
            "recent_user_2",
            "recent_answer_2",
        ),
    )

    service.contextualize(
        ContextualizationRequest(
            query="What are its limits?",
            turns=turns,
        )
    )

    assert fake_llm.calls == 1

    print(
        "History turn limit             : PASS"
    )


def test_empty_query_validation():

    service, fake_llm = create_service(
        answer="unused"
    )

    try:

        service.contextualize(
            ContextualizationRequest(
                query="   ",
                turns=(),
            )
        )

    except ValueError:

        assert fake_llm.calls == 0

        print(
            "Empty-query validation       : PASS"
        )

        return

    raise AssertionError(
        "Expected ValueError."
    )

def test_architecture_intent_preservation():

    service, fake_llm = create_service(
        answer=(
            "Who owns the Query Service in the "
            "company's microservices architecture?"
        )
    )

    response = service.contextualize(
        ContextualizationRequest(
            query="Who owns the Query Service?",
            turns=(
                create_turn(
                    (
                        "What is the architecture "
                        "of the company's microservices?"
                    ),
                    (
                        "The architecture contains "
                        "multiple services including "
                        "the Query Service."
                    ),
                ),
            ),
        )
    )

    query = (
        response.contextualized_query
        .lower()
    )

    assert "query service" in query

    assert "own" in query

    assert "datastore" not in query

    assert "database" not in query

    assert fake_llm.calls == 1

    print(
        "Architecture intent preserved  : PASS"
    )


def test_remote_work_intent_preservation():

    service, fake_llm = create_service(
        answer=(
            "Who approves remote work "
            "arrangements?"
        )
    )

    response = service.contextualize(
        ContextualizationRequest(
            query="Who approves it?",
            turns=(
                create_turn(
                    (
                        "What are the rules for "
                        "working remotely?"
                    ),
                    (
                        "The conversation concerns "
                        "remote work arrangements."
                    ),
                ),
            ),
        )
    )

    query = (
        response.contextualized_query
        .lower()
    )

    assert "remote work" in query

    assert "approv" in query

    assert "policy" not in query

    assert fake_llm.calls == 1

    print(
        "Remote-work intent preserved   : PASS"
    )


def main():

    print()
    print("=" * 80)
    print(
        "CONVERSATION CONTEXTUALIZER TEST"
    )
    print("=" * 80)
    print()

    test_contextualization()

    test_unchanged_query()

    test_disabled_fallback()

    test_no_history_fallback()

    test_output_cleaning()

    test_invalid_output_fallback()

    test_history_turn_limit()

    test_empty_query_validation()

    test_architecture_intent_preservation()

    test_remote_work_intent_preservation()

    print()
    print("=" * 80)
    print("FINAL STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()