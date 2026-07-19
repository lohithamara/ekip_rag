from rag.llm.config import LLMConfig
from rag.llm.factory import create_llm
from rag.llm.schemas import LLMRequest
from rag.llm.service import LLMService


def main():

    config = LLMConfig()

    llm = create_llm(
        config
    )

    service = LLMService(
        llm
    )

    request = LLMRequest(

        system_prompt=(
            "You are a helpful AI assistant."
        ),

        user_prompt=(
            "Explain Retrieval-Augmented "
            "Generation in 100 words."
        ),

        temperature=0.0,

        max_tokens=200,
    )

    response = service.generate(
        request
    )

    print("=" * 80)
    print("MODEL")
    print("=" * 80)

    print(
        response.model_name
    )

    print()

    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(
        response.answer
    )

    print()

    print("=" * 80)
    print("TOKEN USAGE")
    print("=" * 80)

    print(
        f"Prompt Tokens     : {response.prompt_tokens}"
    )

    print(
        f"Completion Tokens : {response.completion_tokens}"
    )

    print(
        f"Total Tokens      : {response.total_tokens}"
    )


if __name__ == "__main__":
    main()