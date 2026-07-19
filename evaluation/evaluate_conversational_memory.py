import json
from pathlib import Path

from api.dependencies import create_container

from rag.generation.schemas import (
    ContextRequest,
)

from rag.llm.schemas import (
    LLMRequest,
)

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

from rag.prompts.schemas import (
    PromptRequest,
)

from rag.reranking.schemas import (
    RerankRequest,
)


EVALUATION_FILE = Path(
    "data/evaluation/"
    "conversational_memory_cases.json"
)


UNKNOWN_TEXT = (
    "i do not know based on the "
    "provided context"
)


def source_filenames(
    context,
) -> set[str]:

    return {
        chunk.source_filename
        for chunk in context.chunks
        if chunk.source_filename
    }


def generate_answer(
    rag_service,
    query,
    context,
    memory=None,
):

    prompt = (
        rag_service.prompt_builder.build(
            PromptRequest(
                query=query,
                context=context,
                memory=memory,
            )
        )
    )

    response = (
        rag_service.llm_service.generate(
            LLMRequest(
                system_prompt=(
                    prompt.system_prompt
                ),
                user_prompt=(
                    prompt.user_prompt
                ),
                temperature=(
                    rag_service
                    .config
                    .temperature
                ),
                max_tokens=(
                    rag_service
                    .config
                    .max_tokens
                ),
            )
        )
    )

    allowed_sources = source_filenames(
        context
    )

    validated_sources = (
        rag_service
        .citation_validator
        .validate(
            answer=response.answer,
            allowed_sources=allowed_sources,
        )
    )

    return (
        response,
        validated_sources,
    )


def build_context(
    rag_service,
    query,
    tenant_id,
    department,
):

    retrieved = (
        rag_service
        .retrieval_service
        .retrieve(
            query=query,
            tenant_id=tenant_id,
            department=department,
            limit=(
                rag_service
                .config
                .retrieval_limit
            ),
        )
    )

    reranked = (
        rag_service
        .reranking_service
        .rerank(
            RerankRequest(
                query=query,
                results=retrieved,
                top_k=(
                    rag_service
                    .config
                    .rerank_top_k
                ),
            )
        )
    )

    return (
        rag_service
        .context_builder
        .build(
            ContextRequest(
                query=query,
                reranked_results=(
                    reranked.results
                ),
            )
        )
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

    memory_service = (
        ConversationMemoryService(
            store=(
                InMemoryConversationStore()
            ),
            config=MemoryConfig(
                enabled=True,
                max_turns=5,
                max_characters=6000,
            ),
        )
    )

    baseline_answered = 0
    memory_answered = 0

    baseline_source_hits = 0
    memory_source_hits = 0

    baseline_cited = 0
    memory_cited = 0

    memory_improved = 0
    memory_regressed = 0

    baseline_prompt_tokens = 0
    memory_prompt_tokens = 0

    try:

        rag_service = container.rag_service

        print()
        print("=" * 100)
        print(
            "CONVERSATIONAL MEMORY "
            "EVALUATION"
        )
        print("=" * 100)

        for index, case in enumerate(
            cases,
            start=1,
        ):

            conversation_id = (
                f"evaluation_{index}"
            )

            first_context = build_context(
                rag_service=rag_service,
                query=case["first_query"],
                tenant_id=case["tenant_id"],
                department=case["department"],
            )

            first_response, _ = (
                generate_answer(
                    rag_service=rag_service,
                    query=case["first_query"],
                    context=first_context,
                )
            )

            memory_service.add_turn(
                tenant_id=case["tenant_id"],
                conversation_id=conversation_id,
                user_message=(
                    case["first_query"]
                ),
                assistant_message=(
                    first_response.answer
                ),
            )

            follow_up_context = build_context(
                rag_service=rag_service,
                query=case["follow_up_query"],
                tenant_id=case["tenant_id"],
                department=case["department"],
            )

            baseline_response, (
                baseline_sources
            ) = generate_answer(
                rag_service=rag_service,
                query=case["follow_up_query"],
                context=follow_up_context,
            )

            memory = (
                memory_service.get_memory(
                    MemoryRequest(
                        tenant_id=(
                            case["tenant_id"]
                        ),
                        conversation_id=(
                            conversation_id
                        ),
                    )
                )
            )

            memory_response, (
                memory_sources
            ) = generate_answer(
                rag_service=rag_service,
                query=case["follow_up_query"],
                context=follow_up_context,
                memory=memory,
            )

            expected = set(
                case["expected_documents"]
            )

            context_sources = (
                source_filenames(
                    follow_up_context
                )
            )

            source_hit = bool(
                expected & context_sources
            )

            baseline_unknown = (
                UNKNOWN_TEXT
                in baseline_response.answer.lower()
            )

            memory_unknown = (
                UNKNOWN_TEXT
                in memory_response.answer.lower()
            )

            baseline_has_citation = bool(
                baseline_sources
            )

            memory_has_citation = bool(
                memory_sources
            )

            if not baseline_unknown:
                baseline_answered += 1

            if not memory_unknown:
                memory_answered += 1

            if source_hit:
                baseline_source_hits += 1
                memory_source_hits += 1

            if baseline_has_citation:
                baseline_cited += 1

            if memory_has_citation:
                memory_cited += 1

            if (
                baseline_unknown
                and not memory_unknown
                and memory_has_citation
            ):
                memory_improved += 1
                outcome = "IMPROVED"

            elif (
                not baseline_unknown
                and memory_unknown
            ):
                memory_regressed += 1
                outcome = "REGRESSED"

            else:
                outcome = "UNCHANGED"

            baseline_prompt_tokens += (
                baseline_response.prompt_tokens
            )

            memory_prompt_tokens += (
                memory_response.prompt_tokens
            )

            print()
            print("=" * 100)

            print(
                f"[{index}/{len(cases)}] "
                f"{outcome} "
                f"{case['name']}"
            )

            print("=" * 100)

            print("FIRST QUERY")
            print(case["first_query"])

            print()
            print("FOLLOW-UP QUERY")
            print(case["follow_up_query"])

            print()
            print("EXPECTED")
            print(sorted(expected))

            print()
            print("CONTEXT SOURCES")
            print(sorted(context_sources))

            print()
            print("BASELINE ANSWER")
            print(baseline_response.answer)

            print()
            print("MEMORY ANSWER")
            print(memory_response.answer)

            print()
            print(
                "BASELINE VALIDATED SOURCES"
            )

            for source in baseline_sources:
                print(f"- {source}")

            print()
            print(
                "MEMORY VALIDATED SOURCES"
            )

            for source in memory_sources:
                print(f"- {source}")

        query_count = len(cases)

        average_baseline_tokens = (
            baseline_prompt_tokens
            / query_count
        )

        average_memory_tokens = (
            memory_prompt_tokens
            / query_count
        )

        if baseline_prompt_tokens > 0:

            token_growth = (
                (
                    memory_prompt_tokens
                    - baseline_prompt_tokens
                )
                / baseline_prompt_tokens
                * 100
            )

        else:

            token_growth = 0.0

        print()
        print("=" * 100)

        print(
            "CONVERSATIONAL MEMORY "
            "EVALUATION SUMMARY"
        )

        print("=" * 100)

        print(
            f"Cases                     : "
            f"{query_count}"
        )

        print(
            f"Baseline answered         : "
            f"{baseline_answered}/{query_count}"
        )

        print(
            f"Memory answered           : "
            f"{memory_answered}/{query_count}"
        )

        print(
            f"Baseline source hits      : "
            f"{baseline_source_hits}/{query_count}"
        )

        print(
            f"Memory source hits        : "
            f"{memory_source_hits}/{query_count}"
        )

        print(
            f"Baseline cited answers    : "
            f"{baseline_cited}/{query_count}"
        )

        print(
            f"Memory cited answers      : "
            f"{memory_cited}/{query_count}"
        )

        print(
            f"Memory improved           : "
            f"{memory_improved}"
        )

        print(
            f"Memory regressed          : "
            f"{memory_regressed}"
        )

        print(
            f"Average baseline tokens   : "
            f"{average_baseline_tokens:.1f}"
        )

        print(
            f"Average memory tokens     : "
            f"{average_memory_tokens:.1f}"
        )

        print(
            f"Prompt token growth       : "
            f"{token_growth:+.1f}%"
        )

    finally:

        container.close()


if __name__ == "__main__":
    main()