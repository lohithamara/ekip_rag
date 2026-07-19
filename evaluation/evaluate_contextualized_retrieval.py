import json
from pathlib import Path

from api.dependencies import create_container

from rag.generation.schemas import (
    ContextRequest,
)

from rag.llm.schemas import (
    LLMRequest,
)

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


def source_filenames(
    context,
) -> set[str]:

    return {
        chunk.source_filename
        for chunk in context.chunks
        if chunk.source_filename
    }


def build_context(
    rag_service,
    query: str,
    tenant_id: str,
    department: str,
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

    context = (
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

    return context


def generate_first_answer(
    rag_service,
    query: str,
    context,
) -> str:

    prompt = (
        rag_service
        .prompt_builder
        .build(
            PromptRequest(
                query=query,
                context=context,
            )
        )
    )

    response = (
        rag_service
        .llm_service
        .generate(
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

    return response.answer


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

        rag_service = (
            container.rag_service
        )

        contextualization_service = (
            ConversationContextualizationService(
                llm_service=(
                    rag_service.llm_service
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

        original_any_hits = 0
        contextualized_any_hits = 0

        original_all_hits = 0
        contextualized_all_hits = 0

        improved_count = 0
        regressed_count = 0
        unchanged_count = 0

        total_contextualizer_prompt_tokens = 0
        total_contextualizer_completion_tokens = 0
        total_contextualizer_tokens = 0

        print()
        print("=" * 100)
        print(
            "CONTEXTUALIZED RETRIEVAL "
            "EVALUATION"
        )
        print("=" * 100)

        for index, case in enumerate(
            cases,
            start=1,
        ):

            tenant_id = (
                case["tenant_id"]
            )

            department = (
                case["department"]
            )

            first_query = (
                case["first_query"]
            )

            follow_up_query = (
                case["follow_up_query"]
            )

            expected = set(
                case["expected_documents"]
            )

            # --------------------------------
            # FIRST TURN
            # --------------------------------

            first_context = build_context(
                rag_service=rag_service,
                query=first_query,
                tenant_id=tenant_id,
                department=department,
            )

            first_answer = (
                generate_first_answer(
                    rag_service=rag_service,
                    query=first_query,
                    context=first_context,
                )
            )

            turn = ConversationTurn(
                user_message=first_query,
                assistant_message=first_answer,
            )

            # --------------------------------
            # ORIGINAL FOLLOW-UP RETRIEVAL
            # --------------------------------

            original_context = (
                build_context(
                    rag_service=rag_service,
                    query=follow_up_query,
                    tenant_id=tenant_id,
                    department=department,
                )
            )

            original_sources = (
                source_filenames(
                    original_context
                )
            )

            # --------------------------------
            # CONTEXTUALIZATION
            # --------------------------------

            contextualization = (
                contextualization_service
                .contextualize(
                    ContextualizationRequest(
                        query=follow_up_query,
                        turns=(turn,),
                    )
                )
            )

            contextualized_query = (
                contextualization
                .contextualized_query
            )

            total_contextualizer_prompt_tokens += (
                contextualization
                .prompt_tokens
            )

            total_contextualizer_completion_tokens += (
                contextualization
                .completion_tokens
            )

            total_contextualizer_tokens += (
                contextualization
                .total_tokens
            )

            # --------------------------------
            # CONTEXTUALIZED RETRIEVAL
            # --------------------------------

            contextualized_context = (
                build_context(
                    rag_service=rag_service,
                    query=(
                        contextualized_query
                    ),
                    tenant_id=tenant_id,
                    department=department,
                )
            )

            contextualized_sources = (
                source_filenames(
                    contextualized_context
                )
            )

            # --------------------------------
            # METRICS
            # --------------------------------

            original_any = bool(
                expected
                & original_sources
            )

            contextualized_any = bool(
                expected
                & contextualized_sources
            )

            original_all = (
                expected
                <= original_sources
            )

            contextualized_all = (
                expected
                <= contextualized_sources
            )

            if original_any:
                original_any_hits += 1

            if contextualized_any:
                contextualized_any_hits += 1

            if original_all:
                original_all_hits += 1

            if contextualized_all:
                contextualized_all_hits += 1

            # Primary outcome:
            # compare expected-source coverage.

            if (
                not original_any
                and contextualized_any
            ):

                outcome = "IMPROVED"

                improved_count += 1

            elif (
                original_any
                and not contextualized_any
            ):

                outcome = "REGRESSED"

                regressed_count += 1

            else:

                outcome = "UNCHANGED"

                unchanged_count += 1

            # --------------------------------
            # OUTPUT
            # --------------------------------

            print()
            print("=" * 100)

            print(
                f"[{index}/{len(cases)}] "
                f"{outcome} "
                f"{case['name']}"
            )

            print("=" * 100)

            print("PREVIOUS QUERY")
            print(first_query)

            print()

            print("ORIGINAL FOLLOW-UP")
            print(follow_up_query)

            print()

            print("CONTEXTUALIZED QUERY")
            print(contextualized_query)

            print()

            print("WAS CONTEXTUALIZED")
            print(
                contextualization
                .was_contextualized
            )

            print()

            print("EXPECTED DOCUMENTS")

            for source in sorted(expected):
                print(f"- {source}")

            print()

            print(
                "ORIGINAL CONTEXT SOURCES"
            )

            for source in sorted(
                original_sources
            ):
                print(f"- {source}")

            print()

            print(
                "CONTEXTUALIZED "
                "CONTEXT SOURCES"
            )

            for source in sorted(
                contextualized_sources
            ):
                print(f"- {source}")

            print()

            print(
                "ORIGINAL ANY HIT        :",
                original_any,
            )

            print(
                "CONTEXTUALIZED ANY HIT  :",
                contextualized_any,
            )

            print(
                "ORIGINAL ALL HIT        :",
                original_all,
            )

            print(
                "CONTEXTUALIZED ALL HIT  :",
                contextualized_all,
            )

            print(
                "CONTEXTUALIZER TOKENS    :",
                contextualization
                .total_tokens,
            )

        # --------------------------------
        # SUMMARY
        # --------------------------------

        case_count = len(cases)

        average_prompt_tokens = (
            total_contextualizer_prompt_tokens
            / case_count
        )

        average_completion_tokens = (
            total_contextualizer_completion_tokens
            / case_count
        )

        average_total_tokens = (
            total_contextualizer_tokens
            / case_count
        )

        print()
        print("=" * 100)
        print(
            "CONTEXTUALIZED RETRIEVAL "
            "EVALUATION SUMMARY"
        )
        print("=" * 100)

        print(
            f"Cases                       : "
            f"{case_count}"
        )

        print(
            f"Original any-source hits    : "
            f"{original_any_hits}/"
            f"{case_count}"
        )

        print(
            f"Contextualized any hits     : "
            f"{contextualized_any_hits}/"
            f"{case_count}"
        )

        print(
            f"Original all-source hits    : "
            f"{original_all_hits}/"
            f"{case_count}"
        )

        print(
            f"Contextualized all hits     : "
            f"{contextualized_all_hits}/"
            f"{case_count}"
        )

        print(
            f"Improved cases              : "
            f"{improved_count}"
        )

        print(
            f"Regressed cases             : "
            f"{regressed_count}"
        )

        print(
            f"Unchanged cases             : "
            f"{unchanged_count}"
        )

        print(
            f"Average contextualizer "
            f"prompt tokens : "
            f"{average_prompt_tokens:.1f}"
        )

        print(
            f"Average contextualizer "
            f"completion tokens : "
            f"{average_completion_tokens:.1f}"
        )

        print(
            f"Average contextualizer "
            f"total tokens : "
            f"{average_total_tokens:.1f}"
        )

        print()

        if (
            contextualized_any_hits
            > original_any_hits
            and regressed_count == 0
        ):

            print(
                "DECISION: CONTEXTUALIZED "
                "RETRIEVAL IMPROVED THE "
                "PIPELINE."
            )

        elif (
            contextualized_any_hits
            == original_any_hits
            and regressed_count == 0
        ):

            print(
                "DECISION: NO RETRIEVAL "
                "IMPROVEMENT."
            )

        else:

            print(
                "DECISION: CONTEXTUALIZED "
                "RETRIEVAL CAUSED "
                "REGRESSIONS."
            )

    finally:

        container.close()


if __name__ == "__main__":
    main()