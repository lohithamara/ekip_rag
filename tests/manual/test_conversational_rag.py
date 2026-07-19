import json
from pathlib import Path

from api.dependencies import create_container

from rag.service.schemas import (
    RAGRequest,
)
from security.authentication.schemas import AuthenticatedUser

from tests.test_auth import (
    TestAuthentication,
)

EVALUATION_FILE = Path(
    "data/evaluation/"
    "conversational_memory_cases.json"
)

ROLE_MAPPING = {

    "finance": "finance_manager",

    "hr": "hr_manager",

    "engineering": "engineering_manager",

    "legal": "legal_employee",

    "support": "admin",
    "general": "admin",
}


def to_source_set(
    values,
) -> set[str]:

    return {
        value
        for value in values
        if value
    }


def print_sources(
    title: str,
    sources: set[str],
) -> None:

    print()
    print(title)

    if not sources:
        print("- NONE")
        return

    for source in sorted(sources):
        print(f"- {source}")


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

    passed = 0

    contextualized_count = 0

    memory_used_count = 0

    retrieval_any_hits = 0
    reranking_any_hits = 0
    context_any_hits = 0
    citation_any_hits = 0

    retrieval_all_hits = 0
    reranking_all_hits = 0
    context_all_hits = 0
    citation_all_hits = 0

    try:

        print()
        print("=" * 100)

        print(
            "END-TO-END CONVERSATIONAL "
            "RAG EVALUATION"
        )

        print("=" * 100)

        for index, case in enumerate(
            cases,
            start=1,
        ):

            conversation_id = (
                f"conversational_test_{index}"
            )

            # --------------------------------
            # FIRST TURN
            # --------------------------------
            first_response = (
                container.rag_service.answer(
                    RAGRequest(
                        query=(
                            case["first_query"]
                        ),
                        user= TestAuthentication.manager_for_department(
                            case["department"]
                        ),
                        conversation_id=(
                            conversation_id
                        ),
                    )
                )
            )
            print("=" * 80)
            print("CURRENT CASE")
            print(case)
            print("=" * 80)
            # --------------------------------
            # FOLLOW-UP TURN
            # --------------------------------

            follow_up_response = (
                container.rag_service.answer(
                    RAGRequest(
                        query=(
                            case["follow_up_query"]
                        ),
                        user=TestAuthentication.manager_for_department(
                            case["department"]
                        ),
                        conversation_id=(
                            conversation_id
                        ),
                    )
                )
            )

            # --------------------------------
            # EXPECTED SOURCES
            # --------------------------------

            expected = set(
                case["expected_documents"]
            )

            # --------------------------------
            # STAGE SOURCES
            # --------------------------------

            retrieved_sources = (
                to_source_set(
                    follow_up_response
                    .metadata.get(
                        "retrieved_sources",
                        [],
                    )
                )
            )

            reranked_sources = (
                to_source_set(
                    follow_up_response
                    .metadata.get(
                        "reranked_sources",
                        [],
                    )
                )
            )

            context_sources = (
                to_source_set(
                    follow_up_response
                    .metadata.get(
                        "context_sources",
                        [],
                    )
                )
            )

            citation_sources = (
                to_source_set(
                    follow_up_response.sources
                )
            )

            # --------------------------------
            # ANY-SOURCE METRICS
            # --------------------------------

            retrieval_any_hit = bool(
                expected
                & retrieved_sources
            )

            reranking_any_hit = bool(
                expected
                & reranked_sources
            )

            context_any_hit = bool(
                expected
                & context_sources
            )

            citation_any_hit = bool(
                expected
                & citation_sources
            )

            # --------------------------------
            # ALL-SOURCE METRICS
            # --------------------------------

            retrieval_all_hit = (
                expected
                <= retrieved_sources
            )

            reranking_all_hit = (
                expected
                <= reranked_sources
            )

            context_all_hit = (
                expected
                <= context_sources
            )

            citation_all_hit = (
                expected
                <= citation_sources
            )

            # --------------------------------
            # CONVERSATIONAL METRICS
            # --------------------------------

            contextualized = bool(
                follow_up_response
                .metadata.get(
                    "query_contextualized",
                    False,
                )
            )

            memory_turns_used = (
                follow_up_response
                .metadata.get(
                    "memory_turns_used",
                    0,
                )
            )

            memory_used = (
                memory_turns_used > 0
            )

            answer_present = bool(
                follow_up_response
                .answer
                .strip()
            )

            # --------------------------------
            # UPDATE COUNTERS
            # --------------------------------

            if contextualized:
                contextualized_count += 1

            if memory_used:
                memory_used_count += 1

            if retrieval_any_hit:
                retrieval_any_hits += 1

            if reranking_any_hit:
                reranking_any_hits += 1

            if context_any_hit:
                context_any_hits += 1

            if citation_any_hit:
                citation_any_hits += 1

            if retrieval_all_hit:
                retrieval_all_hits += 1

            if reranking_all_hit:
                reranking_all_hits += 1

            if context_all_hit:
                context_all_hits += 1

            if citation_all_hit:
                citation_all_hits += 1

            # --------------------------------
            # INFRASTRUCTURE SUCCESS
            #
            # Citation correctness is measured
            # separately.
            # --------------------------------

            success = (
                contextualized
                and memory_used
                and answer_present
                and context_any_hit
            )

            if success:
                passed += 1

            status = (
                "PASS"
                if success
                else "FAIL"
            )

            # --------------------------------
            # OUTPUT
            # --------------------------------

            print()
            print("=" * 100)

            print(
                f"[{index}/{len(cases)}] "
                f"{status} "
                f"{case['name']}"
            )

            print("=" * 100)

            print("FIRST QUERY")
            print(case["first_query"])

            print()

            print("FIRST ANSWER")
            print(first_response.answer)

            print()

            print("FOLLOW-UP QUERY")
            print(case["follow_up_query"])

            print()

            print("RETRIEVAL QUERY")

            print(
                follow_up_response
                .metadata.get(
                    "retrieval_query",
                    case["follow_up_query"],
                )
            )

            print()

            print("FOLLOW-UP ANSWER")
            print(follow_up_response.answer)

            print_sources(
                "EXPECTED DOCUMENTS",
                expected,
            )

            print_sources(
                "RETRIEVED SOURCES",
                retrieved_sources,
            )

            print_sources(
                "RERANKED SOURCES",
                reranked_sources,
            )

            print_sources(
                "CONTEXT SOURCES",
                context_sources,
            )

            print_sources(
                "CITATION SOURCES",
                citation_sources,
            )

            print()
            print(
                "CONVERSATIONAL VALIDATION"
            )

            print(
                "QUERY CONTEXTUALIZED :",
                contextualized,
            )

            print(
                "MEMORY TURNS USED    :",
                memory_turns_used,
            )

            print(
                "ANSWER PRESENT       :",
                answer_present,
            )

            print(
                "CONTEXTUALIZER TOKENS:",
                follow_up_response
                .metadata.get(
                    "contextualizer_tokens",
                    0,
                ),
            )

            print()
            print(
                "STAGE-WISE ANY-SOURCE HITS"
            )

            print(
                "RETRIEVAL HIT :",
                retrieval_any_hit,
            )

            print(
                "RERANKING HIT :",
                reranking_any_hit,
            )

            print(
                "CONTEXT HIT   :",
                context_any_hit,
            )

            print(
                "CITATION HIT  :",
                citation_any_hit,
            )

            print()
            print(
                "STAGE-WISE ALL-SOURCE HITS"
            )

            print(
                "RETRIEVAL HIT :",
                retrieval_all_hit,
            )

            print(
                "RERANKING HIT :",
                reranking_all_hit,
            )

            print(
                "CONTEXT HIT   :",
                context_all_hit,
            )

            print(
                "CITATION HIT  :",
                citation_all_hit,
            )

        # --------------------------------
        # SUMMARY
        # --------------------------------

        case_count = len(cases)

        print()
        print("=" * 100)

        print(
            "CONVERSATIONAL RAG "
            "EVALUATION SUMMARY"
        )

        print("=" * 100)

        print(
            f"Cases                    : "
            f"{case_count}"
        )

        print(
            f"Infrastructure passed    : "
            f"{passed}/{case_count}"
        )

        print(
            f"Infrastructure pass rate : "
            f"{passed / case_count:.3f}"
        )

        print()

        print("CONVERSATIONAL BEHAVIOR")

        print(
            f"Contextualized            : "
            f"{contextualized_count}/"
            f"{case_count}"
        )

        print(
            f"Memory used               : "
            f"{memory_used_count}/"
            f"{case_count}"
        )

        print()

        print("ANY-SOURCE RETENTION")

        print(
            f"Retrieval                 : "
            f"{retrieval_any_hits}/"
            f"{case_count}"
        )

        print(
            f"Reranking                 : "
            f"{reranking_any_hits}/"
            f"{case_count}"
        )

        print(
            f"Context                   : "
            f"{context_any_hits}/"
            f"{case_count}"
        )

        print(
            f"Citation                  : "
            f"{citation_any_hits}/"
            f"{case_count}"
        )

        print()

        print("ALL-SOURCE RETENTION")

        print(
            f"Retrieval                 : "
            f"{retrieval_all_hits}/"
            f"{case_count}"
        )

        print(
            f"Reranking                 : "
            f"{reranking_all_hits}/"
            f"{case_count}"
        )

        print(
            f"Context                   : "
            f"{context_all_hits}/"
            f"{case_count}"
        )

        print(
            f"Citation                  : "
            f"{citation_all_hits}/"
            f"{case_count}"
        )

        print()

        # --------------------------------
        # PIPELINE LOSS ANALYSIS
        # --------------------------------

        retrieval_to_reranking_loss = (
            retrieval_any_hits
            - reranking_any_hits
        )

        reranking_to_context_loss = (
            reranking_any_hits
            - context_any_hits
        )

        context_to_citation_loss = (
            context_any_hits
            - citation_any_hits
        )

        print("PIPELINE ANY-SOURCE LOSSES")

        print(
            "Retrieval -> Reranking :",
            retrieval_to_reranking_loss,
        )

        print(
            "Reranking -> Context   :",
            reranking_to_context_loss,
        )

        print(
            "Context -> Citation     :",
            context_to_citation_loss,
        )

        print()

        # --------------------------------
        # DECISION
        # --------------------------------

        if context_any_hits == 0:

            print(
                "DECISION: EXPECTED EVIDENCE "
                "IS NOT REACHING CONTEXT."
            )

        elif (
            retrieval_any_hits
            > reranking_any_hits
        ):

            print(
                "DECISION: RERANKING IS "
                "REMOVING EXPECTED EVIDENCE."
            )

        elif (
            reranking_any_hits
            > context_any_hits
        ):

            print(
                "DECISION: CONTEXT BUILDING "
                "IS REMOVING EXPECTED EVIDENCE."
            )

        elif (
            context_any_hits
            > citation_any_hits
        ):

            print(
                "DECISION: EXPECTED EVIDENCE "
                "REACHES CONTEXT BUT IS LOST "
                "DURING GENERATION/CITATION."
            )

        else:

            print(
                "DECISION: EXPECTED EVIDENCE "
                "IS RETAINED THROUGH THE "
                "FULL PIPELINE."
            )

    finally:

        container.close()


if __name__ == "__main__":
    main()