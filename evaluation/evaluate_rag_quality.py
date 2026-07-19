import json
from pathlib import Path

from api.dependencies import create_container

from rag.evaluation.schemas import (
    RAGQualityCase,
)

from rag.evaluation.service import (
    RAGQualityEvaluationService,
)

from rag.evaluation.citation_completeness import (
    CitationCompletenessEvaluator,
)

from rag.evaluation.citation_faithfulness_service import (
    CitationFaithfulnessEvaluationService,
)

from rag.evaluation.citation_judge import (
    CitationFaithfulnessJudge,
)

from rag.service.schemas import (
    RAGRequest,
)
from security.authentication.schemas import AuthenticatedUser

from tests.test_auth import TestAuthentication

EVALUATION_FILE = Path(
    "data/evaluation/rag_quality_cases.json"
)


def load_cases() -> tuple[
    RAGQualityCase,
    ...,
]:

    if not EVALUATION_FILE.is_file():
        raise RuntimeError(
            "RAG quality evaluation dataset "
            "does not exist."
        )

    raw_cases = json.loads(
        EVALUATION_FILE.read_text(
            encoding="utf-8"
        )
    )

    if not raw_cases:
        raise RuntimeError(
            "RAG quality evaluation dataset "
            "is empty."
        )

    cases = []

    for item in raw_cases:

        cases.append(
            RAGQualityCase(
                name=item["name"],
                query=item["query"],
                tenant_id=item["tenant_id"],
                department=item["department"],
                expected_behavior=(
                    item["expected_behavior"]
                ),
                expected_facts=tuple(
                    tuple(fact_group)
                    for fact_group
                    in item.get(
                        "expected_facts",
                        [],
                    )
                ),
                expected_sources=tuple(
                    item.get(
                        "expected_sources",
                        [],
                    )
                ),
            )
        )

    return tuple(cases)


def print_fact_groups(
    title: str,
    fact_groups,
):

    print()
    print(title)

    if fact_groups:

        for fact_group in fact_groups:

            print(
                "- "
                + " OR ".join(
                    fact_group
                )
            )

    else:
        print("- NONE")


def print_string_items(
    title: str,
    items,
):

    print()
    print(title)

    if items:

        for item in items:
            print(f"- {item}")

    else:
        print("- NONE")


def print_faithfulness_details(
    result,
):

    print()
    print(
        "CITATION FAITHFULNESS DETAILS"
    )

    if not (
        result
        .citation_faithfulness_results
    ):
        print("- NONE")
        return

    for claim_result in (
        result
        .citation_faithfulness_results
    ):

        print()

        print(
            "CLAIM   :",
            claim_result.claim,
        )

        print(
            "VERDICT :",
            claim_result.verdict,
        )

        print(
            "SUPPORTED SOURCES :",
            (
                ", ".join(
                    claim_result
                    .supported_sources
                )
                if claim_result
                .supported_sources
                else "NONE"
            ),
        )

        print(
            "UNSUPPORTED SOURCES :",
            (
                ", ".join(
                    claim_result
                    .unsupported_sources
                )
                if claim_result
                .unsupported_sources
                else "NONE"
            ),
        )

        missing_sources = (
            claim_result
            .metadata.get(
                "missing_sources",
                (),
            )
        )

        print(
            "MISSING SOURCES :",
            (
                ", ".join(
                    missing_sources
                )
                if missing_sources
                else "NONE"
            ),
        )

        if claim_result.source_results:

            for source_result in (
                claim_result.source_results
            ):

                print()

                print(
                    "SOURCE  :",
                    source_result
                    .source_filename,
                )

                print(
                    "RESULT  :",
                    source_result.verdict,
                )

                print(
                    "REASON  :",
                    source_result.reason,
                )

        else:

            print(
                "SOURCE RESULTS : NONE"
            )


def main():

    cases = load_cases()

    container = create_container()

    rag_service = (
        container.rag_service
    )

    # Reuse the same production LLM service.
    #
    # It is used by CitationFaithfulnessJudge
    # only for evaluation.
    llm_service = (
        rag_service.llm_service
    )

    citation_completeness_evaluator = (
        CitationCompletenessEvaluator()
    )

    citation_faithfulness_service = (
        CitationFaithfulnessEvaluationService(
            judge=(
                CitationFaithfulnessJudge(
                    llm_service=llm_service,
                )
            )
        )
    )

    evaluation_service = (
        RAGQualityEvaluationService(
            citation_completeness_evaluator=(
                citation_completeness_evaluator
            ),
            citation_faithfulness_service=(
                citation_faithfulness_service
            ),
        )
    )

    rag_service.trace_enabled = True

    passed = 0

    answer_present_count = 0

    abstention_correct_count = 0

    total_fact_recall = 0.0

    total_source_recall = 0.0

    total_citation_completeness = 0.0

    fully_cited_count = 0

    total_citation_faithfulness = 0.0

    fully_faithful_count = 0

    total_faithfulness_claims = 0

    total_supported_claims = 0

    total_partially_supported_claims = 0

    total_unsupported_claims = 0

    faithfulness_prompt_tokens = 0

    faithfulness_completion_tokens = 0

    faithfulness_total_tokens = 0

    # Store diagnostics across all cases.
    #
    # Your old script printed matched facts,
    # missing facts, and uncited units only
    # from the final case because it used
    # `result` after the loop.
    all_matched_facts = []

    all_missing_facts = []

    all_uncited_units = []

    non_faithful_claims = []

    try:

        print()
        print("=" * 100)
        print("RAG QUALITY EVALUATION")
        print("=" * 100)

        for index, case in enumerate(
            cases,
            start=1,
        ):
            
            response = (
                rag_service.answer(
                    RAGRequest(
                        query=case.query,
                        user=TestAuthentication.finance_manager(),
                        conversation_id="1",
                    )
                )
            )

            trace = (
                rag_service.last_trace
            )

            if trace is None:
                raise RuntimeError(
                    "RAG trace was not captured."
                )

            result = (
                evaluation_service.evaluate(
                    case=case,
                    trace=trace,
                )
            )

            # ====================================================
            # AGGREGATE BASIC METRICS
            # ====================================================

            if result.answer_present:
                answer_present_count += 1

            if result.abstention_correct:
                abstention_correct_count += 1

            total_fact_recall += (
                result.fact_recall
            )

            total_source_recall += (
                result.source_recall
            )

            # ====================================================
            # AGGREGATE CITATION COMPLETENESS
            # ====================================================

            total_citation_completeness += (
                result
                .citation_completeness
            )

            if (
                result.citation_completeness
                == 1.0
            ):
                fully_cited_count += 1

            # ====================================================
            # AGGREGATE CITATION FAITHFULNESS
            # ====================================================

            total_citation_faithfulness += (
                result
                .citation_faithfulness
            )

            if (
                result.citation_faithfulness
                == 1.0
            ):
                fully_faithful_count += 1

            total_faithfulness_claims += (
                result.citation_claims
            )

            total_supported_claims += (
                result
                .citation_supported_claims
            )

            total_partially_supported_claims += (
                result
                .citation_partially_supported_claims
            )

            total_unsupported_claims += (
                result
                .citation_unsupported_claims
            )

            faithfulness_prompt_tokens += (
                result
                .citation_faithfulness_prompt_tokens
            )

            faithfulness_completion_tokens += (
                result
                .citation_faithfulness_completion_tokens
            )

            faithfulness_total_tokens += (
                result
                .citation_faithfulness_total_tokens
            )

            # ====================================================
            # COLLECT DIAGNOSTICS
            # ====================================================

            matched = (
                result.metadata.get(
                    "matched_facts",
                    (),
                )
            )

            missing = (
                result.metadata.get(
                    "missing_facts",
                    (),
                )
            )

            all_matched_facts.extend(
                (
                    case.name,
                    fact_group,
                )
                for fact_group in matched
            )

            all_missing_facts.extend(
                (
                    case.name,
                    fact_group,
                )
                for fact_group in missing
            )

            all_uncited_units.extend(
                (
                    case.name,
                    unit,
                )
                for unit
                in result.uncited_units
            )

            for claim_result in (
                result
                .citation_faithfulness_results
            ):

                if (
                    claim_result.verdict
                    != "SUPPORTED"
                ):
                    non_faithful_claims.append(
                        (
                            case.name,
                            claim_result,
                        )
                    )

            # ====================================================
            # DETERMINISTIC PASS CONDITION
            #
            # Do NOT include citation faithfulness yet.
            #
            # First observe the live judge behavior.
            # ====================================================

            success = (
                result.answer_present
                and result.abstention_correct
                and result.fact_recall
                == 1.0
                and result.source_recall
                == 1.0
                and result.citation_completeness
                == 1.0
            )

            if success:
                passed += 1

            status = (
                "PASS"
                if success
                else "FAIL"
            )

            # ====================================================
            # CASE OUTPUT
            # ====================================================

            print()
            print("=" * 100)

            print(
                f"[{index}/{len(cases)}] "
                f"{status} "
                f"{case.name}"
            )

            print("=" * 100)

            print()
            print("QUERY")
            print(case.query)

            print()
            print("EXPECTED BEHAVIOR")
            print(
                case.expected_behavior
            )

            print()
            print("ANSWER")
            print(response.answer)

            print_fact_groups(
                title="EXPECTED FACTS",
                fact_groups=(
                    case.expected_facts
                ),
            )

            print_string_items(
                title="EXPECTED SOURCES",
                items=(
                    case.expected_sources
                ),
            )

            print_string_items(
                title="VALIDATED SOURCES",
                items=(
                    result.validated_sources
                ),
            )

            print()
            print("METRICS")

            print(
                "Answer present              :",
                result.answer_present,
            )

            print(
                "Abstention correct          :",
                result.abstention_correct,
            )

            print(
                "Fact recall                 :",
                f"{result.fact_recall:.3f}",
            )

            print(
                "Source recall               :",
                f"{result.source_recall:.3f}",
            )

            print(
                "Citation factual units      :",
                result.citation_factual_units,
            )

            print(
                "Citation cited units        :",
                result.citation_cited_units,
            )

            print(
                "Citation completeness       :",
                (
                    f"{result.citation_completeness:.3f}"
                ),
            )

            print(
                "Citation claims             :",
                result.citation_claims,
            )

            print(
                "Supported claims            :",
                (
                    result
                    .citation_supported_claims
                ),
            )

            print(
                "Partially supported claims  :",
                (
                    result
                    .citation_partially_supported_claims
                ),
            )

            print(
                "Unsupported claims          :",
                (
                    result
                    .citation_unsupported_claims
                ),
            )

            print(
                "Citation faithfulness       :",
                (
                    f"{result.citation_faithfulness:.3f}"
                ),
            )

            print(
                "Faithfulness prompt tokens  :",
                (
                    result
                    .citation_faithfulness_prompt_tokens
                ),
            )

            print(
                "Faithfulness completion "
                "tokens:",
                (
                    result
                    .citation_faithfulness_completion_tokens
                ),
            )

            print(
                "Faithfulness total tokens   :",
                (
                    result
                    .citation_faithfulness_total_tokens
                ),
            )

            print_faithfulness_details(
                result
            )

            print()
            print("TRACE")

            print(
                "Retrieval query             :",
                result.metadata[
                    "retrieval_query"
                ],
            )

            print(
                "Retrieved count             :",
                result.metadata[
                    "retrieved_count"
                ],
            )

            print(
                "Reranked count              :",
                result.metadata[
                    "reranked_count"
                ],
            )

            print(
                "Context chunks              :",
                result.metadata[
                    "context_chunks"
                ],
            )

            print(
                "Context tokens              :",
                result.metadata[
                    "context_tokens"
                ],
            )

            print(
                "Citation extracted pairs    :",
                result.metadata[
                    "citation_extracted_pairs"
                ],
            )

            print(
                "Citation resolved claims    :",
                result.metadata[
                    "citation_resolved_claims"
                ],
            )

            print_fact_groups(
                title="MATCHED FACTS",
                fact_groups=matched,
            )

            print_fact_groups(
                title="MISSING FACTS",
                fact_groups=missing,
            )

            print_string_items(
                title="UNCITED UNITS",
                items=result.uncited_units,
            )

        # ========================================================
        # SUMMARY
        # ========================================================

        case_count = len(cases)

        average_fact_recall = (
            total_fact_recall
            / case_count
        )

        average_source_recall = (
            total_source_recall
            / case_count
        )

        average_citation_completeness = (
            total_citation_completeness
            / case_count
        )

        average_citation_faithfulness = (
            total_citation_faithfulness
            / case_count
        )

        print()
        print("=" * 100)
        print(
            "RAG QUALITY EVALUATION SUMMARY"
        )
        print("=" * 100)

        print(
            f"Cases                         : "
            f"{case_count}"
        )

        print(
            f"Passed                        : "
            f"{passed}/{case_count}"
        )

        print(
            f"Pass rate                     : "
            f"{passed / case_count:.3f}"
        )

        print(
            f"Answers present               : "
            f"{answer_present_count}/"
            f"{case_count}"
        )

        print(
            f"Abstention correct            : "
            f"{abstention_correct_count}/"
            f"{case_count}"
        )

        print(
            f"Average fact recall           : "
            f"{average_fact_recall:.3f}"
        )

        print(
            f"Average source recall         : "
            f"{average_source_recall:.3f}"
        )

        print(
            f"Fully cited answers           : "
            f"{fully_cited_count}/"
            f"{case_count}"
        )

        print(
            f"Average citation completeness : "
            f"{average_citation_completeness:.3f}"
        )

        print(
            f"Fully faithful answers        : "
            f"{fully_faithful_count}/"
            f"{case_count}"
        )

        print(
            f"Average citation faithfulness : "
            f"{average_citation_faithfulness:.3f}"
        )

        print(
            f"Faithfulness claims           : "
            f"{total_faithfulness_claims}"
        )

        print(
            f"Supported claims              : "
            f"{total_supported_claims}"
        )

        print(
            f"Partially supported claims    : "
            f"{total_partially_supported_claims}"
        )

        print(
            f"Unsupported claims            : "
            f"{total_unsupported_claims}"
        )

        print(
            f"Faithfulness prompt tokens    : "
            f"{faithfulness_prompt_tokens}"
        )

        print(
            f"Faithfulness completion tokens: "
            f"{faithfulness_completion_tokens}"
        )

        print(
            f"Faithfulness total tokens     : "
            f"{faithfulness_total_tokens}"
        )

        # ========================================================
        # GLOBAL DIAGNOSTICS
        # ========================================================

        print()
        print("MATCHED FACTS")

        if all_matched_facts:

            for (
                case_name,
                fact_group,
            ) in all_matched_facts:

                print(
                    f"- [{case_name}] "
                    + " OR ".join(
                        fact_group
                    )
                )

        else:
            print("- NONE")

        print()
        print("MISSING FACTS")

        if all_missing_facts:

            for (
                case_name,
                fact_group,
            ) in all_missing_facts:

                print(
                    f"- [{case_name}] "
                    + " OR ".join(
                        fact_group
                    )
                )

        else:
            print("- NONE")

        print()
        print("UNCITED UNITS")

        if all_uncited_units:

            for (
                case_name,
                unit,
            ) in all_uncited_units:

                print(
                    f"- [{case_name}] "
                    f"{unit}"
                )

        else:
            print("- NONE")

        print()
        print("NON-FULLY-SUPPORTED CLAIMS")

        if non_faithful_claims:

            for (
                case_name,
                claim_result,
            ) in non_faithful_claims:

                print()
                print(
                    "CASE    :",
                    case_name,
                )

                print(
                    "CLAIM   :",
                    claim_result.claim,
                )

                print(
                    "VERDICT :",
                    claim_result.verdict,
                )

                for source_result in (
                    claim_result.source_results
                ):

                    print(
                        "SOURCE  :",
                        source_result
                        .source_filename,
                    )

                    print(
                        "RESULT  :",
                        source_result.verdict,
                    )

                    print(
                        "REASON  :",
                        source_result.reason,
                    )

                missing_sources = (
                    claim_result
                    .metadata.get(
                        "missing_sources",
                        (),
                    )
                )

                if missing_sources:

                    print(
                        "MISSING SOURCES :",
                        ", ".join(
                            missing_sources
                        ),
                    )

        else:
            print("- NONE")

        print()

        if passed == case_count:

            print(
                "DECISION: ALL DETERMINISTIC "
                "RAG QUALITY CASES PASSED."
            )

        else:

            print(
                "DECISION: ONE OR MORE "
                "DETERMINISTIC RAG QUALITY "
                "CASES FAILED."
            )

        print()

        print(
            "NOTE: CITATION FAITHFULNESS "
            "IS OBSERVATIONAL IN THIS RUN "
            "AND DOES NOT AFFECT PASS/FAIL."
        )

    finally:

        rag_service.trace_enabled = False

        container.close()


if __name__ == "__main__":
    main()