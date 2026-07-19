from sqlalchemy import case

from api.dependencies import (
    create_container,
)

from rag.service.schemas import (
    RAGRequest,
)
from security.authentication.schemas import AuthenticatedUser

from tests.test_auth import TestAuthentication

def main():

    container = create_container()

    try:

        rag_service = (
            container.rag_service
        )

        rag_service.trace_enabled = True

        response = rag_service.answer(
            RAGRequest(
                query="What are the approval limits for the finance team?",
                user=TestAuthentication.finance_manager(),
                conversation_id="1",
            )
        )

        trace = rag_service.last_trace

        assert trace is not None

        assert trace.original_query == (
            "What are the approval limits for the finance team?"
        )

        assert trace.retrieval_query

        assert trace.retrieved_results

        assert (
            trace.reranked_response.results
        )

        assert trace.context.chunks

        assert trace.generated_answer == (
            response.answer
        )

        assert trace.validated_sources == (
            tuple(response.sources)
        )

        print()
        print("=" * 80)
        print("RAG TRACE TEST")
        print("=" * 80)

        print(
            "Original query captured    : PASS"
        )

        print(
            "Retrieval query captured   : PASS"
        )

        print(
            "Retrieved results captured : PASS"
        )

        print(
            "Reranked results captured  : PASS"
        )

        print(
            "Final context captured      : PASS"
        )

        print(
            "Generated answer captured  : PASS"
        )

        print(
            "Validated sources captured : PASS"
        )

        print()
        print("=" * 80)
        print("FINAL STATUS: PASS")
        print("=" * 80)

    finally:

        container.close()


if __name__ == "__main__":
    main()