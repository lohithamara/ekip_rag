from rag.analytics.service import (
    AnalyticsService,
)

from rag.documents.local_repository import (
    LocalDocumentRepository,
)

from rag.documents.service import (
    DocumentService,
)

from rag.executors.analytics_executor import (
    AnalyticsExecutor,
)

from rag.executors.request import (
    ExecutionRequest,
)

from rag.agent.decision import (
    AgentDecision,
)

from rag.agent.planner_schemas import (
    ExecutionPlan,
)

from rag.service.schemas import (
    RAGRequest,
)

from security.authentication.schemas import (
    AuthenticatedUser,
)
from tests.test_auth import TestAuthentication


def main():

    repository = LocalDocumentRepository(
        root="data/test_data"
    )

    documents = DocumentService(
        repository
    )

    analytics = AnalyticsService()

    executor = AnalyticsExecutor(
        documents=documents,
        analytics=analytics,
    )

    decision = AgentDecision(

        intent="analytics",

        confidence=1.0,

        execution=ExecutionPlan(

            operation="mean",

            parameters={

                "filename":
                    "sales.csv",

                "column":
                    "Sales",
            },
        ),
    )

    request = ExecutionRequest(

        decision=decision,

        request=RAGRequest(

            query="Average sales",

            user=TestAuthentication.finance_manager(),

            conversation_id="1",
        ),
    )

    response = executor.execute(
        request
    )

    print()

    print("=" * 80)

    print("ANALYTICS TEST")

    print("=" * 80)

    print(response.answer)

    print(response.metadata)

    print("=" * 80)


if __name__ == "__main__":

    main()