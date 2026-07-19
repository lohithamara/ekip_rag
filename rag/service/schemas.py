from dataclasses import dataclass, field

from security.authentication.schemas import (
    AuthenticatedUser,
)


@dataclass(slots=True)
class RAGRequest:
    """
    Request entering the
    RAG pipeline.

    Identity is supplied by
    Authentication.
    """

    query: str

    user: AuthenticatedUser

    conversation_id: str


@dataclass(slots=True)
class RAGResponse:

    answer: str

    sources: list[str] = field(
        default_factory=list
    )

    metadata: dict = field(
        default_factory=dict
    )

    file_name: str | None = None

    file_path: str | None = None

    content_type: str | None = None