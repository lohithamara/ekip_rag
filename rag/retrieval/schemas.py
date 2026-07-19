from dataclasses import dataclass

from security.authorization.schemas import (
    DepartmentScope,
)


@dataclass(slots=True)
class RetrievalRequest:
    """
    Retrieval request sent
    to dense/sparse retrieval.
    """

    query: str

    scope: DepartmentScope

    limit: int


@dataclass(slots=True)
class RetrievalResult:
    """
    Standard retrieval object shared
    across dense, sparse and hybrid
    retrieval.
    """

    chunk_id: str

    document_id: str

    text: str

    score: float

    metadata: dict