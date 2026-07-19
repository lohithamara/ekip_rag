from typing import Any

from pydantic import BaseModel
from pydantic import Field


class QueryRequest(BaseModel):

    query: str = Field(
        min_length=1,
    )

    conversation_id: str | None = None


class QueryResponse(BaseModel):

    answer: str

    sources: list[str]

    metadata: dict[str, Any]