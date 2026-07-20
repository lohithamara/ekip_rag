from uuid import uuid4

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from rag.cache.semantic_config import SemanticCacheConfig


class SemanticCacheStore:

    def __init__(
        self,
        client,
        vector_size: int,
        config: SemanticCacheConfig,
    ):
        self.client = client
        self.vector_size = vector_size
        self.config = config

    def ensure_collection(self) -> bool:

        try:
            if self.client.collection_exists(
                self.config.collection_name
            ):
                return True

            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="tenant_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )

            return True

        except Exception as exc:
            print(
                "Semantic cache collection error:",
                exc,
            )
            return False

    def search(
        self,
        vector: list[float],
        tenant_id: str,
        authorized_departments: tuple[str, ...],
    ) -> list:

        if not self.config.enabled:
            return []

        try:
            if not self.ensure_collection():
                return []

            result = self.client.query_points(
                collection_name=self.config.collection_name,
                query=vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="tenant_id",
                            match=MatchValue(
                                value=str(tenant_id)
                            ),
                        ),
                    ]
                ),
                limit=10,
                with_payload=True,
                with_vectors=False,
            )

            authorized = set(
                authorized_departments
            )

            matches = []

            for point in result.points:

                # Results are score ordered, so once the
                # threshold is not met, remaining points
                # can also be ignored.
                if (
                    point.score
                    < self.config.similarity_threshold
                ):
                    continue

                payload = point.payload or {}

                answer_departments = set(
                    payload.get(
                        "answer_departments",
                        [],
                    )
                )

                # Ignore old or malformed cache entries.
                if not answer_departments:
                    continue

                # Admin / unrestricted access.
                if "*" in authorized:
                    matches.append(point)
                    continue

                # A cached answer is safe only when the
                # current user can access every department
                # used to generate that answer.
                if answer_departments.issubset(
                    authorized
                ):
                    matches.append(point)

            return matches

        except Exception as exc:
            print(
                "Semantic cache search failed:",
                exc,
            )
            return []

    def store(
        self,
        vector: list[float],
        cache_id: str,
        tenant_id: str,
        answer_departments: list[str],
        query: str,
    ) -> bool:

        if not self.config.enabled:
            return False

        if not answer_departments:
            return False

        try:
            if not self.ensure_collection():
                return False

            self.client.upsert(
                collection_name=self.config.collection_name,
                points=[
                    PointStruct(
                        id=str(uuid4()),
                        vector=vector,
                        payload={
                            "cache_id": cache_id,
                            "tenant_id": str(
                                tenant_id
                            ),
                            "answer_departments": (
                                answer_departments
                            ),
                            "query": query,
                        },
                    )
                ],
                wait=True,
            )

            return True

        except Exception as exc:
            print(
                "Semantic cache store failed:",
                exc,
            )
            return False