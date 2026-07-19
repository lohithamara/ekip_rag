from uuid import NAMESPACE_URL, uuid5
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from rag.vector_store.config import (
    VectorDBConfig,
)

from rag.chunking.schemas import Chunk
from rag.embeddings.schemas import EmbeddingRecord

class QdrantVectorStore:

    def __init__(
        self,
        config: VectorDBConfig,
    ):
        self.config = config

        if not config.url:

            raise ValueError(
                "QDRANT_URL is not configured."
            )

        if not config.api_key:

            raise ValueError(
                "QDRANT_API_KEY is not configured."
            )


        self.client = QdrantClient(
            url=config.url,
            api_key=config.api_key,
        )
    @staticmethod
    def _point_id(chunk_id: str) -> str:

        return str(
            uuid5(
                NAMESPACE_URL,
                chunk_id,
            )
        )
    
    def collection_exists(self) -> bool:

        return self.client.collection_exists(
            collection_name=(
                self.config.collection_name
            )
        )

    def create_collection(self) -> bool:

        if self.collection_exists():
            return False

        self.client.create_collection(
            collection_name=(
                self.config.collection_name
            ),
            vectors_config=VectorParams(
                size=self.config.vector_size,
                distance=self._get_distance(),
            ),
        )

        self.client.create_payload_index(
            collection_name=(
                self.config.collection_name
            ),
            field_name="tenant_id",
            field_schema=(
                PayloadSchemaType.KEYWORD
            ),
        )

        self.client.create_payload_index(
            collection_name=(
                self.config.collection_name
            ),
            field_name="department",
            field_schema=(
                PayloadSchemaType.KEYWORD
            ),
        )

        return True

    def get_collection_info(self):

        if not self.collection_exists():
            raise ValueError(
                "Vector collection does not exist."
            )

        return self.client.get_collection(
            collection_name=(
                self.config.collection_name
            )
        )

    def close(self) -> None:

        self.client.close()

    def _get_distance(self) -> Distance:

        distance = self.config.distance.lower()

        if distance == "cosine":
            return Distance.COSINE

        if distance == "dot":
            return Distance.DOT

        if distance == "euclid":
            return Distance.EUCLID

        raise ValueError(
            f"Unsupported distance: "
            f"{self.config.distance}"
        )
    
    def build_points(
        self,
        chunks: tuple[Chunk, ...],
        records: tuple[EmbeddingRecord, ...],
    ) -> list[PointStruct]:

        if len(chunks) != len(records):
            raise ValueError(
                "Chunk and embedding counts do not match."
            )

        points = []

        for chunk, record in zip(
            chunks,
            records,
        ):

            if chunk.chunk_id != record.chunk_id:
                raise ValueError(
                    "Chunk and embedding IDs do not match."
                )

            if (
                len(record.vector)
                != self.config.vector_size
            ):
                raise ValueError(
                    "Embedding dimension does not match "
                    "vector collection size."
                )
            # print("Building point:")
            # print("Department:", chunk.department)
            # print("Tenant:", chunk.tenant_id)
            # print("Document:", chunk.document_id)

            points.append(
                PointStruct(
                    id=self._point_id(
                        chunk.chunk_id
                    ),
                    vector=list(record.vector),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "tenant_id": chunk.tenant_id,
                        "department": chunk.department,

                        "content_hash": chunk.content_hash,

                        "source_filename": chunk.source_filename,
                        "source_file_type": chunk.source_file_type,

                        "chunk_index": chunk.chunk_index,
                        "chunk_type": chunk.chunk_type,
                        "strategy": chunk.strategy,

                        "token_count": chunk.token_count,

                        "page_numbers": chunk.page_numbers,

                        "section_path": chunk.section_path,

                        "table_ids": chunk.table_ids,

                        "metadata": chunk.metadata,

                        "text": chunk.text,
                    }
                )
            )

        return points

    def upsert_points(
        self,
        points: list[PointStruct],
    ) -> int:

        if not points:
            return 0

        if not self.collection_exists():
            self.create_collection()

        batch_size = (
            self.config.upsert_batch_size
        )

        for start in range(
            0,
            len(points),
            batch_size,
        ):

            batch = points[
                start:start + batch_size
            ]

            self.client.upsert(
                collection_name=(
                    self.config.collection_name
                ),
                points=batch,
                wait=True,
            )

        return len(points)
    
    def delete_stale_points(
        self,
        current_chunk_ids: set[str],
    ) -> int:

        if not self.collection_exists():
            return 0

        current_point_ids = {
            self._point_id(chunk_id)
            for chunk_id in current_chunk_ids
        }

        stale_point_ids = []

        offset = None

        while True:

            points, offset = self.client.scroll(
                collection_name=(
                    self.config.collection_name
                ),
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )

            for point in points:

                if str(point.id) not in current_point_ids:
                    stale_point_ids.append(
                        point.id
                    )

            if offset is None:
                break

        if not stale_point_ids:
            return 0

        self.client.delete(
            collection_name=(
                self.config.collection_name
            ),
            points_selector=PointIdsList(
                points=stale_point_ids
            ),
            wait=True,
        )

        return len(stale_point_ids)

    def delete_document_points(
        self,
        document_id: int,
        tenant_id: int,
    ) -> None:

        if not self.collection_exists():
            return

        self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=str(document_id)
                        ),
                    ),
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(
                            value=str(tenant_id)
                        ),
                    ),
                ]
            ),
            wait=True,
        )

    def recreate_collection(self):

        if self.collection_exists():
            self.client.delete_collection(
                self.config.collection_name
            )

        self.create_collection()

    def search(
        self,
        query_vector,
        limit: int = 5,
        tenant_id: str | None = None,
        department: str | None = None,
        score_threshold: float | None = None,
    ):

        if not self.collection_exists():
            raise ValueError(
                "Vector collection does not exist."
            )

        if (
            len(query_vector)
            != self.config.vector_size
        ):
            raise ValueError(
                "Query vector dimension does not match "
                "vector collection size."
            )

        if limit <= 0:
            raise ValueError(
                "Search limit must be greater than zero."
            )

        conditions = []

        if tenant_id is not None:
            conditions.append(
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(
                        value=tenant_id
                    ),
                )
            )

        if department is not None:
            conditions.append(
                FieldCondition(
                    key="department",
                    match=MatchValue(
                        value=department
                    ),
                )
            )

        query_filter = None

        if conditions:
            query_filter = Filter(
                must=conditions
            )

        result = self.client.query_points(
            collection_name=(
                self.config.collection_name
            ),
            query=list(query_vector),
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )

        return result.points