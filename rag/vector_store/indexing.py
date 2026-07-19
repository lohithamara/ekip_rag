from pathlib import Path

from rag.chunking.storage.serializer import (
    ChunkSerializer,
)

from rag.embeddings.serializer import (
    EmbeddingSerializer,
)

from rag.vector_store.config import (
    VectorDBConfig,
)

from rag.vector_store.qdrant_store import (
    QdrantVectorStore,
)


class VectorIndexingService:

    def __init__(self):

        self.store = QdrantVectorStore(
            VectorDBConfig()
        )

    def index_document(
        self,
        chunk_file: Path,
        embedding_file: Path,
    ) -> int:

        chunks = tuple(

            ChunkSerializer.load_jsonl(
                chunk_file
            )
        )

        records = tuple(

            EmbeddingSerializer.load_jsonl(
                embedding_file
            )
        )

        points = self.store.build_points(

            chunks=chunks,

            records=records,
        )

        return self.store.upsert_points(
            points
        )

    def close(
        self,
    ):

        self.store.close()