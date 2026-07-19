from rag.chunking.schemas import Chunk
from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.model import EmbeddingModel
from rag.embeddings.schemas import (
    EmbeddingRecord,
    EmbeddingResult,
)


class EmbeddingService:

    def __init__(
        self,
        model: EmbeddingModel,
        config: EmbeddingConfig,
    ):
        self.model = model
        self.config = config

    def embed_chunks(
        self,
        chunks: tuple[Chunk, ...],
    ) -> EmbeddingResult:

        if not chunks:
            raise ValueError(
                "Cannot embed empty chunk collection."
            )

        self._validate_chunks(chunks)

        texts = [
            chunk.text
            for chunk in chunks
        ]

        vectors = self.model.encode(texts)

        if len(vectors) != len(chunks):
            raise ValueError(
                "Embedding count does not match chunk count."
            )

        dimension = len(vectors[0])

        if dimension == 0:
            raise ValueError(
                "Embedding vectors cannot be empty."
            )

        records = []

        for chunk, vector in zip(
            chunks,
            vectors,
        ):
            if len(vector) != dimension:
                raise ValueError(
                    "Embedding dimensions are inconsistent."
                )

            records.append(
                EmbeddingRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    tenant_id=chunk.tenant_id,
                    department=chunk.department,
                    vector=vector,
                    model_name=self.model.model_name,
                    embedding_version=(
                        self.config.embedding_version
                    ),
                    metadata=self._build_metadata(chunk),
                )
            )

        first_chunk = chunks[0]

        return EmbeddingResult(
            document_id=first_chunk.document_id,
            tenant_id=first_chunk.tenant_id,
            model_name=self.model.model_name,
            embedding_version=(
                self.config.embedding_version
            ),
            dimension=dimension,
            records=tuple(records),
            total_embeddings=len(records),
        )

    @staticmethod
    def _validate_chunks(
        chunks: tuple[Chunk, ...],
    ) -> None:

        first_chunk = chunks[0]

        chunk_ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Duplicate chunk IDs detected."
            )

        for chunk in chunks:
            if not chunk.text.strip():
                raise ValueError(
                    f"Chunk {chunk.chunk_id} has empty text."
                )

            if chunk.document_id != first_chunk.document_id:
                raise ValueError(
                    "Chunks belong to different documents."
                )

            if chunk.tenant_id != first_chunk.tenant_id:
                raise ValueError(
                    "Chunks belong to different tenants."
                )

    @staticmethod
    def _build_metadata(
        chunk: Chunk,
    ) -> dict:

        return {
            "chunk_type": chunk.chunk_type,
            "strategy": chunk.strategy,
            "chunk_index": chunk.chunk_index,
            "page_numbers": list(chunk.page_numbers),
            "table_ids": list(chunk.table_ids),
            "section_path": list(chunk.section_path),
            "source_filename": chunk.source_filename,
            "source_file_type": chunk.source_file_type,
            "source_s3_key": chunk.source_s3_key,
            "content_hash": chunk.content_hash,
        }
    
    def embed_query(
        self,
        query: str,
    ) -> list[float]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        vectors = self.model.encode(
            [query]
        )

        if not vectors:
            raise ValueError(
                "Query embedding failed."
            )

        vector = vectors[0]

        if not vector:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        return list(vector)