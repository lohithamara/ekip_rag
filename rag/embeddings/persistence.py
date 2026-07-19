from pathlib import Path

from rag.chunking.storage.serializer import (
    ChunkSerializer,
)

from rag.embeddings.config import (
    EmbeddingConfig,
)

from rag.embeddings.model import (
    EmbeddingModel,
)

from rag.embeddings.service import (
    EmbeddingService,
)

from rag.embeddings.serializer import (
    EmbeddingSerializer,
)

from rag.embeddings.manifest import (
    create_embedding_manifest,
    write_manifest,
)


class EmbeddingPersistenceService:

    def __init__(self):

        config = EmbeddingConfig()

        model = EmbeddingModel(
            config
        )

        self.config = config

        self.service = EmbeddingService(
            model=model,
            config=config,
        )

    def persist_document(
        self,
        chunk_file: Path,
        output_directory: Path,
    ):

        chunks = tuple(

            ChunkSerializer.load_jsonl(
                chunk_file
            )
        )

        if not chunks:

            raise ValueError(
                "No chunks found."
            )

        result = self.service.embed_chunks(
            chunks
        )

        embedding_file = (
            output_directory
            / "embeddings.jsonl"
        )

        manifest_file = (
            output_directory
            / "embedding_manifest.json"
        )

        EmbeddingSerializer.write_jsonl(
            result.records,
            embedding_file,
        )

        manifest = create_embedding_manifest(

            chunks=chunks,

            config=self.config,

            dimension=result.dimension,

            total_embeddings=result.total_embeddings,
        )

        write_manifest(
            manifest,
            manifest_file,
        )

        return result