from pathlib import Path
from math import isfinite

from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.model import EmbeddingModel
from rag.embeddings.service import EmbeddingService
from rag.chunking.storage.serializer import ChunkSerializer


CHUNK_ROOT = Path(
    "data/processed/chunks"
)


def test_persisted_chunk_corpus_can_be_embedded():

    chunk_files = sorted(
        CHUNK_ROOT.rglob("chunks.jsonl")
    )

    assert chunk_files

    config = EmbeddingConfig()

    model = EmbeddingModel(config)

    service = EmbeddingService(
        model=model,
        config=config,
    )

    total_chunks = 0

    dimension = None

    for chunk_file in chunk_files:

        chunks = tuple(
            ChunkSerializer.load_jsonl(
                chunk_file
            )
        )

        assert chunks

        result = service.embed_chunks(
            chunks
        )

        assert result.total_embeddings == len(
            chunks
        )

        assert len(result.records) == len(
            chunks
        )

        assert result.document_id == (
            chunks[0].document_id
        )

        assert result.tenant_id == (
            chunks[0].tenant_id
        )

        if dimension is None:
            dimension = result.dimension

        assert result.dimension == dimension

        for record in result.records:

            assert len(record.vector) == dimension

            assert all(
                isfinite(value)
                for value in record.vector
            )

        total_chunks += len(chunks)

    assert total_chunks > 0

    assert dimension == 384