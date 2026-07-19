import pytest

from rag.chunking.schemas import Chunk
from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.service import EmbeddingService


class FakeEmbeddingModel:

    model_name = "fake-model"

    def encode(self, texts):
        return [
            (float(index), 1.0, 2.0)
            for index, _ in enumerate(texts)
        ]


@pytest.fixture
def embedding_service():
    config = EmbeddingConfig(
        model_name="fake-model",
    )

    return EmbeddingService(
        model=FakeEmbeddingModel(),
        config=config,
    )


@pytest.fixture
def sample_chunks():
    return (
        create_chunk(
            chunk_id="chunk_1",
            chunk_index=0,
            text="First chunk text.",
        ),
        create_chunk(
            chunk_id="chunk_2",
            chunk_index=1,
            text="Second chunk text.",
        ),
    )


def create_chunk(
    chunk_id,
    chunk_index,
    text,
):
    return Chunk(
        chunk_id=chunk_id,
        document_id="document_1",
        content_hash="hash_1",
        tenant_id="tenant_1",
        department="engineering",
        text=text,
        chunk_index=chunk_index,
        strategy="recursive",
        chunking_version="1.0",
        character_count=len(text),
        token_count=len(text.split()),
        source_filename="sample.txt",
        source_file_type=".txt",
        source_s3_key="documents/sample.txt",
        chunk_type="text",
        page_numbers=[],
        table_ids=[],
        section_path=[],
        start_char=None,
        end_char=None,
        parent_chunk_id=None,
        metadata={},
    )


def test_embed_chunks_returns_valid_result(
    embedding_service,
    sample_chunks,
):
    result = embedding_service.embed_chunks(
        sample_chunks
    )

    assert result.document_id == "document_1"
    assert result.tenant_id == "tenant_1"
    assert result.model_name == "fake-model"

    assert result.dimension == 3
    assert result.total_embeddings == 2

    assert len(result.records) == 2


def test_embedding_records_match_chunks(
    embedding_service,
    sample_chunks,
):
    result = embedding_service.embed_chunks(
        sample_chunks
    )

    for chunk, record in zip(
        sample_chunks,
        result.records,
    ):
        assert record.chunk_id == chunk.chunk_id
        assert record.document_id == chunk.document_id
        assert record.tenant_id == chunk.tenant_id
        assert record.department == chunk.department


def test_embedding_metadata_preserves_provenance(
    embedding_service,
    sample_chunks,
):
    result = embedding_service.embed_chunks(
        sample_chunks
    )

    record = result.records[0]

    assert record.metadata["chunk_type"] == "text"
    assert record.metadata["strategy"] == "recursive"
    assert record.metadata["source_filename"] == "sample.txt"


def test_empty_chunks_are_rejected(
    embedding_service,
):
    with pytest.raises(
        ValueError,
        match="empty chunk collection",
    ):
        embedding_service.embed_chunks(())


def test_duplicate_chunk_ids_are_rejected(
    embedding_service,
):
    chunks = (
        create_chunk(
            chunk_id="duplicate",
            chunk_index=0,
            text="First chunk.",
        ),
        create_chunk(
            chunk_id="duplicate",
            chunk_index=1,
            text="Second chunk.",
        ),
    )

    with pytest.raises(
        ValueError,
        match="Duplicate chunk IDs",
    ):
        embedding_service.embed_chunks(chunks)