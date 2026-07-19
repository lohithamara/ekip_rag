from rag.chunking.chunk_factory import ChunkFactory
from rag.chunking.config import ChunkingConfig
from rag.chunking.tokenization.factory import create_token_counter


def test_chunk_factory_creates_deterministic_id(
    clean_document,
):
    config = ChunkingConfig(
        strategy="component_test",
        chunk_size=512,
        chunk_overlap=64,
        min_chunk_size=50,
        length_unit="tokens",
    )

    factory = ChunkFactory(
        config=config,
        token_counter=create_token_counter(config),
    )

    text = (
        "This is a deterministic chunk factory "
        "test for the production RAG system."
    )

    common_arguments = {
        "document": clean_document,
        "text": text,
        "chunk_index": 0,
        "section_path": [
            "Architecture",
            "Retrieval",
        ],
        "metadata": {
            "test": True,
        },
    }

    chunk_1 = factory.create_chunk(
        page_numbers=[2, 1, 2],
        **common_arguments,
    )

    chunk_2 = factory.create_chunk(
        page_numbers=[1, 2],
        **common_arguments,
    )

    assert chunk_1.chunk_id == chunk_2.chunk_id


def test_chunk_factory_populates_metadata(
    clean_document,
):
    config = ChunkingConfig(
        strategy="component_test",
        chunk_size=512,
        chunk_overlap=64,
        min_chunk_size=50,
        length_unit="tokens",
    )

    factory = ChunkFactory(
        config=config,
        token_counter=create_token_counter(config),
    )

    text = "Chunk factory metadata test."

    chunk = factory.create_chunk(
        document=clean_document,
        text=text,
        chunk_index=0,
        page_numbers=[2, 1, 2],
        section_path=[
            "Architecture",
            "Retrieval",
        ],
        metadata={
            "test": True,
        },
    )

    assert chunk.document_id == clean_document.metadata.document_id
    assert chunk.tenant_id == clean_document.metadata.tenant_id
    assert chunk.department == clean_document.metadata.department

    assert chunk.strategy == config.strategy
    assert chunk.chunking_version == config.chunking_version

    assert chunk.character_count == len(text)
    assert chunk.token_count > 0

    assert chunk.page_numbers == [1, 2]

    assert chunk.section_path == [
        "Architecture",
        "Retrieval",
    ]

    assert chunk.metadata["test"] is True