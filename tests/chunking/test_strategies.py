import pytest

from rag.chunking.config import ChunkingConfig
from rag.chunking.service import ChunkingService


@pytest.mark.parametrize(
    "strategy",
    [
        "fixed_size",
        "recursive",
    ],
)
def test_strategy_produces_valid_chunks(
    clean_document,
    strategy,
):
    config = ChunkingConfig(
        strategy=strategy,
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=20,
        length_unit="tokens",
        merge_small_chunks=True,
    )

    result = ChunkingService().chunk_document(
        document=clean_document,
        config=config,
    )

    assert result.strategy == strategy
    assert result.total_chunks == len(result.chunks)
    assert result.total_chunks > 0

    for expected_index, chunk in enumerate(result.chunks):
        assert chunk.chunk_index == expected_index
        assert chunk.strategy == strategy
        assert chunk.document_id == clean_document.metadata.document_id
        assert chunk.text.strip()
        assert chunk.token_count > 0
        assert chunk.token_count <= config.chunk_size


@pytest.mark.parametrize(
    "strategy",
    [
        "fixed_size",
        "recursive",
    ],
)
def test_strategy_is_deterministic(
    clean_document,
    strategy,
):
    config = ChunkingConfig(
        strategy=strategy,
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=20,
        length_unit="tokens",
        merge_small_chunks=True,
    )

    service = ChunkingService()

    first_result = service.chunk_document(
        document=clean_document,
        config=config,
    )

    second_result = service.chunk_document(
        document=clean_document,
        config=config,
    )

    assert first_result.chunks == second_result.chunks