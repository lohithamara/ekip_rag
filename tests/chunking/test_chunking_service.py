from rag.chunking.factory import get_chunking_strategy_names


EXPECTED_STRATEGIES = {
    "fixed_size",
    "recursive",
    "sentence",
    "paragraph",
    "page_aware",
    "structure_aware",
    "table_aware",
    "json_aware",
}


def test_registered_strategies():
    assert set(get_chunking_strategy_names()) == EXPECTED_STRATEGIES


def test_chunk_document(
    clean_document,
    chunking_config,
    chunking_service,
):
    result = chunking_service.chunk_document(
        document=clean_document,
        config=chunking_config,
    )

    assert result.document_id == clean_document.metadata.document_id
    assert result.strategy == chunking_config.strategy
    assert result.total_chunks == len(result.chunks)
    assert result.total_chunks > 0

    for expected_index, chunk in enumerate(result.chunks):
        assert chunk.chunk_index == expected_index
        assert chunk.document_id == result.document_id
        assert chunk.strategy == result.strategy
        assert chunk.text.strip()