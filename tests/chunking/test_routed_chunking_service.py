import pytest

from rag.chunking.routing.results import RoutedChunkingResult


def test_routed_service_produces_valid_result(
    clean_document,
    production_chunking_config,
    routed_chunking_service,
):
    result = routed_chunking_service.chunk_document(
        document=clean_document,
        base_config=production_chunking_config,
    )

    assert isinstance(result, RoutedChunkingResult)

    assert result.document_id == clean_document.metadata.document_id
    assert result.tenant_id == clean_document.metadata.tenant_id

    assert result.router_version
    assert result.routing_result.decisions

    assert result.total_chunks == len(result.chunks)
    assert result.total_chunks > 0

    assert len(result.strategy_results) == len(
        result.routing_result.decisions
    )


def test_routed_service_combines_strategy_results(
    clean_document,
    production_chunking_config,
    routed_chunking_service,
):
    result = routed_chunking_service.chunk_document(
        document=clean_document,
        base_config=production_chunking_config,
    )

    expected_chunks = tuple(
        chunk
        for strategy_result in result.strategy_results
        for chunk in strategy_result.chunks
    )

    assert result.chunks == expected_chunks


def test_routed_service_produces_unique_chunk_ids(
    clean_document,
    production_chunking_config,
    routed_chunking_service,
):
    result = routed_chunking_service.chunk_document(
        document=clean_document,
        base_config=production_chunking_config,
    )

    chunk_ids = [
        chunk.chunk_id
        for chunk in result.chunks
    ]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_routed_service_chunks_match_document(
    clean_document,
    production_chunking_config,
    routed_chunking_service,
):
    result = routed_chunking_service.chunk_document(
        document=clean_document,
        base_config=production_chunking_config,
    )

    for chunk in result.chunks:
        assert chunk.document_id == clean_document.metadata.document_id
        assert chunk.tenant_id == clean_document.metadata.tenant_id
        assert chunk.text.strip()