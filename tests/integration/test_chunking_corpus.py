from collections import Counter
from pathlib import Path

import pytest

from ingestion.serialization.document_serializer import (
    DocumentSerializer,
)
from rag.chunking.config import ChunkingConfig
from rag.chunking.routing.profiler import DocumentProfiler
from rag.chunking.routing.router import ChunkingRouter
from rag.chunking.routing.service import RoutedChunkingService
from rag.chunking.service import ChunkingService
from rag.chunking.tokenization.factory import create_token_counter


PROCESSED_CLEAN_ROOT = Path("data/processed/clean")


@pytest.fixture(scope="module")
def corpus_chunking_setup():
    clean_files = sorted(
        PROCESSED_CLEAN_ROOT.rglob("*.json")
    )

    if not clean_files:
        pytest.fail("No clean documents found.")

    config = ChunkingConfig(
        strategy="recursive",
        chunking_version="1.0",
        chunk_size=512,
        chunk_overlap=64,
        min_chunk_size=50,
        length_unit="tokens",
        merge_small_chunks=True,
        table_chunking_mode="row_group",
        table_rows_per_chunk=25,
        repeat_table_headers=True,
    )

    token_counter = create_token_counter(config)

    service = RoutedChunkingService(
        profiler=DocumentProfiler(
            token_counter=token_counter,
            short_document_threshold=512,
        ),
        router=ChunkingRouter(),
        chunking_service=ChunkingService(),
    )

    return clean_files, config, service


def test_all_clean_documents_can_be_chunked(
    corpus_chunking_setup,
):
    clean_files, config, service = corpus_chunking_setup

    total_chunks = 0
    chunk_ids = set()
    strategy_counts = Counter()
    chunk_type_counts = Counter()

    for clean_path in clean_files:
        document = DocumentSerializer.load_clean_document(
            clean_path
        )

        result = service.chunk_document(
            document=document,
            base_config=config,
        )

        assert result.total_chunks > 0
        assert result.total_chunks == len(result.chunks)

        total_chunks += result.total_chunks

        for chunk in result.chunks:
            assert chunk.chunk_id not in chunk_ids

            chunk_ids.add(chunk.chunk_id)
            strategy_counts[chunk.strategy] += 1
            chunk_type_counts[chunk.chunk_type] += 1

    assert len(clean_files) == 108

    assert total_chunks == 452
    assert len(chunk_ids) == 452

    assert strategy_counts == {
        "page_aware": 93,
        "recursive": 2,
        "structure_aware": 54,
        "table_aware": 176,
        "json_aware": 127,
    }

    assert chunk_type_counts == {
    "text": 276,
    "table": 176,
    }