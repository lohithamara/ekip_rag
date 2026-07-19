from collections import Counter
from pathlib import Path

import pytest

from ingestion.serialization.document_serializer import (
    DocumentSerializer,
)
from rag.chunking.config import ChunkingConfig
from rag.chunking.factory import get_chunking_strategy_names
from rag.chunking.service import ChunkingService


PROCESSED_CLEAN_ROOT = Path("data/processed/clean")

EXPECTED_STRATEGIES = {
    "fixed_size",
    "recursive",
    "sentence",
    "paragraph",
    "page_aware",
    "structure_aware",
    "table_aware",
}


@pytest.fixture(scope="module")
def clean_documents():
    clean_files = sorted(
        PROCESSED_CLEAN_ROOT.rglob("*.json")
    )

    if not clean_files:
        pytest.fail("No clean documents found.")

    return [
        DocumentSerializer.load_clean_document(path)
        for path in clean_files
    ]


@pytest.mark.parametrize(
    "strategy",
    sorted(EXPECTED_STRATEGIES),
)
def test_strategy_across_corpus(
    clean_documents,
    strategy,
):
    config = ChunkingConfig(
        strategy=strategy,
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

    service = ChunkingService()

    executed = 0
    not_applicable = 0
    total_chunks = 0
    chunk_types = Counter()

    for document in clean_documents:
        if strategy == "page_aware" and not document.pages:
            not_applicable += 1
            continue

        if strategy == "table_aware" and not document.tables:
            not_applicable += 1
            continue

        result = service.chunk_document(
            document=document,
            config=config,
        )

        executed += 1
        total_chunks += result.total_chunks

        assert result.strategy == strategy
        assert result.total_chunks == len(result.chunks)

        for expected_index, chunk in enumerate(result.chunks):
            assert chunk.chunk_index == expected_index
            assert chunk.strategy == strategy
            assert chunk.document_id == document.metadata.document_id
            assert chunk.text.strip()
            assert chunk.token_count <= config.chunk_size

            chunk_types[chunk.chunk_type] += 1

    assert executed > 0
    assert total_chunks > 0

    if strategy == "page_aware":
        assert not_applicable > 0

    if strategy == "table_aware":
        assert not_applicable > 0