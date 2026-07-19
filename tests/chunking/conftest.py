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


@pytest.fixture(scope="session")
def clean_files():
    files = sorted(PROCESSED_CLEAN_ROOT.rglob("*.json"))

    if not files:
        pytest.fail("No clean documents found.")

    return files


@pytest.fixture(scope="session")
def clean_document(clean_files):
    return DocumentSerializer.load_clean_document(
        clean_files[0]
    )


@pytest.fixture
def chunking_config():
    return ChunkingConfig(
        strategy="recursive",
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=20,
        length_unit="tokens",
        merge_small_chunks=True,
    )


@pytest.fixture(scope="session")
def production_chunking_config():
    return ChunkingConfig(
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


@pytest.fixture
def chunking_service():
    return ChunkingService()


@pytest.fixture(scope="session")
def routed_chunking_service(
    production_chunking_config,
):
    token_counter = create_token_counter(
        production_chunking_config
    )

    profiler = DocumentProfiler(
        token_counter=token_counter,
        short_document_threshold=512,
    )

    return RoutedChunkingService(
        profiler=profiler,
        router=ChunkingRouter(),
        chunking_service=ChunkingService(),
    )

@pytest.fixture(scope="session")
def multi_strategy_chunking_result(
    clean_files,
    production_chunking_config,
    routed_chunking_service,
):
    for clean_path in clean_files:
        document = DocumentSerializer.load_clean_document(
            clean_path
        )

        result = routed_chunking_service.chunk_document(
            document=document,
            base_config=production_chunking_config,
        )

        if len(result.routing_result.decisions) >= 2:
            return document, result

    pytest.fail("No multi-strategy document found.")