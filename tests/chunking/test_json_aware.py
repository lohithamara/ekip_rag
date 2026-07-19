import json

import pytest

from ingestion.serialization.document_serializer import (
    DocumentSerializer,
)
from rag.chunking.config import ChunkingConfig
from rag.chunking.service import ChunkingService


@pytest.fixture(scope="module")
def json_clean_document(
    clean_files,
):

    for clean_path in clean_files:

        # Clean artifacts for original JSON files
        # end with ".json.json".
        if not clean_path.name.endswith(
            ".json.json"
        ):
            continue

        return DocumentSerializer.load_clean_document(
            clean_path
        )

    pytest.fail(
        "No cleaned JSON source document found."
    )


@pytest.fixture
def json_config():

    return ChunkingConfig(
        strategy="json_aware",
        chunk_size=512,
        chunk_overlap=0,
        min_chunk_size=1,
        length_unit="tokens",
        merge_small_chunks=False,
    )


def test_json_aware_produces_valid_chunks(
    json_clean_document,
    json_config,
):

    result = ChunkingService().chunk_document(
        document=json_clean_document,
        config=json_config,
    )

    assert result.strategy == "json_aware"

    assert (
        result.total_chunks
        == len(result.chunks)
    )

    assert result.total_chunks > 0

    for expected_index, chunk in enumerate(
        result.chunks
    ):

        assert (
            chunk.chunk_index
            == expected_index
        )

        assert chunk.strategy == "json_aware"

        assert (
            chunk.document_id
            == json_clean_document.metadata.document_id
        )

        assert chunk.text.strip()

        assert chunk.token_count > 0


def test_json_aware_is_deterministic(
    json_clean_document,
    json_config,
):

    service = ChunkingService()

    first = service.chunk_document(
        document=json_clean_document,
        config=json_config,
    )

    second = service.chunk_document(
        document=json_clean_document,
        config=json_config,
    )

    assert first.chunks == second.chunks


def test_json_aware_outputs_valid_semantic_pieces(
    json_clean_document,
    json_config,
):

    result = ChunkingService().chunk_document(
        document=json_clean_document,
        config=json_config,
    )

    for chunk in result.chunks:

        assert ":" in chunk.text

        path, serialized_value = (
            chunk.text.split(
                ":",
                maxsplit=1,
            )
        )

        assert path.strip()

        json.loads(
            serialized_value.strip()
        )


def test_json_aware_preserves_directory_records(
    json_clean_document,
    json_config,
):

    result = ChunkingService().chunk_document(
        document=json_clean_document,
        config=json_config,
    )

    directory_chunks = [
        chunk
        for chunk in result.chunks
        if chunk.text.startswith(
            "directory."
        )
    ]

    if not directory_chunks:
        pytest.skip(
            "Selected JSON document has no directory records."
        )

    for chunk in directory_chunks:

        _, serialized_value = chunk.text.split(
            ":",
            maxsplit=1,
        )

        value = json.loads(
            serialized_value.strip()
        )

        assert isinstance(value, dict)