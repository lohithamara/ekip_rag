from collections import Counter

import pytest

from rag.chunking.storage.manifest import (
    ManifestBuilder,
    ManifestSerializer,
)
from rag.chunking.storage.serializer import ChunkSerializer


@pytest.fixture
def serialization_artifacts(
    tmp_path,
    multi_strategy_chunking_result,
    production_chunking_config,
):
    document, routed_result = (
        multi_strategy_chunking_result
    )

    manifest = ManifestBuilder.build(
        document=document,
        result=routed_result,
        base_config=production_chunking_config,
    )

    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.json"

    ChunkSerializer.write_jsonl(
        routed_result.chunks,
        chunks_path,
    )

    ManifestSerializer.write_json(
        manifest,
        manifest_path,
    )

    loaded_chunks = ChunkSerializer.load_jsonl(
        chunks_path
    )

    loaded_manifest = ManifestSerializer.load_json(
        manifest_path
    )

    return {
        "document": document,
        "routed_result": routed_result,
        "manifest": manifest,
        "loaded_chunks": loaded_chunks,
        "loaded_manifest": loaded_manifest,
    }

def test_chunks_survive_serialization_round_trip(
    serialization_artifacts,
):
    result = serialization_artifacts["routed_result"]
    loaded_chunks = serialization_artifacts["loaded_chunks"]

    assert len(loaded_chunks) == len(result.chunks)

    assert all(
        original == loaded
        for original, loaded in zip(
            result.chunks,
            loaded_chunks,
        )
    )

def test_manifest_survives_serialization_round_trip(
    serialization_artifacts,
):
    manifest = serialization_artifacts["manifest"]
    loaded_manifest = serialization_artifacts["loaded_manifest"]

    assert loaded_manifest == manifest


def test_manifest_chunk_count_matches_serialized_chunks(
    serialization_artifacts,
):
    loaded_chunks = serialization_artifacts["loaded_chunks"]
    loaded_manifest = serialization_artifacts["loaded_manifest"]

    assert loaded_manifest.total_chunks == len(
        loaded_chunks
    )


def test_manifest_chunk_type_counts_match_serialized_chunks(
    serialization_artifacts,
):
    loaded_chunks = serialization_artifacts["loaded_chunks"]
    loaded_manifest = serialization_artifacts["loaded_manifest"]

    actual_counts = Counter(
        chunk.chunk_type
        for chunk in loaded_chunks
    )

    assert loaded_manifest.chunk_type_counts == dict(
        sorted(actual_counts.items())
    )

