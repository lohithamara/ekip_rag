from rag.chunking.storage.service import ChunkStorageService
from dataclasses import replace
import pytest
from rag.chunking.routing.results import RoutedChunkingResult


def test_storage_rejects_zero_chunks(
    tmp_path,
    clean_document,
    production_chunking_config,
    routed_chunking_service,
):
    result = routed_chunking_service.chunk_document(
        document=clean_document,
        base_config=production_chunking_config,
    )

    empty_result = replace(
        result,
        chunks=(),
        total_chunks=0,
    )

    storage = ChunkStorageService(
        storage_root=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="Refusing to persist zero chunks",
    ):
        storage.persist(
            document=clean_document,
            result=empty_result,
            base_config=production_chunking_config,
        )


def test_storage_rejects_document_result_mismatch(
    tmp_path,
    clean_document,
    production_chunking_config,
    routed_chunking_service,
):
    result = routed_chunking_service.chunk_document(
        document=clean_document,
        base_config=production_chunking_config,
    )

    invalid_result = replace(
        result,
        document_id="different-document-id",
    )

    storage = ChunkStorageService(
        storage_root=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="Storage document/result ID mismatch",
    ):
        storage.persist(
            document=clean_document,
            result=invalid_result,
            base_config=production_chunking_config,
        )


def test_corrupted_manifest_causes_rebuild(
    tmp_path,
    clean_document,
    production_chunking_config,
    routed_chunking_service,
):
    result = routed_chunking_service.chunk_document(
        document=clean_document,
        base_config=production_chunking_config,
    )

    storage = ChunkStorageService(
        storage_root=tmp_path,
    )

    first_status = storage.persist(
        document=clean_document,
        result=result,
        base_config=production_chunking_config,
    )

    document_dir = (
        tmp_path
        / clean_document.metadata.tenant_id
        / clean_document.metadata.department
        / clean_document.metadata.content_hash
    )

    manifest_path = (
        document_dir
        / ChunkStorageService.MANIFEST_FILENAME
    )

    manifest_path.write_text(
        "{invalid-json",
        encoding="utf-8",
    )

    rebuild_status = storage.persist(
        document=clean_document,
        result=result,
        base_config=production_chunking_config,
    )

    assert first_status == "WRITTEN"
    assert rebuild_status == "REBUILT"

    loaded_manifest = storage.load_manifest(
        clean_document
    )

    assert loaded_manifest.total_chunks == result.total_chunks

def test_first_persist_writes_artifacts(
    tmp_path,
    clean_document,
    production_chunking_config,
    routed_chunking_service,
):
    result = routed_chunking_service.chunk_document(
        document=clean_document,
        base_config=production_chunking_config,
    )

    storage = ChunkStorageService(
        storage_root=tmp_path,
    )

    status = storage.persist(
        document=clean_document,
        result=result,
        base_config=production_chunking_config,
    )

    assert status == "WRITTEN"

    document_dir = (
        tmp_path
        / clean_document.metadata.tenant_id
        / clean_document.metadata.department
        / clean_document.metadata.content_hash
    )

    assert (
        document_dir
        / ChunkStorageService.CHUNKS_FILENAME
    ).is_file()

    assert (
        document_dir
        / ChunkStorageService.MANIFEST_FILENAME
    ).is_file()


def test_unchanged_persist_is_skipped(
    tmp_path,
    clean_document,
    production_chunking_config,
    routed_chunking_service,
):
    result = routed_chunking_service.chunk_document(
        document=clean_document,
        base_config=production_chunking_config,
    )

    storage = ChunkStorageService(
        storage_root=tmp_path,
    )

    first_status = storage.persist(
        document=clean_document,
        result=result,
        base_config=production_chunking_config,
    )

    second_status = storage.persist(
        document=clean_document,
        result=result,
        base_config=production_chunking_config,
    )

    assert first_status == "WRITTEN"
    assert second_status == "SKIPPED_UP_TO_DATE"


def test_force_persist_rebuilds_existing_artifacts(
    tmp_path,
    clean_document,
    production_chunking_config,
    routed_chunking_service,
):
    result = routed_chunking_service.chunk_document(
        document=clean_document,
        base_config=production_chunking_config,
    )

    storage = ChunkStorageService(
        storage_root=tmp_path,
    )

    storage.persist(
        document=clean_document,
        result=result,
        base_config=production_chunking_config,
    )

    status = storage.persist(
        document=clean_document,
        result=result,
        base_config=production_chunking_config,
        force=True,
    )

    assert status == "REBUILT"


def test_stored_chunks_survive_round_trip(
    tmp_path,
    clean_document,
    production_chunking_config,
    routed_chunking_service,
):
    result = routed_chunking_service.chunk_document(
        document=clean_document,
        base_config=production_chunking_config,
    )

    storage = ChunkStorageService(
        storage_root=tmp_path,
    )

    storage.persist(
        document=clean_document,
        result=result,
        base_config=production_chunking_config,
    )

    loaded_chunks = storage.load_chunks(
        clean_document
    )

    assert len(loaded_chunks) == len(result.chunks)

    assert all(
        original == loaded
        for original, loaded in zip(
            result.chunks,
            loaded_chunks,
        )
    )


def test_stored_manifest_matches_result(
    tmp_path,
    clean_document,
    production_chunking_config,
    routed_chunking_service,
):
    result = routed_chunking_service.chunk_document(
        document=clean_document,
        base_config=production_chunking_config,
    )

    storage = ChunkStorageService(
        storage_root=tmp_path,
    )

    storage.persist(
        document=clean_document,
        result=result,
        base_config=production_chunking_config,
    )

    manifest = storage.load_manifest(
        clean_document
    )

    assert manifest.document_id == result.document_id
    assert manifest.tenant_id == result.tenant_id
    assert manifest.total_chunks == result.total_chunks

    expected_ids = [
        chunk.chunk_id
        for chunk in result.chunks
    ]

    loaded_ids = [
        chunk.chunk_id
        for chunk in storage.load_chunks(
            clean_document
        )
    ]

    assert loaded_ids == expected_ids