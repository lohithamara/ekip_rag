import json
from dataclasses import replace

from rag.chunking.schemas import Chunk
from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.manifest import (
    calculate_chunk_hash,
    create_embedding_manifest,
    load_manifest,
    write_manifest,
)

from scripts.persist_embedding_corpus import (
    is_up_to_date,
)

from rag.embeddings.schemas import EmbeddingRecord
from rag.embeddings.serializer import EmbeddingSerializer

def create_chunk(
    chunk_id: str = "chunk_1",
    text: str = "Enterprise retrieval systems",
) -> Chunk:

    return Chunk(
        chunk_id=chunk_id,
        document_id="document_1",
        tenant_id="tenant_1",
        department="engineering",

        content_hash="content_hash_1",
        source_filename="document.txt",
        source_file_type=".txt",
        source_s3_key=(
            "tenant_1/engineering/document.txt"
        ),

        text=text,
        chunk_index=0,
        chunk_type="text",
        strategy="recursive",
        chunking_version="1.0",

        character_count=len(text),
        token_count=len(text.split()),

        page_numbers=[],
        table_ids=[],

        metadata={},
    )


def create_config() -> EmbeddingConfig:

    return EmbeddingConfig()


def persist_artifacts(
    tmp_path,
    chunks,
    config,
):

    embedding_file = (
        tmp_path / "embeddings.jsonl"
    )

    manifest_file = (
        tmp_path / "embedding_manifest.json"
    )

    records = tuple(
        EmbeddingRecord(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            tenant_id=chunk.tenant_id,
            department=chunk.department,
            vector=(0.1,) * 384,
            model_name=config.model_name,
            embedding_version=config.embedding_version,
            metadata={},
        )
        for chunk in chunks
    )

    EmbeddingSerializer.write_jsonl(
        records,
        embedding_file,
    )

    manifest = create_embedding_manifest(
        chunks=chunks,
        config=config,
        dimension=384,
        total_embeddings=len(chunks),
    )

    write_manifest(
        manifest,
        manifest_file,
    )

    return (
        embedding_file,
        manifest_file,
        manifest,
    )


def test_manifest_survives_round_trip(
    tmp_path,
):

    chunks = (
        create_chunk(),
    )

    config = create_config()

    manifest = create_embedding_manifest(
        chunks=chunks,
        config=config,
        dimension=384,
        total_embeddings=1,
    )

    file_path = (
        tmp_path
        / "embedding_manifest.json"
    )

    write_manifest(
        manifest,
        file_path,
    )

    loaded_manifest = load_manifest(
        file_path
    )

    assert loaded_manifest == manifest


def test_chunk_hash_is_deterministic():

    chunks = (
        create_chunk(),
    )

    first_hash = calculate_chunk_hash(
        chunks
    )

    second_hash = calculate_chunk_hash(
        chunks
    )

    assert first_hash == second_hash


def test_chunk_text_change_changes_hash():

    original_chunks = (
        create_chunk(
            text="Original chunk text"
        ),
    )

    changed_chunks = (
        create_chunk(
            text="Changed chunk text"
        ),
    )

    assert calculate_chunk_hash(
        original_chunks
    ) != calculate_chunk_hash(
        changed_chunks
    )


def test_matching_artifacts_are_up_to_date(
    tmp_path,
):

    chunks = (
        create_chunk(),
    )

    config = create_config()

    (
        embedding_file,
        manifest_file,
        _,
    ) = persist_artifacts(
        tmp_path=tmp_path,
        chunks=chunks,
        config=config,
    )

    assert is_up_to_date(
        chunks=chunks,
        config=config,
        embedding_file=embedding_file,
        manifest_file=manifest_file,
    )


def test_changed_chunks_are_not_up_to_date(
    tmp_path,
):

    original_chunks = (
        create_chunk(
            text="Original text"
        ),
    )

    changed_chunks = (
        create_chunk(
            text="Changed text"
        ),
    )

    config = create_config()

    (
        embedding_file,
        manifest_file,
        _,
    ) = persist_artifacts(
        tmp_path=tmp_path,
        chunks=original_chunks,
        config=config,
    )

    assert not is_up_to_date(
        chunks=changed_chunks,
        config=config,
        embedding_file=embedding_file,
        manifest_file=manifest_file,
    )


def test_changed_embedding_version_is_not_up_to_date(
    tmp_path,
):

    chunks = (
        create_chunk(),
    )

    config = create_config()

    (
        embedding_file,
        manifest_file,
        _,
    ) = persist_artifacts(
        tmp_path=tmp_path,
        chunks=chunks,
        config=config,
    )

    changed_config = replace(
        config,
        embedding_version="2.0",
    )

    assert not is_up_to_date(
        chunks=chunks,
        config=changed_config,
        embedding_file=embedding_file,
        manifest_file=manifest_file,
    )


def test_changed_model_is_not_up_to_date(
    tmp_path,
):

    chunks = (
        create_chunk(),
    )

    config = create_config()

    (
        embedding_file,
        manifest_file,
        _,
    ) = persist_artifacts(
        tmp_path=tmp_path,
        chunks=chunks,
        config=config,
    )

    changed_config = replace(
        config,
        model_name="different-model",
    )

    assert not is_up_to_date(
        chunks=chunks,
        config=changed_config,
        embedding_file=embedding_file,
        manifest_file=manifest_file,
    )


def test_changed_normalization_is_not_up_to_date(
    tmp_path,
):

    chunks = (
        create_chunk(),
    )

    config = create_config()

    (
        embedding_file,
        manifest_file,
        _,
    ) = persist_artifacts(
        tmp_path=tmp_path,
        chunks=chunks,
        config=config,
    )

    changed_config = replace(
        config,
        normalize_embeddings=(
            not config.normalize_embeddings
        ),
    )

    assert not is_up_to_date(
        chunks=chunks,
        config=changed_config,
        embedding_file=embedding_file,
        manifest_file=manifest_file,
    )


def test_missing_embedding_file_is_not_up_to_date(
    tmp_path,
):

    chunks = (
        create_chunk(),
    )

    config = create_config()

    embedding_file = (
        tmp_path / "embeddings.jsonl"
    )

    manifest_file = (
        tmp_path / "embedding_manifest.json"
    )

    manifest = create_embedding_manifest(
        chunks=chunks,
        config=config,
        dimension=384,
        total_embeddings=1,
    )

    write_manifest(
        manifest,
        manifest_file,
    )

    assert not is_up_to_date(
        chunks=chunks,
        config=config,
        embedding_file=embedding_file,
        manifest_file=manifest_file,
    )


def test_missing_manifest_is_not_up_to_date(
    tmp_path,
):

    chunks = (
        create_chunk(),
    )

    config = create_config()

    embedding_file = (
        tmp_path / "embeddings.jsonl"
    )

    manifest_file = (
        tmp_path / "embedding_manifest.json"
    )

    embedding_file.write_text(
        '{"dummy": true}\n',
        encoding="utf-8",
    )

    assert not is_up_to_date(
        chunks=chunks,
        config=config,
        embedding_file=embedding_file,
        manifest_file=manifest_file,
    )


def test_corrupt_manifest_is_not_up_to_date(
    tmp_path,
):

    chunks = (
        create_chunk(),
    )

    config = create_config()

    embedding_file = (
        tmp_path / "embeddings.jsonl"
    )

    manifest_file = (
        tmp_path / "embedding_manifest.json"
    )

    embedding_file.write_text(
        '{"dummy": true}\n',
        encoding="utf-8",
    )

    manifest_file.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    assert not is_up_to_date(
        chunks=chunks,
        config=config,
        embedding_file=embedding_file,
        manifest_file=manifest_file,
    )