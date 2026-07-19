from dataclasses import replace

from rag.chunking.schemas import Chunk
from rag.sparse_retrieval.config import (
    SparseRetrievalConfig,
)
from rag.sparse_retrieval.manifest import (
    build_manifest,
    compute_corpus_hash,
    load_manifest,
    save_manifest,
)


def create_chunk(
    chunk_id: str = "a" * 64,
    text: str = "Enterprise retrieval system",
    content_hash: str = "hash_1",
) -> Chunk:

    return Chunk(
        chunk_id=chunk_id,
        document_id="document_1",
        tenant_id="tenant_1",
        department="engineering",
        content_hash=content_hash,
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


def test_corpus_hash_is_deterministic():

    chunks = [
        create_chunk(),
        create_chunk(
            chunk_id="b" * 64,
            content_hash="hash_2",
        ),
    ]

    first_hash = compute_corpus_hash(chunks)
    second_hash = compute_corpus_hash(chunks)

    assert first_hash == second_hash


def test_corpus_hash_ignores_chunk_order():

    chunk_1 = create_chunk()

    chunk_2 = create_chunk(
        chunk_id="b" * 64,
        content_hash="hash_2",
    )

    first_hash = compute_corpus_hash(
        [chunk_1, chunk_2]
    )

    second_hash = compute_corpus_hash(
        [chunk_2, chunk_1]
    )

    assert first_hash == second_hash


def test_corpus_hash_changes_when_content_changes():

    chunk = create_chunk()

    changed_chunk = replace(
        chunk,
        content_hash="changed_hash",
    )

    assert (
        compute_corpus_hash([chunk])
        != compute_corpus_hash(
            [changed_chunk]
        )
    )


def test_manifest_changes_when_config_changes():

    chunks = [create_chunk()]

    config = SparseRetrievalConfig()

    manifest = build_manifest(
        chunks=chunks,
        config=config,
    )

    changed_config = replace(
        config,
        tokenizer_version="2.0",
    )

    changed_manifest = build_manifest(
        chunks=chunks,
        config=changed_config,
    )

    assert manifest != changed_manifest


def test_manifest_survives_json_round_trip(
    tmp_path,
):

    chunks = [create_chunk()]

    config = SparseRetrievalConfig()

    manifest = build_manifest(
        chunks=chunks,
        config=config,
    )

    path = tmp_path / "manifest.json"

    save_manifest(
        manifest=manifest,
        path=path,
    )

    loaded_manifest = load_manifest(path)

    assert loaded_manifest == manifest


def test_manifest_has_correct_chunk_count():

    chunks = [
        create_chunk(),
        create_chunk(
            chunk_id="b" * 64,
            content_hash="hash_2",
        ),
    ]

    manifest = build_manifest(
        chunks=chunks,
        config=SparseRetrievalConfig(),
    )

    assert manifest.chunk_count == 2