import pytest

from rag.vector_store.config import VectorDBConfig
from rag.vector_store.qdrant_store import (
    QdrantVectorStore,
)

from dataclasses import replace

from rag.chunking.schemas import Chunk
from rag.embeddings.schemas import EmbeddingRecord


def create_chunk(
    chunk_id="a" * 64,
    text="Enterprise retrieval system",
):

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


def create_record(
    chunk_id="a" * 64,
    dimension=384,
):

    return EmbeddingRecord(
        chunk_id=chunk_id,
        document_id="document_1",
        tenant_id="tenant_1",
        department="engineering",
        vector=(0.1,) * dimension,
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        embedding_version="1.0",
        metadata={},
    )


def test_build_points_creates_valid_point(
    vector_store,
):

    chunk = create_chunk()
    record = create_record()

    points = vector_store.build_points(
        chunks=(chunk,),
        records=(record,),
    )

    assert len(points) == 1

    point = points[0]

    assert point.id == (
        vector_store._point_id(
            chunk.chunk_id
        )
    )

    assert len(point.vector) == 384

    assert (
        point.payload["chunk_id"]
        == chunk.chunk_id
    )

    assert (
        point.payload["tenant_id"]
        == chunk.tenant_id
    )

    assert (
        point.payload["department"]
        == chunk.department
    )

    assert (
        point.payload["text"]
        == chunk.text
    )


def test_build_points_rejects_count_mismatch(
    vector_store,
):

    with pytest.raises(
        ValueError,
        match="counts do not match",
    ):
        vector_store.build_points(
            chunks=(create_chunk(),),
            records=(),
        )


def test_build_points_rejects_id_mismatch(
    vector_store,
):

    chunk = create_chunk(
        chunk_id="a" * 64
    )

    record = create_record(
        chunk_id="b" * 64
    )

    with pytest.raises(
        ValueError,
        match="IDs do not match",
    ):
        vector_store.build_points(
            chunks=(chunk,),
            records=(record,),
        )


def test_build_points_rejects_wrong_dimension(
    vector_store,
):

    chunk = create_chunk()

    record = create_record(
        dimension=2
    )

    with pytest.raises(
        ValueError,
        match="dimension does not match",
    ):
        vector_store.build_points(
            chunks=(chunk,),
            records=(record,),
        )


def test_points_can_be_upserted(
    vector_store,
):

    chunk = create_chunk()
    record = create_record()

    points = vector_store.build_points(
        chunks=(chunk,),
        records=(record,),
    )

    count = vector_store.upsert_points(
        points
    )

    assert count == 1

    info = vector_store.get_collection_info()

    assert info.points_count == 1

def test_search_returns_similar_point(
    vector_store,
):

    chunk = create_chunk()
    record = create_record()

    points = vector_store.build_points(
        chunks=(chunk,),
        records=(record,),
    )

    vector_store.upsert_points(points)

    results = vector_store.search(
        query_vector=record.vector,
        limit=1,
    )

    assert len(results) == 1

    assert (
        results[0].payload["chunk_id"]
        == chunk.chunk_id
    )

    assert results[0].score > 0.99


def test_search_rejects_wrong_dimension(
    vector_store,
):

    vector_store.create_collection()

    with pytest.raises(
        ValueError,
        match="dimension does not match",
    ):
        vector_store.search(
            query_vector=(0.1, 0.2),
            limit=5,
        )


def test_search_rejects_invalid_limit(
    vector_store,
):

    vector_store.create_collection()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        vector_store.search(
            query_vector=(0.1,) * 384,
            limit=0,
        )

def test_search_filters_by_tenant(
    vector_store,
):

    chunk_1 = create_chunk(
        chunk_id="a" * 64,
    )

    chunk_2 = create_chunk(
        chunk_id="b" * 64,
    )

    chunk_2 = replace(
        chunk_2,
        tenant_id="tenant_2",
    )

    record_1 = create_record(
        chunk_id=chunk_1.chunk_id,
    )

    record_2 = create_record(
        chunk_id=chunk_2.chunk_id,
    )

    points = vector_store.build_points(
        chunks=(chunk_1, chunk_2),
        records=(record_1, record_2),
    )

    vector_store.upsert_points(points)

    results = vector_store.search(
        query_vector=record_1.vector,
        limit=5,
        tenant_id="tenant_1",
    )

    assert len(results) == 1

    assert (
        results[0].payload["tenant_id"]
        == "tenant_1"
    )


def test_search_filters_by_department(
    vector_store,
):

    chunk_1 = create_chunk(
        chunk_id="a" * 64,
    )

    chunk_2 = replace(
        create_chunk(
            chunk_id="b" * 64,
        ),
        department="finance",
    )

    record_1 = create_record(
        chunk_id=chunk_1.chunk_id,
    )

    record_2 = create_record(
        chunk_id=chunk_2.chunk_id,
    )

    points = vector_store.build_points(
        chunks=(chunk_1, chunk_2),
        records=(record_1, record_2),
    )

    vector_store.upsert_points(points)

    results = vector_store.search(
        query_vector=record_1.vector,
        limit=5,
        department="engineering",
    )

    assert len(results) == 1

    assert (
        results[0].payload["department"]
        == "engineering"
    )


def test_search_combines_tenant_and_department_filters(
    vector_store,
):

    chunk_1 = create_chunk(
        chunk_id="a" * 64,
    )

    chunk_2 = replace(
        create_chunk(
            chunk_id="b" * 64,
        ),
        department="finance",
    )

    record_1 = create_record(
        chunk_id=chunk_1.chunk_id,
    )

    record_2 = create_record(
        chunk_id=chunk_2.chunk_id,
    )

    points = vector_store.build_points(
        chunks=(chunk_1, chunk_2),
        records=(record_1, record_2),
    )

    vector_store.upsert_points(points)

    results = vector_store.search(
        query_vector=record_1.vector,
        limit=5,
        tenant_id="tenant_1",
        department="finance",
    )

    assert len(results) == 1

    assert (
        results[0].payload["tenant_id"]
        == "tenant_1"
    )

    assert (
        results[0].payload["department"]
        == "finance"
    )

def test_search_applies_score_threshold(
    vector_store,
):

    chunk = create_chunk()
    record = create_record()

    points = vector_store.build_points(
        chunks=(chunk,),
        records=(record,),
    )

    vector_store.upsert_points(points)

    results = vector_store.search(
        query_vector=record.vector,
        limit=5,
        score_threshold=0.99,
    )

    assert len(results) == 1

    results = vector_store.search(
        query_vector=record.vector,
        limit=5,
        score_threshold=1.01,
    )

    assert results == []


def test_search_without_threshold_still_works(
    vector_store,
):

    chunk = create_chunk()
    record = create_record()

    points = vector_store.build_points(
        chunks=(chunk,),
        records=(record,),
    )

    vector_store.upsert_points(points)

    results = vector_store.search(
        query_vector=record.vector,
        limit=1,
    )

    assert len(results) == 1

 
@pytest.fixture
def vector_store(tmp_path):

    config = VectorDBConfig(
        storage_path=str(tmp_path / "qdrant")
    )

    store = QdrantVectorStore(config)

    yield store

    store.close()


def test_collection_does_not_exist_initially(
    vector_store,
):

    assert not vector_store.collection_exists()


def test_collection_can_be_created(
    vector_store,
):

    created = vector_store.create_collection()

    assert created
    assert vector_store.collection_exists()


def test_collection_creation_is_idempotent(
    vector_store,
):

    first_result = (
        vector_store.create_collection()
    )

    second_result = (
        vector_store.create_collection()
    )

    assert first_result
    assert not second_result


def test_collection_has_correct_vector_size(
    vector_store,
):

    vector_store.create_collection()

    info = vector_store.get_collection_info()

    assert (
        info.config.params.vectors.size
        == vector_store.config.vector_size
    )


def test_missing_collection_info_raises_error(
    vector_store,
):

    with pytest.raises(
        ValueError,
        match="Vector collection does not exist",
    ):
        vector_store.get_collection_info()


def test_unsupported_distance_raises_error(
    tmp_path,
):

    config = VectorDBConfig(
        storage_path=str(tmp_path / "qdrant"),
        distance="invalid",
    )

    store = QdrantVectorStore(config)

    try:

        with pytest.raises(
            ValueError,
            match="Unsupported distance",
        ):
            store.create_collection()

    finally:
        store.close()