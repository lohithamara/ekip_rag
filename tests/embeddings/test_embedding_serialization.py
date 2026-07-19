from rag.embeddings.schemas import EmbeddingRecord
from rag.embeddings.serializer import EmbeddingSerializer


def create_record(
    chunk_id: str,
    vector: tuple[float, ...],
) -> EmbeddingRecord:

    return EmbeddingRecord(
        chunk_id=chunk_id,
        document_id="document_1",
        tenant_id="tenant_1",
        department="engineering",
        vector=vector,
        model_name="fake-model",
        embedding_version="1.0",
        metadata={
            "chunk_type": "text",
            "page_numbers": [1],
        },
    )


def test_embeddings_survive_serialization_round_trip(
    tmp_path,
):
    records = (
        create_record(
            "chunk_1",
            (0.1, 0.2, 0.3),
        ),
        create_record(
            "chunk_2",
            (0.4, 0.5, 0.6),
        ),
    )

    file_path = tmp_path / "embeddings.jsonl"

    EmbeddingSerializer.write_jsonl(
        records,
        file_path,
    )

    loaded_records = (
        EmbeddingSerializer.load_jsonl(
            file_path
        )
    )

    assert loaded_records == records


def test_serialization_preserves_record_order(
    tmp_path,
):
    records = (
        create_record(
            "chunk_1",
            (0.1, 0.2),
        ),
        create_record(
            "chunk_2",
            (0.3, 0.4),
        ),
        create_record(
            "chunk_3",
            (0.5, 0.6),
        ),
    )

    file_path = tmp_path / "embeddings.jsonl"

    EmbeddingSerializer.write_jsonl(
        records,
        file_path,
    )

    loaded_records = (
        EmbeddingSerializer.load_jsonl(
            file_path
        )
    )

    assert [
        record.chunk_id
        for record in loaded_records
    ] == [
        "chunk_1",
        "chunk_2",
        "chunk_3",
    ]


def test_empty_collection_creates_empty_file(
    tmp_path,
):
    file_path = tmp_path / "embeddings.jsonl"

    EmbeddingSerializer.write_jsonl(
        (),
        file_path,
    )

    loaded_records = (
        EmbeddingSerializer.load_jsonl(
            file_path
        )
    )

    assert file_path.exists()
    assert loaded_records == ()