import pytest
from dataclasses import replace
from rag.chunking.schemas import Chunk
from rag.sparse_retrieval.bm25_index import BM25Index
from rag.sparse_retrieval.config import (
    SparseRetrievalConfig,
)


def create_chunk(
    chunk_id: str,
    text: str,
) -> Chunk:

    return Chunk(
        chunk_id=chunk_id,
        document_id="document_1",
        tenant_id="tenant_1",
        department="engineering",
        content_hash="content_hash",
        source_filename="document.txt",
        source_file_type=".txt",
        source_s3_key="tenant_1/engineering/document.txt",
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


@pytest.fixture
def bm25_index():

    index = BM25Index(
        SparseRetrievalConfig()
    )

    index.build(
        [
            create_chunk(
                "chunk_1",
                "refund policy standard customers",
            ),
            create_chunk(
                "chunk_2",
                "employee benefits enrollment guide",
            ),
            create_chunk(
                "chunk_3",
                "api endpoint document ingestion",
            ),
        ]
    )

    return index


def test_build_returns_chunk_count():

    index = BM25Index(
        SparseRetrievalConfig()
    )

    count = index.build(
        [
            create_chunk(
                "chunk_1",
                "refund policy",
            )
        ]
    )

    assert count == 1


def test_search_returns_relevant_chunk(
    bm25_index,
):

    results = bm25_index.search(
        query="refund policy",
        limit=2,
    )

    assert results

    assert (
        results[0][0].chunk_id
        == "chunk_1"
    )


def test_search_results_are_ranked(
    bm25_index,
):

    results = bm25_index.search(
        query="employee benefits enrollment",
        limit=3,
    )

    scores = [
        score
        for _, score in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_search_returns_empty_for_no_match(
    bm25_index,
):

    results = bm25_index.search(
        query="quantum astrophysics",
    )

    assert results == []


def test_search_requires_built_index():

    index = BM25Index(
        SparseRetrievalConfig()
    )

    with pytest.raises(
        ValueError,
        match="has not been built",
    ):
        index.search("refund")


def test_search_requires_query(
    bm25_index,
):

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        bm25_index.search(" ")


def test_search_requires_positive_limit(
    bm25_index,
):

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        bm25_index.search(
            query="refund",
            limit=0,
        )

def test_search_filters_by_tenant():

    index = BM25Index(
        SparseRetrievalConfig()
    )

    chunk_1 = create_chunk(
        "chunk_1",
        "refund policy standard customers",
    )

    chunk_2 = create_chunk(
        "chunk_2",
        "refund policy enterprise customers",
    )

    chunk_2 = replace(
        chunk_2,
        tenant_id="tenant_2",
    )

    index.build(
        [chunk_1, chunk_2]
    )

    results = index.search(
        query="refund policy",
        tenant_id="tenant_1",
    )

    assert len(results) == 1

    assert (
        results[0][0].tenant_id
        == "tenant_1"
    )


def test_search_filters_by_department():

    index = BM25Index(
        SparseRetrievalConfig()
    )

    chunk_1 = create_chunk(
        "chunk_1",
        "api endpoint document ingestion",
    )

    chunk_2 = replace(
        create_chunk(
            "chunk_2",
            "api endpoint document retrieval",
        ),
        department="finance",
    )

    index.build(
        [chunk_1, chunk_2]
    )

    results = index.search(
        query="api endpoint document",
        department="engineering",
    )

    assert len(results) == 1

    assert (
        results[0][0].department
        == "engineering"
    )


def test_search_combines_tenant_and_department_filters():

    index = BM25Index(
        SparseRetrievalConfig()
    )

    chunk_1 = create_chunk(
        "chunk_1",
        "employee benefits enrollment",
    )

    chunk_2 = replace(
        create_chunk(
            "chunk_2",
            "employee benefits policy",
        ),
        department="hr",
    )

    chunk_3 = replace(
        create_chunk(
            "chunk_3",
            "employee benefits guide",
        ),
        tenant_id="tenant_2",
        department="hr",
    )

    index.build(
        [
            chunk_1,
            chunk_2,
            chunk_3,
        ]
    )

    results = index.search(
        query="employee benefits",
        tenant_id="tenant_1",
        department="hr",
    )

    assert len(results) == 1

    assert (
        results[0][0].chunk_id
        == "chunk_2"
    )

def test_index_survives_save_load_round_trip(
    tmp_path,
):

    config = SparseRetrievalConfig(
        index_path=str(
            tmp_path / "bm25.pkl"
        )
    )

    index = BM25Index(config)

    index.build(
        [
            create_chunk(
                "chunk_1",
                "refund policy standard customers",
            ),
            create_chunk(
                "chunk_2",
                "employee benefits enrollment",
            ),
            create_chunk(
                "chunk_3",
                "api endpoint document ingestion",
            ),
        ]
    )

    index.save()

    loaded = BM25Index.load(config)

    results = loaded.search(
        query="refund standard",
        limit=1,
    )

    assert results

    assert (
        results[0][0].chunk_id
        == "chunk_1"
    )


def test_save_requires_built_index(
    tmp_path,
):

    config = SparseRetrievalConfig(
        index_path=str(
            tmp_path / "bm25.pkl"
        )
    )

    index = BM25Index(config)

    with pytest.raises(
        ValueError,
        match="has not been built",
    ):
        index.save()


def test_load_requires_existing_file(
    tmp_path,
):

    config = SparseRetrievalConfig(
        index_path=str(
            tmp_path / "missing.pkl"
        )
    )

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        BM25Index.load(config)