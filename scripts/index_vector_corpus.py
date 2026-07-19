from pathlib import Path

from rag.chunking.storage.serializer import (
    ChunkSerializer,
)
from rag.embeddings.serializer import (
    EmbeddingSerializer,
)
from rag.vector_store.config import VectorDBConfig
from rag.vector_store.qdrant_store import (
    QdrantVectorStore,
)


CHUNK_ROOT = Path("data/processed/chunks")

EMBEDDING_ROOT = Path(
    "data/processed/embeddings"
)


def main():

    chunk_files = sorted(
        CHUNK_ROOT.rglob("chunks.jsonl")
    )

    if not chunk_files:
        raise RuntimeError(
            "No chunk files found."
        )

    store = QdrantVectorStore(
        VectorDBConfig()
    )

    documents_indexed = 0
    points_indexed = 0
    failed = 0

    current_chunk_ids = set()

    print()
    print("===================================")
    print("VECTOR CORPUS INDEXING")
    print("===================================")

    try:

        store.recreate_collection()

        for index, chunk_file in enumerate(
            chunk_files,
            start=1,
        ):

            relative_dir = (
                chunk_file.parent.relative_to(
                    CHUNK_ROOT
                )
            )

            embedding_file = (
                EMBEDDING_ROOT
                / relative_dir
                / "embeddings.jsonl"
            )

            try:

                if not embedding_file.is_file():
                    raise FileNotFoundError(
                        "Embedding file not found."
                    )

                chunks = tuple(
                    ChunkSerializer.load_jsonl(
                        chunk_file
                    )
                )

                records = tuple(
                    EmbeddingSerializer.load_jsonl(
                        embedding_file
                    )
                )

                points = store.build_points(
                    chunks=chunks,
                    records=records,
                )

                indexed_count = (
                    store.upsert_points(points)
                )

                current_chunk_ids.update(
                    chunk.chunk_id
                    for chunk in chunks
                )

                documents_indexed += 1

                points_indexed += indexed_count

                print(
                    f"[{index}/{len(chunk_files)}] "
                    f"INDEXED "
                    f"{relative_dir} "
                    f"({indexed_count} points)"
                )

            except Exception as exc:

                failed += 1

                print(
                    f"[{index}/{len(chunk_files)}] "
                    f"FAIL "
                    f"{relative_dir}: {exc}"
                )

        if failed == 0:

            stale_deleted = (
                store.delete_stale_points(
                    current_chunk_ids
                )
            )

        else:

            stale_deleted = 0

        info = store.get_collection_info()

        print()
        print("===================================")
        print("VECTOR INDEXING SUMMARY")
        print("===================================")

        print(
            f"Documents discovered : "
            f"{len(chunk_files)}"
        )

        print(
            f"Documents indexed    : "
            f"{documents_indexed}"
        )

        print(
            f"Points upserted       : "
            f"{points_indexed}"
        )

        print(
            f"Stale points deleted : "
            f"{stale_deleted}"
        )

        print(
            f"Collection points     : "
            f"{info.points_count}"
        )

        print(
            f"Failed                : "
            f"{failed}"
        )

        print()

        if failed:

            print("FINAL STATUS: FAIL")
            raise SystemExit(1)

        print("FINAL STATUS: PASS")

    finally:

        store.close()


if __name__ == "__main__":
    main()