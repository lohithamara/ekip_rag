from pathlib import Path

from rag.chunking.storage.serializer import (
    ChunkSerializer,
)
from rag.sparse_retrieval.bm25_index import BM25Index
from rag.sparse_retrieval.config import (
    SparseRetrievalConfig,
)

from pathlib import Path

from rag.sparse_retrieval.manifest import (
    build_manifest,
    save_manifest,
)


CHUNK_ROOT = Path("data/processed/chunks")


def main():

    chunk_files = sorted(
        CHUNK_ROOT.rglob("chunks.jsonl")
    )

    if not chunk_files:
        raise RuntimeError(
            "No chunk files found."
        )

    chunks = []
    failed = 0

    print()
    print("===================================")
    print("SPARSE CORPUS INDEXING")
    print("===================================")

    for index, chunk_file in enumerate(
        chunk_files,
        start=1,
    ):

        try:

            loaded_chunks = (
                ChunkSerializer.load_jsonl(
                    chunk_file
                )
            )

            chunks.extend(loaded_chunks)

            print(
                f"[{index}/{len(chunk_files)}] "
                f"LOADED "
                f"{chunk_file.parent.relative_to(CHUNK_ROOT)} "
                f"({len(loaded_chunks)} chunks)"
            )

        except Exception as exc:

            failed += 1

            print(
                f"[{index}/{len(chunk_files)}] "
                f"FAIL "
                f"{chunk_file}: {exc}"
            )

    if failed:
        print()
        print("FINAL STATUS: FAIL")
        raise SystemExit(1)

    config = SparseRetrievalConfig()

    bm25_index = BM25Index(config)

    indexed_count = bm25_index.build(chunks)

    bm25_index.save()
    manifest = build_manifest(
        chunks=chunks,
        config=config,
    )

    save_manifest(
        manifest=manifest,
        path=Path(config.manifest_path),
    )
    print()
    print("===================================")
    print("SPARSE INDEXING SUMMARY")
    print("===================================")

    print(
        f"Documents discovered : "
        f"{len(chunk_files)}"
    )

    print(
        f"Chunks indexed        : "
        f"{indexed_count}"
    )

    print(
        f"Index path            : "
        f"{config.index_path}"
    )

    print(
        f"Tokenizer version     : "
        f"{config.tokenizer_version}"
    )

    print(
        f"BM25 version          : "
        f"{config.bm25_version}"
    )

    print(
        f"Failed                : "
        f"{failed}"
    )

    print(
        f"Manifest path         : "
        f"{config.manifest_path}"
    )
    print()
    print("FINAL STATUS: PASS")


if __name__ == "__main__":
    main()