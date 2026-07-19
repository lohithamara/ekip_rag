from pathlib import Path

from rag.chunking.storage.serializer import (
    ChunkSerializer,
)
from rag.sparse_retrieval.bm25_index import BM25Index
from rag.sparse_retrieval.config import (
    SparseRetrievalConfig,
)
from rag.sparse_retrieval.manifest import (
    build_manifest,
    load_manifest,
)


CHUNK_ROOT = Path("data/processed/chunks")


def main():

    config = SparseRetrievalConfig()

    errors = []

    chunk_files = sorted(
        CHUNK_ROOT.rglob("chunks.jsonl")
    )

    chunks = []

    for chunk_file in chunk_files:

        chunks.extend(
            ChunkSerializer.load_jsonl(
                chunk_file
            )
        )

    index_path = Path(config.index_path)

    manifest_path = Path(
        config.manifest_path
    )

    if not index_path.is_file():
        errors.append(
            "Sparse index file is missing."
        )

    if not manifest_path.is_file():
        errors.append(
            "Sparse manifest file is missing."
        )

    loaded_index = None
    saved_manifest = None

    if index_path.is_file():

        try:
            loaded_index = BM25Index.load(config)

        except Exception as exc:
            errors.append(
                f"Unable to load sparse index: {exc}"
            )

    if manifest_path.is_file():

        try:
            saved_manifest = load_manifest(
                manifest_path
            )

        except Exception as exc:
            errors.append(
                f"Unable to load sparse manifest: {exc}"
            )

    expected_manifest = build_manifest(
        chunks=chunks,
        config=config,
    )

    if (
        loaded_index is not None
        and len(loaded_index.chunks)
        != len(chunks)
    ):
        errors.append(
            "Sparse index chunk count mismatch."
        )

    if (
        saved_manifest is not None
        and saved_manifest != expected_manifest
    ):
        errors.append(
            "Sparse manifest does not match "
            "the current corpus or configuration."
        )

    print()
    print("===================================")
    print("SPARSE CORPUS VALIDATION SUMMARY")
    print("===================================")

    print(
        f"Documents checked    : "
        f"{len(chunk_files)}"
    )

    print(
        f"Current chunks       : "
        f"{len(chunks)}"
    )

    print(
        f"Indexed chunks       : "
        f"{len(loaded_index.chunks) if loaded_index else 0}"
    )

    print(
        f"Tokenizer version    : "
        f"{config.tokenizer_version}"
    )

    print(
        f"BM25 version         : "
        f"{config.bm25_version}"
    )

    print(
        f"Errors               : "
        f"{len(errors)}"
    )

    if errors:

        print()
        print("ERROR DETAILS")
        print("-----------------------------------")

        for error in errors:
            print(error)

    print()
    print("===================================")
    print("FINAL STATUS")
    print("===================================")

    if errors:
        print(
            "FAIL: sparse corpus is invalid."
        )
        raise SystemExit(1)

    print(
        "PASS: sparse corpus is valid."
    )


if __name__ == "__main__":
    main()