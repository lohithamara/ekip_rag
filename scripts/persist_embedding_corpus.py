import argparse
from pathlib import Path
from math import isfinite

from rag.chunking.storage.serializer import ChunkSerializer
from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.manifest import (
    create_embedding_manifest,
    load_manifest,
    write_manifest,
)
from rag.embeddings.model import EmbeddingModel
from rag.embeddings.service import EmbeddingService
from rag.embeddings.schemas import EmbeddingRecord
from rag.embeddings.serializer import (
    EmbeddingSerializer,
)

CHUNK_ROOT = Path("data/processed/chunks")

EMBEDDING_ROOT = Path(
    "data/processed/embeddings"
)

EMBEDDINGS_FILENAME = "embeddings.jsonl"

MANIFEST_FILENAME = "embedding_manifest.json"


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def is_up_to_date(
    chunks,
    config,
    embedding_file,
    manifest_file,
):

    if (
        not embedding_file.is_file()
        or not manifest_file.is_file()
    ):
        return False

    try:
        manifest = load_manifest(
            manifest_file
        )

        records = (
            EmbeddingSerializer.load_jsonl(
                embedding_file
            )
        )

    except Exception:
        return False

    current_manifest = (
        create_embedding_manifest(
            chunks=chunks,
            config=config,
            dimension=(
                manifest.embedding_dimension
            ),
            total_embeddings=len(chunks),
        )
    )

    if manifest != current_manifest:
        return False

    if len(records) != len(chunks):
        return False

    for chunk, record in zip(
        chunks,
        records,
    ):
        if record.chunk_id != chunk.chunk_id:
            return False

        if (
            len(record.vector)
            != manifest.embedding_dimension
        ):
            return False

        if not all(
            isfinite(value)
            for value in record.vector
        ):
            return False

    return True


def main():

    args = parse_args()

    chunk_files = sorted(
        CHUNK_ROOT.rglob("chunks.jsonl")
    )

    if not chunk_files:

        raise RuntimeError(
            "No persisted chunk files found."
        )

    config = EmbeddingConfig()

    model = EmbeddingModel(config)

    service = EmbeddingService(
        model=model,
        config=config,
    )

    written = 0
    rebuilt = 0
    skipped = 0
    failed = 0

    total_embeddings = 0

    print()
    print("===================================")
    print("EMBEDDING CORPUS PERSISTENCE")
    print("===================================")

    for index, chunk_file in enumerate(
        chunk_files,
        start=1,
    ):

        relative_dir = (
            chunk_file.parent.relative_to(
                CHUNK_ROOT
            )
        )

        output_dir = (
            EMBEDDING_ROOT
            / relative_dir
        )

        embedding_file = (
            output_dir
            / EMBEDDINGS_FILENAME
        )

        manifest_file = (
            output_dir
            / MANIFEST_FILENAME
        )

        existed_before = (
            embedding_file.exists()
            or manifest_file.exists()
        )

        try:

            chunks = tuple(
                ChunkSerializer.load_jsonl(
                    chunk_file
                )
            )

            if not chunks:

                raise ValueError(
                    "Chunk file contains no chunks."
                )

            if (
                not args.force
                and is_up_to_date(
                    chunks=chunks,
                    config=config,
                    embedding_file=embedding_file,
                    manifest_file=manifest_file,
                )
            ):

                skipped += 1

                print(
                    f"[{index}/{len(chunk_files)}] "
                    f"SKIPPED_UP_TO_DATE "
                    f"{embedding_file}"
                )

                continue

            result = service.embed_chunks(
                chunks
            )

            manifest = (
                create_embedding_manifest(
                    chunks=chunks,
                    config=config,
                    dimension=result.dimension,
                    total_embeddings=(
                        result.total_embeddings
                    ),
                )
            )

            EmbeddingSerializer.write_jsonl(
                result.records,
                embedding_file,
            )

            write_manifest(
                manifest,
                manifest_file,
            )

            total_embeddings += (
                result.total_embeddings
            )

            if existed_before:

                rebuilt += 1

                status = "REBUILT"

            else:

                written += 1

                status = "WRITTEN"

            print(
                f"[{index}/{len(chunk_files)}] "
                f"{status} "
                f"{embedding_file} "
                f"({result.total_embeddings} embeddings)"
            )

        except Exception as exc:

            failed += 1

            print(
                f"[{index}/{len(chunk_files)}] "
                f"FAIL {chunk_file}: {exc}"
            )

    print()
    print("===================================")
    print("EMBEDDING PERSISTENCE SUMMARY")
    print("===================================")

    print(
        f"Document directories : "
        f"{len(chunk_files)}"
    )

    print(
        f"Written              : "
        f"{written}"
    )

    print(
        f"Rebuilt              : "
        f"{rebuilt}"
    )

    print(
        f"Skipped up to date   : "
        f"{skipped}"
    )

    print(
        f"Failed               : "
        f"{failed}"
    )

    print(
        f"Embeddings generated : "
        f"{total_embeddings}"
    )

    print()

    if failed:

        print("FINAL STATUS: FAIL")

        raise SystemExit(1)

    print("FINAL STATUS: PASS")


if __name__ == "__main__":
    main()