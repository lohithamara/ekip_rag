from math import isfinite
from pathlib import Path

from rag.chunking.storage.serializer import ChunkSerializer
from rag.embeddings.serializer import EmbeddingSerializer


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
            "No persisted chunk files found."
        )

    documents_checked = 0
    documents_valid = 0
    documents_invalid = 0

    total_chunks = 0
    total_embeddings = 0

    global_chunk_ids = set()

    expected_dimension = None
    expected_model_name = None
    expected_embedding_version = None

    issues = []

    print()
    print("===================================")
    print("EMBEDDING CORPUS VALIDATION")
    print("===================================")

    for chunk_file in chunk_files:

        documents_checked += 1

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

        document_issues = []

        if not embedding_file.is_file():

            document_issues.append(
                "Missing embeddings.jsonl."
            )

            issues.append(
                (
                    str(embedding_file),
                    "Missing embeddings.jsonl.",
                )
            )

            documents_invalid += 1

            continue

        try:

            chunks = tuple(
                ChunkSerializer.load_jsonl(
                    chunk_file
                )
            )

            records = (
                EmbeddingSerializer.load_jsonl(
                    embedding_file
                )
            )

        except Exception as exc:

            message = (
                f"Unable to load artifacts: {exc}"
            )

            document_issues.append(message)

            issues.append(
                (
                    str(embedding_file),
                    message,
                )
            )

            documents_invalid += 1

            continue

        total_chunks += len(chunks)
        total_embeddings += len(records)

        if not records:

            document_issues.append(
                "Embedding file contains no records."
            )

        if len(chunks) != len(records):

            document_issues.append(
                "Chunk/embedding count mismatch."
            )

        chunk_ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        record_ids = [
            record.chunk_id
            for record in records
        ]

        if chunk_ids != record_ids:

            document_issues.append(
                "Chunk IDs or order do not match."
            )

        if len(record_ids) != len(
            set(record_ids)
        ):

            document_issues.append(
                "Duplicate embedding IDs "
                "within document."
            )

        for chunk, record in zip(
            chunks,
            records,
        ):

            if record.document_id != chunk.document_id:

                document_issues.append(
                    f"{record.chunk_id}: "
                    "document_id mismatch."
                )

            if record.tenant_id != chunk.tenant_id:

                document_issues.append(
                    f"{record.chunk_id}: "
                    "tenant_id mismatch."
                )

            if record.department != chunk.department:

                document_issues.append(
                    f"{record.chunk_id}: "
                    "department mismatch."
                )

            if not record.vector:

                document_issues.append(
                    f"{record.chunk_id}: "
                    "empty embedding vector."
                )

                continue

            dimension = len(record.vector)

            if expected_dimension is None:
                expected_dimension = dimension

            elif dimension != expected_dimension:

                document_issues.append(
                    f"{record.chunk_id}: "
                    "embedding dimension mismatch."
                )

            if expected_model_name is None:

                expected_model_name = (
                    record.model_name
                )

            elif (
                record.model_name
                != expected_model_name
            ):

                document_issues.append(
                    f"{record.chunk_id}: "
                    "model name mismatch."
                )

            if expected_embedding_version is None:

                expected_embedding_version = (
                    record.embedding_version
                )

            elif (
                record.embedding_version
                != expected_embedding_version
            ):

                document_issues.append(
                    f"{record.chunk_id}: "
                    "embedding version mismatch."
                )

            if not all(
                isfinite(value)
                for value in record.vector
            ):

                document_issues.append(
                    f"{record.chunk_id}: "
                    "non-finite vector value."
                )

            if record.chunk_id in global_chunk_ids:

                document_issues.append(
                    f"{record.chunk_id}: "
                    "duplicate global embedding ID."
                )

            else:

                global_chunk_ids.add(
                    record.chunk_id
                )

        if document_issues:

            documents_invalid += 1

            for message in document_issues:

                issues.append(
                    (
                        str(embedding_file),
                        message,
                    )
                )

        else:

            documents_valid += 1

    print()
    print("===================================")
    print("EMBEDDING CORPUS VALIDATION SUMMARY")
    print("===================================")

    print(
        f"Documents checked    : "
        f"{documents_checked}"
    )

    print(
        f"Valid documents      : "
        f"{documents_valid}"
    )

    print(
        f"Invalid documents    : "
        f"{documents_invalid}"
    )

    print(
        f"Total chunks         : "
        f"{total_chunks}"
    )

    print(
        f"Total embeddings     : "
        f"{total_embeddings}"
    )

    print(
        f"Unique global IDs    : "
        f"{len(global_chunk_ids)}"
    )

    print(
        f"Embedding dimension  : "
        f"{expected_dimension}"
    )

    print(
        f"Model name           : "
        f"{expected_model_name}"
    )

    print(
        f"Embedding version    : "
        f"{expected_embedding_version}"
    )

    print(
        f"Issues               : "
        f"{len(issues)}"
    )

    if issues:

        print()
        print("ISSUE DETAILS")
        print("-----------------------------------")

        for index, (path, message) in enumerate(
            issues,
            start=1,
        ):

            print()
            print(f"ISSUE #{index}")
            print(f"Path    : {path}")
            print(f"Message : {message}")

    print()
    print("===================================")
    print("FINAL STATUS")
    print("===================================")

    if issues:

        print(
            f"FAIL: {len(issues)} issues detected."
        )

        raise SystemExit(1)

    print("PASS: embedding corpus is valid.")


if __name__ == "__main__":
    main()