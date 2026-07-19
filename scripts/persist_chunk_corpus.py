from collections import Counter
from pathlib import Path
import traceback

from ingestion.serialization.document_serializer import (
    DocumentSerializer,
)

from rag.chunking.config import ChunkingConfig

from rag.chunking.routing.profiler import (
    DocumentProfiler,
)
from rag.chunking.routing.router import (
    ChunkingRouter,
)
from rag.chunking.routing.service import (
    RoutedChunkingService,
)
from rag.chunking.service import ChunkingService
from rag.chunking.storage.service import (
    ChunkStorageService,
)
from rag.chunking.tokenization.factory import (
    create_token_counter,
)

PROCESSED_CLEAN_ROOT = Path(
    "data/processed/clean"
)

CHUNK_STORAGE_ROOT = Path(
    "data/processed/chunks"
)


def create_base_config() -> ChunkingConfig:

    return ChunkingConfig(
        strategy="recursive",
        chunking_version="1.0",
        chunk_size=512,
        chunk_overlap=64,
        min_chunk_size=50,
        length_unit="tokens",
        merge_small_chunks=True,
        table_chunking_mode="row_group",
        table_rows_per_chunk=25,
        repeat_table_headers=True,
    )


def main():

    clean_files = sorted(
        PROCESSED_CLEAN_ROOT.rglob("*.json")
    )

    if not clean_files:

        raise RuntimeError(
            "No clean documents found."
        )

    base_config = create_base_config()

    token_counter = create_token_counter(
        base_config
    )

    profiler = DocumentProfiler(
        token_counter=token_counter,
        short_document_threshold=512,
    )

    chunking_service = ChunkingService()

    routed_service = RoutedChunkingService(
        profiler=profiler,
        router=ChunkingRouter(),
        chunking_service=chunking_service,
    )

    storage_service = ChunkStorageService(
        storage_root=CHUNK_STORAGE_ROOT
    )

    status_counts = Counter()

    strategy_chunk_counts = Counter()

    chunk_type_counts = Counter()

    department_document_counts = Counter()

    department_chunk_counts = Counter()

    routing_combinations = Counter()

    total_chunks = 0

    failures = []

    print()
    print("===================================")
    print("BATCH ROUTED CHUNK PERSISTENCE")
    print("===================================")

    print(
        f"Clean documents discovered : "
        f"{len(clean_files)}"
    )

    print(
        f"Chunk storage root         : "
        f"{CHUNK_STORAGE_ROOT}"
    )

    print()

    for index, clean_path in enumerate(
        clean_files,
        start=1,
    ):

        try:

            document = (
                DocumentSerializer
                .load_clean_document(
                    clean_path
                )
            )

            routed_result = (
                routed_service.chunk_document(
                    document=document,
                    base_config=base_config,
                )
            )

            status = storage_service.persist(
                document=document,
                result=routed_result,
                base_config=base_config,
            )

            # ----------------------------------
            # Reload persisted artifacts
            # ----------------------------------

            loaded_chunks = (
                storage_service.load_chunks(
                    document
                )
            )

            loaded_manifest = (
                storage_service.load_manifest(
                    document
                )
            )

            # ----------------------------------
            # Validate persisted artifacts
            # ----------------------------------

            validate_persisted_document(
                document=document,
                routed_result=routed_result,
                loaded_chunks=loaded_chunks,
                loaded_manifest=loaded_manifest,
            )

            # ----------------------------------
            # Statistics
            # ----------------------------------

            status_counts[status] += 1

            total_chunks += len(
                loaded_chunks
            )

            department = (
                document.metadata.department
            )

            department_document_counts[
                department
            ] += 1

            department_chunk_counts[
                department
            ] += len(loaded_chunks)

            for chunk in loaded_chunks:

                strategy_chunk_counts[
                    chunk.strategy
                ] += 1

                chunk_type_counts[
                    chunk.chunk_type
                ] += 1

            combination = tuple(
                decision.strategy
                for decision
                in routed_result
                .routing_result
                .decisions
            )

            routing_combinations[
                combination
            ] += 1

            print(
                f"[{index}/{len(clean_files)}] "
                f"{status:<20} "
                f"{clean_path} "
                f"({len(loaded_chunks)} chunks)"
            )

        except Exception as exc:

            status_counts["FAILED"] += 1

            failures.append(
                {
                    "path": str(clean_path),
                    "error": str(exc),
                    "traceback": (
                        traceback.format_exc()
                    ),
                }
            )

            print(
                f"[{index}/{len(clean_files)}] "
                f"FAILED "
                f"{clean_path}: {exc}"
            )

    print_summary(
        clean_files=clean_files,
        status_counts=status_counts,
        total_chunks=total_chunks,
        strategy_chunk_counts=(
            strategy_chunk_counts
        ),
        chunk_type_counts=(
            chunk_type_counts
        ),
        department_document_counts=(
            department_document_counts
        ),
        department_chunk_counts=(
            department_chunk_counts
        ),
        routing_combinations=(
            routing_combinations
        ),
        failures=failures,
    )


def validate_persisted_document(
    document,
    routed_result,
    loaded_chunks,
    loaded_manifest,
) -> None:

    # ----------------------------------
    # Count consistency
    # ----------------------------------

    if (
        len(loaded_chunks)
        != routed_result.total_chunks
    ):

        raise ValueError(
            "Persisted/routed chunk count mismatch."
        )

    if (
        loaded_manifest.total_chunks
        != len(loaded_chunks)
    ):

        raise ValueError(
            "Manifest/chunk count mismatch."
        )

    # ----------------------------------
    # Identity consistency
    # ----------------------------------

    expected_ids = [
        chunk.chunk_id
        for chunk in routed_result.chunks
    ]

    loaded_ids = [
        chunk.chunk_id
        for chunk in loaded_chunks
    ]

    if expected_ids != loaded_ids:

        raise ValueError(
            "Persisted chunk IDs/order changed."
        )

    if len(loaded_ids) != len(
        set(loaded_ids)
    ):

        raise ValueError(
            "Duplicate chunk IDs detected."
        )

    # ----------------------------------
    # Manifest/source consistency
    # ----------------------------------

    if (
        loaded_manifest.document_id
        != document.metadata.document_id
    ):

        raise ValueError(
            "Manifest document_id mismatch."
        )

    if (
        loaded_manifest.content_hash
        != document.metadata.content_hash
    ):

        raise ValueError(
            "Manifest content_hash mismatch."
        )

    if (
        loaded_manifest.tenant_id
        != document.metadata.tenant_id
    ):

        raise ValueError(
            "Manifest tenant_id mismatch."
        )

    if (
        loaded_manifest.department
        != document.metadata.department
    ):

        raise ValueError(
            "Manifest department mismatch."
        )

    if (
        loaded_manifest.router_version
        != routed_result.router_version
    ):

        raise ValueError(
            "Manifest router_version mismatch."
        )

    # ----------------------------------
    # Chunk/source consistency
    # ----------------------------------

    for chunk in loaded_chunks:

        if (
            chunk.document_id
            != document.metadata.document_id
        ):

            raise ValueError(
                "Stored chunk document_id mismatch."
            )

        if (
            chunk.content_hash
            != document.metadata.content_hash
        ):

            raise ValueError(
                "Stored chunk content_hash mismatch."
            )

        if (
            chunk.tenant_id
            != document.metadata.tenant_id
        ):

            raise ValueError(
                "Stored chunk tenant_id mismatch."
            )

        if (
            chunk.department
            != document.metadata.department
        ):

            raise ValueError(
                "Stored chunk department mismatch."
            )

        if not chunk.text.strip():

            raise ValueError(
                "Stored empty chunk detected."
            )

    # ----------------------------------
    # Manifest count consistency
    # ----------------------------------

    actual_strategy_counts = Counter(
        chunk.strategy
        for chunk in loaded_chunks
    )

    actual_chunk_type_counts = Counter(
        chunk.chunk_type
        for chunk in loaded_chunks
    )

    if (
        dict(sorted(actual_strategy_counts.items()))
        != loaded_manifest.strategy_counts
    ):

        raise ValueError(
            "Manifest strategy counts mismatch."
        )

    if (
        dict(
            sorted(
                actual_chunk_type_counts.items()
            )
        )
        != loaded_manifest.chunk_type_counts
    ):

        raise ValueError(
            "Manifest chunk type counts mismatch."
        )


def print_summary(
    clean_files,
    status_counts,
    total_chunks,
    strategy_chunk_counts,
    chunk_type_counts,
    department_document_counts,
    department_chunk_counts,
    routing_combinations,
    failures,
):

    print()
    print("===================================")
    print("CHUNK CORPUS PERSISTENCE SUMMARY")
    print("===================================")

    print(
        f"Documents discovered      : "
        f"{len(clean_files)}"
    )

    print(
        f"Written                   : "
        f"{status_counts['WRITTEN']}"
    )

    print(
        f"Skipped up-to-date        : "
        f"{status_counts['SKIPPED_UP_TO_DATE']}"
    )

    print(
        f"Rebuilt                   : "
        f"{status_counts['REBUILT']}"
    )

    print(
        f"Failed                    : "
        f"{status_counts['FAILED']}"
    )

    print(
        f"Total persisted chunks    : "
        f"{total_chunks}"
    )

    print()
    print("CHUNKS BY STRATEGY")
    print("-----------------------------------")

    for strategy, count in sorted(
        strategy_chunk_counts.items()
    ):

        print(
            f"{strategy:<20}: {count}"
        )

    print()
    print("CHUNKS BY TYPE")
    print("-----------------------------------")

    for chunk_type, count in sorted(
        chunk_type_counts.items()
    ):

        print(
            f"{chunk_type:<20}: {count}"
        )

    print()
    print("DOCUMENTS AND CHUNKS BY DEPARTMENT")
    print("-----------------------------------")

    departments = sorted(
        set(department_document_counts)
        | set(department_chunk_counts)
    )

    for department in departments:

        print(
            f"{department:<20}"
            f"documents="
            f"{department_document_counts[department]:<5}"
            f"chunks="
            f"{department_chunk_counts[department]}"
        )

    print()
    print("ROUTING COMBINATIONS")
    print("-----------------------------------")

    for combination, count in sorted(
        routing_combinations.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    ):

        label = " + ".join(
            combination
        )

        print(
            f"{label:<40}: {count}"
        )

    print()
    print("===================================")
    print("FAILURE DETAILS")
    print("===================================")

    if not failures:

        print("No failures detected.")

    else:

        for failure_number, failure in enumerate(
            failures,
            start=1,
        ):

            print()

            print(
                f"FAILURE #{failure_number}"
            )

            print(
                f"Path  : {failure['path']}"
            )

            print(
                f"Error : {failure['error']}"
            )

            print()
            print(failure["traceback"])

    print()
    print("===================================")
    print("FINAL STATUS")
    print("===================================")

    if failures:

        print(
            f"FAIL: {len(failures)} documents "
            "require investigation."
        )

    else:

        print(
            "PASS: Entire routed chunk corpus "
            "was persisted and reloaded "
            "successfully."
        )


if __name__ == "__main__":
    main()