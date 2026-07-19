import json
import os
import tempfile

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ingestion.schemas.documents import (
    CleanDocument,
)

from rag.chunking.config import ChunkingConfig
from rag.chunking.routing.results import (
    RoutedChunkingResult,
)
from rag.chunking.storage.schemas import (
    ChunkManifest,
    StrategyManifest,
)


class ManifestBuilder:

    MANIFEST_VERSION = "1.0"

    @classmethod
    def build(
        cls,
        document: CleanDocument,
        result: RoutedChunkingResult,
        base_config: ChunkingConfig,
    ) -> ChunkManifest:

        if (
            result.document_id
            != document.metadata.document_id
        ):

            raise ValueError(
                "Manifest document/result ID mismatch."
            )

        strategy_result_map = {
            strategy_result.strategy:
            strategy_result
            for strategy_result
            in result.strategy_results
        }

        strategy_manifests = []

        for decision in (
            result.routing_result.decisions
        ):

            strategy_result = (
                strategy_result_map.get(
                    decision.strategy
                )
            )

            if strategy_result is None:

                raise ValueError(
                    "Missing strategy result for "
                    f"{decision.strategy}."
                )

            chunk_type_counts = Counter(
                chunk.chunk_type
                for chunk
                in strategy_result.chunks
            )

            strategy_manifests.append(
                StrategyManifest(
                    strategy=decision.strategy,
                    content_scope=(
                        decision.content_scope
                    ),
                    reason=decision.reason,
                    priority=decision.priority,
                    total_chunks=(
                        strategy_result.total_chunks
                    ),
                    chunk_type_counts=dict(
                        sorted(
                            chunk_type_counts.items()
                        )
                    ),
                    metadata=dict(
                        decision.metadata
                    ),
                )
            )

        chunk_type_counts = Counter(
            chunk.chunk_type
            for chunk in result.chunks
        )

        strategy_counts = Counter(
            chunk.strategy
            for chunk in result.chunks
        )

        return ChunkManifest(
            manifest_version=(
                cls.MANIFEST_VERSION
            ),
            document_id=(
                document.metadata.document_id
            ),
            content_hash=(
                document.metadata.content_hash
            ),
            tenant_id=(
                document.metadata.tenant_id
            ),
            department=(
                document.metadata.department
            ),
            source_filename=(
                document.metadata.filename
            ),
            source_file_type=(
                document.metadata.file_type
            ),
            source_s3_key=(
                document.metadata.original_s3_key
            ),
            router_version=(
                result.router_version
            ),
            chunking_version=(
                base_config.chunking_version
            ),
            total_chunks=result.total_chunks,
            chunk_type_counts=dict(
                sorted(
                    chunk_type_counts.items()
                )
            ),
            strategy_counts=dict(
                sorted(
                    strategy_counts.items()
                )
            ),
            strategies=tuple(
                strategy_manifests
            ),
            chunking_config=cls._config_to_dict(
                base_config
            ),
            generated_at=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            metadata={
                "storage_format": "jsonl",
                "chunk_identity_scope": (
                    "document_strategy_index"
                ),
            },
        )

    @staticmethod
    def _config_to_dict(
        config: ChunkingConfig,
    ) -> dict:

        return asdict(config)


class ManifestSerializer:

    @staticmethod
    def manifest_to_dict(
        manifest: ChunkManifest,
    ) -> dict:

        return asdict(manifest)

    @staticmethod
    def manifest_from_dict(
        data: dict,
    ) -> ChunkManifest:

        payload = dict(data)

        # ----------------------------------
        # Restore nested StrategyManifest
        # ----------------------------------

        payload["strategies"] = tuple(
            StrategyManifest(**strategy)
            for strategy in payload[
                "strategies"
            ]
        )
    

        # ----------------------------------
        # Restore known tuple-valued config
        # fields changed to lists by JSON.
        # ----------------------------------

        chunking_config = dict(
            payload["chunking_config"]
        )

        tuple_config_fields = (
            "separators",
        )

        for field_name in tuple_config_fields:

            value = chunking_config.get(
                field_name
            )

            if isinstance(value, list):

                chunking_config[field_name] = tuple(
                    value
                )

        payload[
            "chunking_config"
        ] = chunking_config

        return ChunkManifest(**payload)

    @classmethod
    def write_json(
        cls,
        manifest: ChunkManifest,
        output_path: Path,
    ) -> None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = cls._create_temp_path(
            output_path
        )

        try:

            with temp_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as file:

                json.dump(
                    cls.manifest_to_dict(
                        manifest
                    ),
                    file,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )

                file.write("\n")

                file.flush()

                os.fsync(file.fileno())

            os.replace(
                temp_path,
                output_path,
            )

        except Exception:

            if temp_path.exists():

                temp_path.unlink()

            raise

    @classmethod
    def load_json(
        cls,
        input_path: Path,
    ) -> ChunkManifest:

        input_path = Path(input_path)

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        return cls.manifest_from_dict(data)

    @staticmethod
    def _create_temp_path(
        output_path: Path,
    ) -> Path:

        file_descriptor, temp_name = (
            tempfile.mkstemp(
                prefix=(
                    f".{output_path.name}."
                ),
                suffix=".tmp",
                dir=output_path.parent,
            )
        )

        os.close(file_descriptor)

        return Path(temp_name)