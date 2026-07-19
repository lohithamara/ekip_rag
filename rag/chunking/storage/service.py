import shutil
import tempfile
from pathlib import Path

from ingestion.schemas.documents import CleanDocument
from rag.chunking.schemas import Chunk
from rag.chunking.storage.schemas import ChunkManifest
from rag.chunking.config import ChunkingConfig
from rag.chunking.routing.results import RoutedChunkingResult
from rag.chunking.storage.manifest import (
    ManifestBuilder,
    ManifestSerializer,
)
from rag.chunking.storage.serializer import ChunkSerializer


class ChunkStorageService:

    CHUNKS_FILENAME = "chunks.jsonl"
    MANIFEST_FILENAME = "manifest.json"

    def __init__(
        self,
        storage_root: Path,
    ):
        self.storage_root = Path(storage_root)

    def persist(
        self,
        document: CleanDocument,
        result: RoutedChunkingResult,
        base_config: ChunkingConfig,
        force: bool = False,
    ) -> str:

        self._validate_input(
            document=document,
            result=result,
        )

        final_dir = self._get_document_directory(
            document
        )

        manifest = ManifestBuilder.build(
            document=document,
            result=result,
            base_config=base_config,
        )

        if (
            not force
            and self._is_up_to_date(
                final_dir=final_dir,
                expected_manifest=manifest,
            )
        ):
            return "SKIPPED_UP_TO_DATE"

        existed_before = final_dir.exists()

        self.storage_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        staging_root = Path(
            tempfile.mkdtemp(
                prefix=".chunk-staging-",
                dir=self.storage_root,
            )
        )

        staging_dir = (
            staging_root
            / document.metadata.tenant_id
            / document.metadata.department
            / document.metadata.content_hash
        )

        staging_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        try:

            chunks_path = (
                staging_dir
                / self.CHUNKS_FILENAME
            )

            manifest_path = (
                staging_dir
                / self.MANIFEST_FILENAME
            )

            ChunkSerializer.write_jsonl(
                result.chunks,
                chunks_path,
            )

            ManifestSerializer.write_json(
                manifest,
                manifest_path,
            )

            self._validate_staged_artifacts(
                document=document,
                result=result,
                expected_manifest=manifest,
                chunks_path=chunks_path,
                manifest_path=manifest_path,
            )

            self._commit_directory(
                staging_dir=staging_dir,
                final_dir=final_dir,
            )

            if existed_before:
                return "REBUILT"

            return "WRITTEN"

        finally:

            shutil.rmtree(
                staging_root,
                ignore_errors=True,
            )

    def load_chunks(
        self,
        document: CleanDocument,
    ) -> tuple[Chunk, ...]:

        document_dir = (
            self._get_document_directory(
                document
            )
        )

        return ChunkSerializer.load_jsonl(
            document_dir
            / self.CHUNKS_FILENAME
        )
    
    def load_manifest(
        self,
        document: CleanDocument,
    ) -> ChunkManifest:

        document_dir = (
            self._get_document_directory(
                document
            )
        )

        return ManifestSerializer.load_json(
            document_dir
            / self.MANIFEST_FILENAME
        )

    def _get_document_directory(
        self,
        document: CleanDocument,
    ) -> Path:
        metadata = document.metadata

        return (
            self.storage_root
            / metadata.tenant_id
            / metadata.department
            / metadata.content_hash
        )  

    @staticmethod
    def _validate_input(
        document: CleanDocument,
        result: RoutedChunkingResult,
    ) -> None:

        if (
            result.document_id
            != document.metadata.document_id
        ):
            raise ValueError(
                "Storage document/result ID mismatch."
            )

        if (
            result.tenant_id
            != document.metadata.tenant_id
        ):
            raise ValueError(
                "Storage tenant/result ID mismatch."
            )

        if result.total_chunks != len(
            result.chunks
        ):
            raise ValueError(
                "Storage result total_chunks mismatch."
            )

        if not result.chunks:
            raise ValueError(
                "Refusing to persist zero chunks."
            )

    def _is_up_to_date(
        self,
        final_dir: Path,
        expected_manifest,
    ) -> bool:

        chunks_path = (
            final_dir
            / self.CHUNKS_FILENAME
        )

        manifest_path = (
            final_dir
            / self.MANIFEST_FILENAME
        )

        if (
            not chunks_path.is_file()
            or not manifest_path.is_file()
        ):
            return False

        try:

            stored_manifest = (
                ManifestSerializer.load_json(
                    manifest_path
                )
            )

        except Exception:

            return False

        return (
            stored_manifest.document_id
            == expected_manifest.document_id
            and stored_manifest.content_hash
            == expected_manifest.content_hash
            and stored_manifest.tenant_id
            == expected_manifest.tenant_id
            and stored_manifest.department
            == expected_manifest.department
            and stored_manifest.router_version
            == expected_manifest.router_version
            and stored_manifest.chunking_version
            == expected_manifest.chunking_version
            and stored_manifest.chunking_config
            == expected_manifest.chunking_config
            and stored_manifest.strategy_counts
            == expected_manifest.strategy_counts
            and stored_manifest.chunk_type_counts
            == expected_manifest.chunk_type_counts
            and stored_manifest.total_chunks
            == expected_manifest.total_chunks
        )

    @staticmethod
    def _validate_staged_artifacts(
        document: CleanDocument,
        result: RoutedChunkingResult,
        expected_manifest,
        chunks_path: Path,
        manifest_path: Path,
    ) -> None:

        loaded_chunks = (
            ChunkSerializer.load_jsonl(
                chunks_path
            )
        )

        loaded_manifest = (
            ManifestSerializer.load_json(
                manifest_path
            )
        )

        if (
            loaded_manifest
            != expected_manifest
        ):
            raise ValueError(
                "Staged manifest round-trip mismatch."
            )

        if len(loaded_chunks) != len(
            result.chunks
        ):
            raise ValueError(
                "Staged chunk count mismatch."
            )

        expected_ids = [
            chunk.chunk_id
            for chunk in result.chunks
        ]

        loaded_ids = [
            chunk.chunk_id
            for chunk in loaded_chunks
        ]

        if loaded_ids != expected_ids:
            raise ValueError(
                "Staged chunk ID/order mismatch."
            )

        if len(loaded_ids) != len(
            set(loaded_ids)
        ):
            raise ValueError(
                "Duplicate chunk IDs in staged artifacts."
            )

        if (
            loaded_manifest.total_chunks
            != len(loaded_chunks)
        ):
            raise ValueError(
                "Manifest/chunk count mismatch."
            )

        for chunk in loaded_chunks:

            if (
                chunk.document_id
                != document.metadata.document_id
            ):
                raise ValueError(
                    "Stored chunk document_id mismatch."
                )

            if (
                chunk.tenant_id
                != document.metadata.tenant_id
            ):
                raise ValueError(
                    "Stored chunk tenant_id mismatch."
                )

    @staticmethod
    def _commit_directory(
        staging_dir: Path,
        final_dir: Path,
    ) -> None:
        final_dir.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        backup_dir = final_dir.with_name(
            f".{final_dir.name}.backup"
        )

        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        try:
            if final_dir.exists():
                final_dir.rename(backup_dir)

            staging_dir.rename(final_dir)

        except Exception:
            if not final_dir.exists() and backup_dir.exists():
                backup_dir.rename(final_dir)

            raise

        if backup_dir.exists():
            shutil.rmtree(backup_dir)