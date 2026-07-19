import hashlib
from typing import Any

from ingestion.schemas.documents import CleanDocument
from rag.chunking.config import ChunkingConfig
from rag.chunking.schemas import Chunk
from rag.chunking.tokenization.base import (
    BaseTokenCounter,
)


class ChunkFactory:

    def __init__(
        self,
        config: ChunkingConfig,
        token_counter: BaseTokenCounter,
    ):
        self.config = config

        self.token_counter = token_counter
    chunk_type: str = "text"
    def create_chunk(
        self,
        document: CleanDocument,
        text: str,
        chunk_index: int,
        *,
        chunk_type: str = "text",
        page_numbers: list[int] | None = None,
        table_ids: list[str] | None = None,
        section_path: list[str] | None = None,
        start_char: int | None = None,
        end_char: int | None = None,
        parent_chunk_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Chunk:

        normalized_text = text.strip()

        if not normalized_text:
            raise ValueError(
                "Cannot create a chunk with empty text."
            )

        chunk_id = self._generate_chunk_id(
            document=document,
            text=normalized_text,
            chunk_index=chunk_index,
        )

        return Chunk(
            chunk_type=chunk_type,
            chunk_id=chunk_id,
            document_id=document.metadata.document_id,
            content_hash=document.metadata.content_hash,
            tenant_id=document.metadata.tenant_id,
            department=document.metadata.department,
            text=normalized_text,
            chunk_index=chunk_index,
            strategy=self.config.strategy,
            chunking_version=(
                self.config.chunking_version
            ),
            character_count=len(normalized_text),
            token_count=self.token_counter.count(
                normalized_text
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
            page_numbers=self._normalize_int_list(
                page_numbers
            ),
            table_ids=self._normalize_string_list(
                table_ids
            ),
            section_path=self._normalize_string_list(
                section_path
            ),
            start_char=start_char,
            end_char=end_char,
            parent_chunk_id=parent_chunk_id,
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

    def _generate_chunk_id(
        self,
        document: CleanDocument,
        text: str,
        chunk_index: int,
    ) -> str:

        text_hash = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        identity = "|".join(
            [
                document.metadata.tenant_id,
                document.metadata.document_id,
                document.metadata.content_hash,
                self.config.strategy,
                self.config.chunking_version,
                str(chunk_index),
                text_hash,
            ]
        )

        return hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _normalize_int_list(
        values: list[int] | None,
    ) -> list[int]:

        if not values:
            return []

        return sorted(set(values))

    @staticmethod
    def _normalize_string_list(
        values: list[str] | None,
    ) -> list[str]:

        if not values:
            return []

        return list(
            dict.fromkeys(
                value
                for value in values
                if value
            )
        )