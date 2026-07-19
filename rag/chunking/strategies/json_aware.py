import json

from ingestion.schemas.documents import CleanDocument

from rag.chunking.base import BaseChunkingStrategy
from rag.chunking.chunk_factory import ChunkFactory
from rag.chunking.config import ChunkingConfig
from rag.chunking.schemas import ChunkingResult
from rag.chunking.tokenization.base import BaseTokenCounter


class JSONAwareChunkingStrategy(
    BaseChunkingStrategy
):

    STRATEGY_NAME = "json_aware"

    def __init__(
        self,
        config: ChunkingConfig,
        token_counter: BaseTokenCounter,
        chunk_factory: ChunkFactory,
    ):
        if config.strategy != self.name:
            raise ValueError(
                f"Expected strategy='{self.name}', "
                f"received '{config.strategy}'."
            )

        self.config = config
        self.token_counter = token_counter
        self.chunk_factory = chunk_factory

    @property
    def name(self) -> str:
        return self.STRATEGY_NAME

    def chunk(
        self,
        document: CleanDocument,
    ) -> ChunkingResult:

        text = document.clean_text

        if not text or not text.strip():
            return self._build_result(
                document,
                [],
            )

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(
                "JSON-aware chunking requires valid JSON."
            )

        pieces = self._split(data)

        chunks = []

        for chunk_index, piece in enumerate(pieces):

            chunks.append(
                self.chunk_factory.create_chunk(
                    document=document,
                    text=piece,
                    chunk_index=chunk_index,
                    section_path=[],
                    start_char=None,
                    end_char=None,
                    metadata={
                        "length_unit": (
                            self.config.length_unit
                        ),
                        "json_aware": True,
                        "token_counter": (
                            self.token_counter.name
                        ),
                    },
                )
            )

        return self._build_result(
            document,
            chunks,
        )

    def _split(
        self,
        data,
    ) -> list[str]:

        chunks = []

        self._walk(
            data,
            [],
            chunks,
        )

        return chunks

    def _walk(
        self,
        value,
        path,
        chunks,
    ) -> None:

        if isinstance(value, dict):

            for key, child in value.items():
                self._walk(
                    child,
                    [*path, str(key)],
                    chunks,
                )

            return

        if isinstance(value, list):

            if not value:
                return

            if all(
                not isinstance(item, (dict, list))
                for item in value
            ):
                chunks.append(
                    self._serialize(path, value)
                )
                return

            for index, child in enumerate(value):

                child_path = [
                    *path,
                    str(index),
                ]

                if isinstance(child, dict):
                    chunks.append(
                        self._serialize(
                            child_path,
                            child,
                        )
                    )
                else:
                    self._walk(
                        child,
                        child_path,
                        chunks,
                    )

            return

        chunks.append(
            self._serialize(path, value)
        )

    @staticmethod
    def _serialize(
        path,
        value,
    ) -> str:

        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

        key = ".".join(path)

        if not key:
            return serialized

        return f"{key}: {serialized}"

    def _build_result(
        self,
        document,
        chunks,
    ) -> ChunkingResult:

        return ChunkingResult(
            document_id=document.metadata.document_id,
            tenant_id=document.metadata.tenant_id,
            strategy=self.name,
            chunking_version=(
                self.config.chunking_version
            ),
            chunks=chunks,
            total_chunks=len(chunks),
            total_characters=sum(
                chunk.character_count
                for chunk in chunks
            ),
            total_tokens=sum(
                chunk.token_count
                for chunk in chunks
            ),
            metadata={
                "json_aware": True,
                "token_counter": (
                    self.token_counter.name
                ),
            },
        )