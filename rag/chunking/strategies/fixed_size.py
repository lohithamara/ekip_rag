from rag.chunking.base import BaseChunkingStrategy
from rag.chunking.chunk_factory import ChunkFactory
from rag.chunking.config import ChunkingConfig
from rag.chunking.schemas import ChunkingResult
from rag.chunking.tokenization.base import BaseTokenCounter
from ingestion.schemas.documents import CleanDocument


class FixedSizeChunkingStrategy(
    BaseChunkingStrategy
):

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
    STRATEGY_NAME = "fixed_size"

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
                document=document,
                chunks=[],
            )

        if self.config.length_unit == "characters":

            chunks = self._chunk_by_characters(
                document
            )

        elif self.config.length_unit == "tokens":

            chunks = self._chunk_by_tokens(
                document
            )

        else:

            raise ValueError(
                f"Unsupported length unit: "
                f"{self.config.length_unit}"
            )

        return self._build_result(
            document=document,
            chunks=chunks,
        )

    def _chunk_by_characters(
        self,
        document: CleanDocument,
    ) -> list:

        text = document.clean_text

        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        step = chunk_size - overlap

        chunks = []

        chunk_index = 0

        start = 0

        while start < len(text):

            end = min(
                start + chunk_size,
                len(text),
            )

            chunk_text = text[start:end]

            if self._should_keep_chunk(
                chunk_text
            ):

                chunk = self.chunk_factory.create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "length_unit": "characters",
                        "window_start": start,
                        "window_end": end,
                    },
                )

                chunks.append(chunk)

                chunk_index += 1

            if end >= len(text):
                break

            start += step

        return chunks

    def _chunk_by_tokens(
        self,
        document: CleanDocument,
    ) -> list:

        text = document.clean_text

        token_spans = self._find_whitespace_token_spans(
            text
        )

        if not token_spans:
            return []

        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        step = chunk_size - overlap

        chunks = []

        chunk_index = 0

        token_start = 0

        while token_start < len(token_spans):

            token_end = min(
                token_start + chunk_size,
                len(token_spans),
            )

            start_char = token_spans[
                token_start
            ][0]

            end_char = token_spans[
                token_end - 1
            ][1]

            chunk_text = text[
                start_char:end_char
            ]

            if self._should_keep_chunk(
                chunk_text
            ):

                chunk = self.chunk_factory.create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "length_unit": "tokens",
                        "token_start": token_start,
                        "token_end": token_end,
                    },
                )

                chunks.append(chunk)

                chunk_index += 1

            if token_end >= len(token_spans):
                break

            token_start += step

        return chunks

    def _should_keep_chunk(
        self,
        text: str,
    ) -> bool:

        if not text or not text.strip():
            return False

        chunk_length = self.token_counter.count(
            text
        )

        return (
            chunk_length
            >= self.config.min_chunk_size
        )

    @staticmethod
    def _find_whitespace_token_spans(
        text: str,
    ) -> list[tuple[int, int]]:

        import re

        return [
            match.span()
            for match in re.finditer(
                r"\S+",
                text,
            )
        ]

    def _build_result(
        self,
        document: CleanDocument,
        chunks: list,
    ) -> ChunkingResult:

        return ChunkingResult(
            document_id=(
                document.metadata.document_id
            ),
            tenant_id=(
                document.metadata.tenant_id
            ),
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
                "length_unit": (
                    self.config.length_unit
                ),
                "chunk_size": (
                    self.config.chunk_size
                ),
                "chunk_overlap": (
                    self.config.chunk_overlap
                ),
                "min_chunk_size": (
                    self.config.min_chunk_size
                ),
                "token_counter": (
                    self.token_counter.name
                ),
            },
        )