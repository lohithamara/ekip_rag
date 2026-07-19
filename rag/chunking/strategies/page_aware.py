import re

from ingestion.schemas.documents import CleanDocument, ExtractedPage

from rag.chunking.base import BaseChunkingStrategy
from rag.chunking.chunk_factory import ChunkFactory
from rag.chunking.config import ChunkingConfig
from rag.chunking.schemas import ChunkingResult
from rag.chunking.tokenization.base import BaseTokenCounter


class PageAwareChunkingStrategy(BaseChunkingStrategy):

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
    STRATEGY_NAME = "page_aware"
    @property
    def name(self) -> str:
        return self.STRATEGY_NAME

    def chunk(
        self,
        document: CleanDocument,
    ) -> ChunkingResult:

        if not document.pages:

            return self._build_result(
                document=document,
                chunks=[],
                skipped_empty_pages=0,
            )

        chunks = []

        chunk_index = 0

        skipped_empty_pages = 0

        for page in document.pages:

            page_text = page.text

            if not page_text or not page_text.strip():

                skipped_empty_pages += 1

                continue

            page_units = self._chunk_page(page)

            page_units = self._handle_small_page_tail(
                page_units
            )

            for unit in page_units:

                chunk_text = unit["text"]

                chunks.append(
                    self.chunk_factory.create_chunk(
                        document=document,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        page_numbers=[
                            page.page_number
                        ],
                        start_char=None,
                        end_char=None,
                        metadata={
                            "length_unit": (
                                self.config.length_unit
                            ),
                            "page_number": (
                                page.page_number
                            ),
                            "page_local_start": (
                                unit["local_start"]
                            ),
                            "page_local_end": (
                                unit["local_end"]
                            ),
                            "ocr_used": page.ocr_used,
                            "page_metadata": dict(
                                page.metadata
                            ),
                            "token_counter": (
                                self.token_counter.name
                            ),
                        },
                    )
                )

                chunk_index += 1

        return self._build_result(
            document=document,
            chunks=chunks,
            skipped_empty_pages=skipped_empty_pages,
        )

    def _chunk_page(
        self,
        page: ExtractedPage,
    ) -> list[dict]:

        text = page.text

        if (
            self.token_counter.count(text)
            <= self.config.chunk_size
        ):

            return [
                {
                    "text": text,
                    "local_start": 0,
                    "local_end": len(text),
                }
            ]

        if self.config.length_unit == "characters":

            return self._chunk_page_by_characters(
                text
            )

        return self._chunk_page_by_tokens(text)

    def _chunk_page_by_characters(
        self,
        text: str,
    ) -> list[dict]:

        units = []

        step = (
            self.config.chunk_size
            - self.config.chunk_overlap
        )

        start = 0

        while start < len(text):

            end = min(
                start + self.config.chunk_size,
                len(text),
            )

            chunk_text = text[start:end]

            if chunk_text.strip():

                units.append(
                    {
                        "text": chunk_text,
                        "local_start": start,
                        "local_end": end,
                    }
                )

            if end >= len(text):
                break

            start += step

        return units

    def _chunk_page_by_tokens(
        self,
        text: str,
    ) -> list[dict]:

        token_spans = [
            match.span()
            for match in re.finditer(
                r"\S+",
                text,
            )
        ]

        if not token_spans:
            return []

        units = []

        step = (
            self.config.chunk_size
            - self.config.chunk_overlap
        )

        token_start = 0

        while token_start < len(token_spans):

            token_end = min(
                token_start + self.config.chunk_size,
                len(token_spans),
            )

            local_start = token_spans[
                token_start
            ][0]

            local_end = token_spans[
                token_end - 1
            ][1]

            units.append(
                {
                    "text": text[
                        local_start:local_end
                    ],
                    "local_start": local_start,
                    "local_end": local_end,
                }
            )

            if token_end >= len(token_spans):
                break

            token_start += step

        return units

    def _handle_small_page_tail(
        self,
        units: list[dict],
    ) -> list[dict]:

        # Preserve all page content.
        #
        # We intentionally retain undersized tails rather
        # than performing unsafe merges across overlapping
        # windows.
        #
        # Chunk validation/evaluation will measure the
        # tiny-chunk rate later.

        return units

    def _build_result(
        self,
        document: CleanDocument,
        chunks: list,
        skipped_empty_pages: int,
    ) -> ChunkingResult:

        pages_with_text = sum(
            1
            for page in document.pages
            if page.text and page.text.strip()
        )

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
                "merge_small_chunks": (
                    self.config.merge_small_chunks
                ),
                "token_counter": (
                    self.token_counter.name
                ),
                "source_page_count": len(
                    document.pages
                ),
                "pages_with_text": (
                    pages_with_text
                ),
                "skipped_empty_pages": (
                    skipped_empty_pages
                ),
                "cross_page_chunks": False,
            },
        )