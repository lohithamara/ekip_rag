import re

from ingestion.schemas.documents import CleanDocument

from rag.chunking.base import BaseChunkingStrategy
from rag.chunking.chunk_factory import ChunkFactory
from rag.chunking.config import ChunkingConfig
from rag.chunking.schemas import ChunkingResult
from rag.chunking.tokenization.base import BaseTokenCounter


class ParagraphChunkingStrategy(BaseChunkingStrategy):

    PARAGRAPH_PATTERN = re.compile(
        r"\S(?:.*?\S)?(?=\n\s*\n|\Z)",
        flags=re.DOTALL,
    )

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
    STRATEGY_NAME = "paragraph"
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

        paragraphs = self._extract_paragraphs(text)

        atomic_units = []

        for paragraph_text, start_char, end_char in paragraphs:

            if (
                self.token_counter.count(paragraph_text)
                <= self.config.chunk_size
            ):

                atomic_units.append(
                    (
                        paragraph_text,
                        start_char,
                        end_char,
                    )
                )

            else:

                atomic_units.extend(
                    self._split_oversized_paragraph(
                        text=paragraph_text,
                        source_start=start_char,
                    )
                )

        groups = self._group_units(atomic_units)

        groups = self._handle_small_tail(groups)

        chunks = []

        for chunk_index, group in enumerate(groups):

            chunk_text = self._join_units(group)

            chunks.append(
                self.chunk_factory.create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=group[0][1],
                    end_char=group[-1][2],
                    metadata={
                        "length_unit": (
                            self.config.length_unit
                        ),
                        "paragraph_count": len(group),
                        "token_counter": (
                            self.token_counter.name
                        ),
                    },
                )
            )

        return self._build_result(
            document=document,
            chunks=chunks,
        )

    def _extract_paragraphs(
        self,
        text: str,
    ) -> list[tuple[str, int, int]]:

        paragraphs = []

        for match in self.PARAGRAPH_PATTERN.finditer(text):

            raw_paragraph = match.group()

            leading = (
                len(raw_paragraph)
                - len(raw_paragraph.lstrip())
            )

            trailing = (
                len(raw_paragraph)
                - len(raw_paragraph.rstrip())
            )

            paragraph_text = raw_paragraph.strip()

            if not paragraph_text:
                continue

            paragraphs.append(
                (
                    paragraph_text,
                    match.start() + leading,
                    match.end() - trailing,
                )
            )

        return paragraphs

    def _split_oversized_paragraph(
        self,
        text: str,
        source_start: int,
    ) -> list[tuple[str, int, int]]:

        if self.config.length_unit == "characters":

            return self._split_by_characters(
                text=text,
                source_start=source_start,
            )

        return self._split_by_tokens(
            text=text,
            source_start=source_start,
        )

    def _split_by_characters(
        self,
        text: str,
        source_start: int,
    ) -> list[tuple[str, int, int]]:

        units = []

        local_start = 0

        while local_start < len(text):

            local_end = min(
                local_start + self.config.chunk_size,
                len(text),
            )

            piece = text[local_start:local_end]

            if piece.strip():

                units.append(
                    (
                        piece,
                        source_start + local_start,
                        source_start + local_end,
                    )
                )

            if local_end >= len(text):
                break

            local_start = local_end

        return units

    def _split_by_tokens(
        self,
        text: str,
        source_start: int,
    ) -> list[tuple[str, int, int]]:

        token_spans = [
            match.span()
            for match in re.finditer(
                r"\S+",
                text,
            )
        ]

        units = []

        token_start = 0

        while token_start < len(token_spans):

            token_end = min(
                token_start + self.config.chunk_size,
                len(token_spans),
            )

            local_start = token_spans[token_start][0]

            local_end = token_spans[token_end - 1][1]

            units.append(
                (
                    text[local_start:local_end],
                    source_start + local_start,
                    source_start + local_end,
                )
            )

            if token_end >= len(token_spans):
                break

            token_start = token_end

        return units

    def _group_units(
        self,
        units: list[tuple[str, int, int]],
    ) -> list[list[tuple[str, int, int]]]:

        groups = []

        current_group = []

        for unit in units:

            candidate = current_group + [unit]

            if (
                current_group
                and self.token_counter.count(
                    self._join_units(candidate)
                ) > self.config.chunk_size
            ):

                groups.append(current_group)

                current_group = (
                    self._build_overlap_group(
                        current_group
                    )
                )

                while (
                    current_group
                    and self.token_counter.count(
                        self._join_units(
                            current_group + [unit]
                        )
                    ) > self.config.chunk_size
                ):
                    current_group.pop(0)

            current_group.append(unit)

        if current_group:
            groups.append(current_group)

        return groups

    def _build_overlap_group(
        self,
        previous_group: list[
            tuple[str, int, int]
        ],
    ) -> list[tuple[str, int, int]]:

        if self.config.chunk_overlap <= 0:
            return []

        overlap_units = []

        for unit in reversed(previous_group):

            candidate = [
                unit,
                *overlap_units,
            ]

            if (
                self.token_counter.count(
                    self._join_units(candidate)
                )
                > self.config.chunk_overlap
            ):
                break

            overlap_units = candidate

        return overlap_units

    def _handle_small_tail(
        self,
        groups: list[
            list[tuple[str, int, int]]
        ],
    ) -> list[list[tuple[str, int, int]]]:

        if len(groups) < 2:
            return groups

        tail_text = self._join_units(
            groups[-1]
        )

        if (
            self.token_counter.count(tail_text)
            >= self.config.min_chunk_size
        ):
            return groups

        if not self.config.merge_small_chunks:
            return groups

        combined = groups[-2] + groups[-1]

        if (
            self.token_counter.count(
                self._join_units(combined)
            )
            <= self.config.chunk_size
        ):
            groups[-2] = combined
            groups.pop()

        return groups

    @staticmethod
    def _join_units(
        units: list[tuple[str, int, int]],
    ) -> str:

        return "\n\n".join(
            unit[0].strip()
            for unit in units
            if unit[0].strip()
        )

    def _build_result(
        self,
        document: CleanDocument,
        chunks: list,
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
                "length_unit": self.config.length_unit,
                "chunk_size": self.config.chunk_size,
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
                "paragraph_separator": (
                    "blank_line"
                ),
            },
        )