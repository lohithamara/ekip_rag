import re

from ingestion.schemas.documents import CleanDocument

from rag.chunking.base import BaseChunkingStrategy
from rag.chunking.chunk_factory import ChunkFactory
from rag.chunking.config import ChunkingConfig
from rag.chunking.schemas import ChunkingResult
from rag.chunking.tokenization.base import BaseTokenCounter


class SentenceChunkingStrategy(BaseChunkingStrategy):

    SENTENCE_PATTERN = re.compile(
        r".+?(?:[.!?]+(?=\s+|$)|$)",
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
    STRATEGY_NAME = "sentence"
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

        sentence_units = self._extract_sentences(text)

        atomic_units = []

        for sentence_text, start_char, end_char in sentence_units:

            if (
                self.token_counter.count(sentence_text)
                <= self.config.chunk_size
            ):
                atomic_units.append(
                    (
                        sentence_text,
                        start_char,
                        end_char,
                    )
                )

            else:
                atomic_units.extend(
                    self._split_oversized_sentence(
                        text=sentence_text,
                        source_start=start_char,
                    )
                )

        chunk_groups = self._group_units(
            atomic_units
        )

        chunk_groups = self._handle_small_tail(
            chunk_groups
        )

        chunks = []

        for chunk_index, group in enumerate(chunk_groups):

            chunk_text = self._join_units(group)

            start_char = group[0][1]
            end_char = group[-1][2]

            chunks.append(
                self.chunk_factory.create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "length_unit": (
                            self.config.length_unit
                        ),
                        "sentence_count": len(group),
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

    def _extract_sentences(
        self,
        text: str,
    ) -> list[tuple[str, int, int]]:

        sentences = []

        for match in self.SENTENCE_PATTERN.finditer(text):

            raw_sentence = match.group()

            leading_whitespace = (
                len(raw_sentence)
                - len(raw_sentence.lstrip())
            )

            trailing_whitespace = (
                len(raw_sentence)
                - len(raw_sentence.rstrip())
            )

            sentence_text = raw_sentence.strip()

            if not sentence_text:
                continue

            start_char = (
                match.start()
                + leading_whitespace
            )

            end_char = (
                match.end()
                - trailing_whitespace
            )

            sentences.append(
                (
                    sentence_text,
                    start_char,
                    end_char,
                )
            )

        return sentences

    def _split_oversized_sentence(
        self,
        text: str,
        source_start: int,
    ) -> list[tuple[str, int, int]]:

        if self.config.length_unit == "characters":

            return self._split_oversized_by_characters(
                text=text,
                source_start=source_start,
            )

        return self._split_oversized_by_tokens(
            text=text,
            source_start=source_start,
        )

    def _split_oversized_by_characters(
        self,
        text: str,
        source_start: int,
    ) -> list[tuple[str, int, int]]:

        units = []

        start = 0

        while start < len(text):

            end = min(
                start + self.config.chunk_size,
                len(text),
            )

            piece = text[start:end]

            if piece.strip():

                units.append(
                    (
                        piece,
                        source_start + start,
                        source_start + end,
                    )
                )

            if end >= len(text):
                break

            start = end

        return units

    def _split_oversized_by_tokens(
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

            local_start = token_spans[
                token_start
            ][0]

            local_end = token_spans[
                token_end - 1
            ][1]

            piece = text[
                local_start:local_end
            ]

            units.append(
                (
                    piece,
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

            candidate_group = (
                current_group + [unit]
            )

            candidate_text = self._join_units(
                candidate_group
            )

            if (
                current_group
                and self.token_counter.count(
                    candidate_text
                ) > self.config.chunk_size
            ):

                groups.append(current_group)

                current_group = self._build_overlap_group(
                    current_group
                )

                # A defensive check. Overlap plus the new unit
                # must never create an oversized chunk.
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

            candidate_text = self._join_units(
                candidate
            )

            if (
                self.token_counter.count(
                    candidate_text
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

        tail = groups[-1]

        tail_text = self._join_units(tail)

        if (
            self.token_counter.count(tail_text)
            >= self.config.min_chunk_size
        ):
            return groups

        if not self.config.merge_small_chunks:
            return groups

        previous = groups[-2]

        combined = previous + tail

        combined_text = self._join_units(
            combined
        )

        if (
            self.token_counter.count(
                combined_text
            )
            <= self.config.chunk_size
        ):
            groups[-2] = combined
            groups.pop()

        # If merging would exceed chunk_size,
        # retain the tail. Never silently discard text.

        return groups

    @staticmethod
    def _join_units(
        units: list[tuple[str, int, int]],
    ) -> str:

        return " ".join(
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
                "sentence_segmenter": (
                    "regex_baseline_v1"
                ),
            },
        )