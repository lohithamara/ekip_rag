import re
from dataclasses import dataclass

from ingestion.schemas.documents import CleanDocument

from rag.chunking.base import BaseChunkingStrategy
from rag.chunking.chunk_factory import ChunkFactory
from rag.chunking.config import ChunkingConfig
from rag.chunking.schemas import ChunkingResult
from rag.chunking.tokenization.base import BaseTokenCounter


@dataclass
class SectionUnit:

    heading: str | None

    level: int | None

    section_path: list[str]

    text: str

    start_char: int

    end_char: int


class StructureAwareChunkingStrategy(
    BaseChunkingStrategy
):

    MARKDOWN_HEADING_PATTERN = re.compile(
        r"^(#{1,6})\s+(.+?)\s*$"
    )

    NUMBERED_HEADING_PATTERN = re.compile(
        r"^("
        r"(?:\d+\.)+\d*"
        r"|"
        r"\d+"
        r")"
        r"[\s.)-]+"
        r"(.+?)\s*$"
    )

    UPPERCASE_HEADING_PATTERN = re.compile(
        r"^[A-Z][A-Z0-9\s/&(),:'-]{2,100}$"
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
    STRATEGY_NAME = "structure_aware"

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
                sections_detected=0,
            )

        sections = self._extract_sections(text)

        atomic_units = []

        for section in sections:

            atomic_units.extend(
                self._split_section(section)
            )

        groups = self._group_units(
            atomic_units
        )

        groups = self._handle_small_tail(
            groups
        )

        chunks = []

        for chunk_index, group in enumerate(groups):

            chunk_text = self._join_units(group)

            section_path = self._resolve_group_section_path(
                group
            )

            start_char = min(
                unit.start_char
                for unit in group
            )

            end_char = max(
                unit.end_char
                for unit in group
            )

            headings = list(
                dict.fromkeys(
                    unit.heading
                    for unit in group
                    if unit.heading
                )
            )

            chunks.append(
                self.chunk_factory.create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    section_path=section_path,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "length_unit": (
                            self.config.length_unit
                        ),
                        "headings": headings,
                        "section_count": len(group),
                        "structure_source": (
                            "text_heading_detection"
                        ),
                        "token_counter": (
                            self.token_counter.name
                        ),
                    },
                )
            )

        return self._build_result(
            document=document,
            chunks=chunks,
            sections_detected=len(sections),
        )

    def _extract_sections(
        self,
        text: str,
    ) -> list[SectionUnit]:

        lines = list(
            re.finditer(
                r".*(?:\n|$)",
                text,
            )
        )

        sections = []

        heading_stack: list[
            tuple[int, str]
        ] = []

        current_heading = None

        current_level = None

        current_path = []

        current_start = 0

        current_parts = []

        def flush_section(
            end_char: int,
        ) -> None:

            nonlocal current_parts
            nonlocal current_start

            section_text = "".join(
                current_parts
            ).strip()

            if section_text:

                sections.append(
                    SectionUnit(
                        heading=current_heading,
                        level=current_level,
                        section_path=list(
                            current_path
                        ),
                        text=section_text,
                        start_char=current_start,
                        end_char=end_char,
                    )
                )

            current_parts = []

        for match in lines:

            raw_line = match.group()

            line = raw_line.strip()

            if not line:
                current_parts.append(raw_line)
                continue

            heading = self._detect_heading(line)

            if heading is None:

                current_parts.append(raw_line)

                continue

            level, heading_text = heading

            if current_parts:

                flush_section(
                    match.start()
                )

            while (
                heading_stack
                and heading_stack[-1][0] >= level
            ):
                heading_stack.pop()

            heading_stack.append(
                (
                    level,
                    heading_text,
                )
            )

            current_heading = heading_text

            current_level = level

            current_path = [
                item[1]
                for item in heading_stack
            ]

            current_start = match.start()

            current_parts = [raw_line]

        if current_parts:

            flush_section(
                len(text)
            )

        if not sections:

            sections.append(
                SectionUnit(
                    heading=None,
                    level=None,
                    section_path=[],
                    text=text,
                    start_char=0,
                    end_char=len(text),
                )
            )

        return sections

    def _detect_heading(
        self,
        line: str,
    ) -> tuple[int, str] | None:

        markdown_match = (
            self.MARKDOWN_HEADING_PATTERN.match(
                line
            )
        )

        if markdown_match:

            level = len(
                markdown_match.group(1)
            )

            heading = (
                markdown_match.group(2).strip()
            )

            return level, heading

        numbered_match = (
            self.NUMBERED_HEADING_PATTERN.match(
                line
            )
        )

        if numbered_match:

            number = numbered_match.group(1)

            heading = numbered_match.group(2).strip()

            level = min(
                number.count(".") + 1,
                6,
            )

            if self._is_reasonable_heading(
                heading
            ):
                return level, heading

        if (
            self.UPPERCASE_HEADING_PATTERN.match(
                line
            )
            and self._is_reasonable_heading(line)
        ):

            return 1, line

        return None

    @staticmethod
    def _is_reasonable_heading(
        text: str,
    ) -> bool:

        words = text.split()

        return (
            1 <= len(words) <= 15
            and len(text) <= 120
        )

    def _split_section(
        self,
        section: SectionUnit,
    ) -> list[SectionUnit]:

        if (
            self.token_counter.count(section.text)
            <= self.config.chunk_size
        ):

            return [section]

        if self.config.length_unit == "characters":

            return self._split_section_by_characters(
                section
            )

        return self._split_section_by_tokens(
            section
        )

    def _split_section_by_characters(
        self,
        section: SectionUnit,
    ) -> list[SectionUnit]:

        units = []

        local_start = 0

        while local_start < len(section.text):

            local_end = min(
                local_start + self.config.chunk_size,
                len(section.text),
            )

            piece = section.text[
                local_start:local_end
            ]

            if piece.strip():

                units.append(
                    SectionUnit(
                        heading=section.heading,
                        level=section.level,
                        section_path=list(
                            section.section_path
                        ),
                        text=piece,
                        start_char=(
                            section.start_char
                            + local_start
                        ),
                        end_char=(
                            section.start_char
                            + local_end
                        ),
                    )
                )

            if local_end >= len(section.text):
                break

            local_start = local_end

        return units

    def _split_section_by_tokens(
        self,
        section: SectionUnit,
    ) -> list[SectionUnit]:

        token_spans = [
            match.span()
            for match in re.finditer(
                r"\S+",
                section.text,
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

            units.append(
                SectionUnit(
                    heading=section.heading,
                    level=section.level,
                    section_path=list(
                        section.section_path
                    ),
                    text=section.text[
                        local_start:local_end
                    ],
                    start_char=(
                        section.start_char
                        + local_start
                    ),
                    end_char=(
                        section.start_char
                        + local_end
                    ),
                )
            )

            if token_end >= len(token_spans):
                break

            token_start = token_end

        return units

    def _group_units(
        self,
        units: list[SectionUnit],
    ) -> list[list[SectionUnit]]:

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

                current_group = []

            current_group.append(unit)

        if current_group:
            groups.append(current_group)

        return groups

    def _handle_small_tail(
        self,
        groups: list[list[SectionUnit]],
    ) -> list[list[SectionUnit]]:

        if len(groups) < 2:
            return groups

        tail = groups[-1]

        if (
            self.token_counter.count(
                self._join_units(tail)
            )
            >= self.config.min_chunk_size
        ):
            return groups

        if not self.config.merge_small_chunks:
            return groups

        previous = groups[-2]

        # Do not merge unrelated sections merely to
        # eliminate a small chunk.
        if (
            self._resolve_group_section_path(previous)
            != self._resolve_group_section_path(tail)
        ):
            return groups

        combined = previous + tail

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
        units: list[SectionUnit],
    ) -> str:

        return "\n\n".join(
            unit.text.strip()
            for unit in units
            if unit.text.strip()
        )

    @staticmethod
    def _resolve_group_section_path(
        units: list[SectionUnit],
    ) -> list[str]:

        paths = [
            unit.section_path
            for unit in units
            if unit.section_path
        ]

        if not paths:
            return []

        common_path = list(paths[0])

        for path in paths[1:]:

            common_length = 0

            for left, right in zip(
                common_path,
                path,
            ):

                if left != right:
                    break

                common_length += 1

            common_path = common_path[
                :common_length
            ]

            if not common_path:
                break

        return common_path

    def _build_result(
        self,
        document: CleanDocument,
        chunks: list,
        sections_detected: int,
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
                "sections_detected": (
                    sections_detected
                ),
                "structure_source": (
                    "text_heading_detection"
                ),
            },
        )