from typing import Any

from ingestion.schemas.documents import (
    CleanDocument,
    ExtractedTable,
)

from rag.chunking.base import BaseChunkingStrategy
from rag.chunking.chunk_factory import ChunkFactory
from rag.chunking.config import ChunkingConfig
from rag.chunking.schemas import ChunkingResult
from rag.chunking.tokenization.base import BaseTokenCounter


class TableAwareChunkingStrategy(BaseChunkingStrategy):

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
    STRATEGY_NAME = "table_aware"

    @property
    def name(self) -> str:
        return self.STRATEGY_NAME

    def chunk(
        self,
        document: CleanDocument,
    ) -> ChunkingResult:

        if not document.tables:

            return self._build_result(
                document=document,
                chunks=[],
                tables_processed=0,
                tables_skipped=0,
            )

        chunks = []

        chunk_index = 0

        tables_processed = 0
        tables_skipped = 0

        for table in document.tables:

            table_id = self._get_table_id(table)

            headers = self._get_headers(table)

            rows = self._normalize_rows(
                self._get_rows(table)
            )

            if not rows:

                tables_skipped += 1
                continue

            page_numbers = self._get_page_numbers(
                table
            )

            if (
                self.config.table_chunking_mode
                == "whole_table"
            ):

                units = self._chunk_whole_table(
                    headers=headers,
                    rows=rows,
                )

            elif (
                self.config.table_chunking_mode
                == "row_group"
            ):

                units = self._chunk_row_groups(
                    headers=headers,
                    rows=rows,
                )

            else:

                raise ValueError(
                    "Unsupported table chunking mode: "
                    f"{self.config.table_chunking_mode}"
                )

            for unit in units:

                chunk = self.chunk_factory.create_chunk(
                    document=document,
                    text=unit["text"],
                    chunk_index=chunk_index,
                    chunk_type="table",
                    table_ids=[table_id],
                    page_numbers=page_numbers,
                    start_char=None,
                    end_char=None,
                    metadata={
                        "length_unit": (
                            self.config.length_unit
                        ),
                        "table_chunking_mode": (
                            self.config.table_chunking_mode
                        ),
                        "row_start": unit[
                            "row_start"
                        ],
                        "row_end": unit[
                            "row_end"
                        ],
                        "row_count": unit[
                            "row_count"
                        ],
                        "headers": headers,
                        "repeated_headers": (
                            self.config.repeat_table_headers
                        ),
                        "token_counter": (
                            self.token_counter.name
                        ),
                        "table_metadata": (
                            self._get_table_metadata(
                                table
                            )
                        ),
                    },
                )

                chunks.append(chunk)

                chunk_index += 1

            tables_processed += 1

        return self._build_result(
            document=document,
            chunks=chunks,
            tables_processed=tables_processed,
            tables_skipped=tables_skipped,
        )

    def _chunk_whole_table(
        self,
        headers: list[str],
        rows: list[list[str]],
    ) -> list[dict]:

        table_text = self._render_table(
            headers=headers,
            rows=rows,
            include_headers=True,
        )

        if (
            self.token_counter.count(table_text)
            <= self.config.chunk_size
        ):

            return [
                {
                    "text": table_text,
                    "row_start": 0,
                    "row_end": len(rows),
                    "row_count": len(rows),
                }
            ]

        # Never emit an oversized chunk.
        # Fall back to row grouping.

        return self._chunk_row_groups(
            headers=headers,
            rows=rows,
        )

    def _chunk_row_groups(
        self,
        headers: list[str],
        rows: list[list[str]],
    ) -> list[dict]:

        units = []

        current_rows = []
        current_start = 0

        row_index = 0

        while row_index < len(rows):

            row = rows[row_index]

            candidate_rows = (
                current_rows + [row]
            )

            candidate_text = self._render_table(
                headers=headers,
                rows=candidate_rows,
                include_headers=(
                    self.config.repeat_table_headers
                ),
            )

            within_row_limit = (
                len(candidate_rows)
                <= self.config.table_rows_per_chunk
            )

            within_size_limit = (
                self.token_counter.count(
                    candidate_text
                )
                <= self.config.chunk_size
            )

            if (
                current_rows
                and (
                    not within_row_limit
                    or not within_size_limit
                )
            ):

                units.append(
                    self._create_unit(
                        headers=headers,
                        rows=current_rows,
                        row_start=current_start,
                    )
                )

                current_rows = []

                current_start = row_index

                continue

            if (
                not current_rows
                and not within_size_limit
            ):

                oversized_units = (
                    self._split_oversized_row(
                        headers=headers,
                        row=row,
                        row_index=row_index,
                    )
                )

                units.extend(oversized_units)

                row_index += 1

                current_start = row_index

                continue

            current_rows.append(row)

            row_index += 1

        if current_rows:

            units.append(
                self._create_unit(
                    headers=headers,
                    rows=current_rows,
                    row_start=current_start,
                )
            )

        return units

    def _split_oversized_row(
        self,
        headers: list[str],
        row: list[str],
        row_index: int,
    ) -> list[dict]:

        # A single row may exceed chunk_size because
        # one cell contains a very large amount of text.
        #
        # Convert the row into labeled cell text and
        # split it safely.

        row_text = self._render_row_as_fields(
            headers=headers,
            row=row,
        )

        header_text = ""

        if (
            self.config.repeat_table_headers
            and headers
        ):

            header_text = (
                self._render_header(headers)
                + "\n"
            )

        available_size = self.config.chunk_size

        if header_text:

            available_size -= (
                self.token_counter.count(
                    header_text
                )
            )

        if available_size <= 0:

            raise ValueError(
                "Table headers alone exceed chunk_size."
            )

        pieces = self._hard_split_text(
            text=row_text,
            size=available_size,
        )

        units = []

        for piece_index, piece in enumerate(pieces):

            text = (
                header_text + piece
            ).strip()

            units.append(
                {
                    "text": text,
                    "row_start": row_index,
                    "row_end": row_index + 1,
                    "row_count": 1,
                    "oversized_row_split": True,
                    "row_piece_index": piece_index,
                }
            )

        return units

    def _create_unit(
        self,
        headers: list[str],
        rows: list[list[str]],
        row_start: int,
    ) -> dict:

        row_end = row_start + len(rows)

        return {
            "text": self._render_table(
                headers=headers,
                rows=rows,
                include_headers=(
                    self.config.repeat_table_headers
                ),
            ),
            "row_start": row_start,
            "row_end": row_end,
            "row_count": len(rows),
            "oversized_row_split": False,
        }

    def _render_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        include_headers: bool,
    ) -> str:

        lines = []

        if include_headers and headers:

            lines.append(
                self._render_header(headers)
            )

        for row in rows:

            lines.append(
                self._render_row(row)
            )

        return "\n".join(lines).strip()

    @staticmethod
    def _render_header(
        headers: list[str],
    ) -> str:

        return " | ".join(headers)

    @staticmethod
    def _render_row(
        row: list[str],
    ) -> str:

        return " | ".join(row)

    @staticmethod
    def _render_row_as_fields(
        headers: list[str],
        row: list[str],
    ) -> str:

        fields = []

        for index, value in enumerate(row):

            if index < len(headers):

                field_name = headers[index]

            else:

                field_name = f"column_{index + 1}"

            fields.append(
                f"{field_name}: {value}"
            )

        return "\n".join(fields)

    def _hard_split_text(
        self,
        text: str,
        size: int,
    ) -> list[str]:

        if self.config.length_unit == "characters":

            return [
                text[start:start + size]
                for start in range(
                    0,
                    len(text),
                    size,
                )
                if text[start:start + size].strip()
            ]

        words = text.split()

        return [
            " ".join(
                words[start:start + size]
            )
            for start in range(
                0,
                len(words),
                size,
            )
            if words[start:start + size]
        ]

    @staticmethod
    def _normalize_rows(
        rows: list[list[Any]],
    ) -> list[list[str]]:

        normalized = []

        for row in rows:

            if row is None:
                continue

            normalized_row = [
                ""
                if value is None
                else str(value).strip()
                for value in row
            ]

            if any(normalized_row):

                normalized.append(
                    normalized_row
                )

        return normalized

    @staticmethod
    def _get_table_id(
        table: ExtractedTable,
    ) -> str:

        table_id = getattr(
            table,
            "table_id",
            None,
        )

        if table_id is None:

            raise ValueError(
                "ExtractedTable is missing table_id."
            )

        return str(table_id)

    @staticmethod
    def _get_headers(
        table: ExtractedTable,
    ) -> list[str]:

        headers = getattr(
            table,
            "headers",
            None,
        )

        if headers is None:

            headers = getattr(
                table,
                "columns",
                [],
            )

        return [
            ""
            if value is None
            else str(value).strip()
            for value in headers
        ]

    @staticmethod
    def _get_rows(
        table: ExtractedTable,
    ) -> list[list[Any]]:

        rows = getattr(
            table,
            "rows",
            [],
        )

        return rows or []

    @staticmethod
    def _get_page_numbers(
        table: ExtractedTable,
    ) -> list[int]:

        page_numbers = getattr(
            table,
            "page_numbers",
            None,
        )

        if page_numbers:

            return sorted(
                set(page_numbers)
            )

        page_number = getattr(
            table,
            "page_number",
            None,
        )

        if page_number is not None:

            return [page_number]

        return []

    @staticmethod
    def _get_table_metadata(
        table: ExtractedTable,
    ) -> dict:

        metadata = getattr(
            table,
            "metadata",
            None,
        )

        return (
            dict(metadata)
            if metadata
            else {}
        )

    def _build_result(
        self,
        document: CleanDocument,
        chunks: list,
        tables_processed: int,
        tables_skipped: int,
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
                "table_chunking_mode": (
                    self.config.table_chunking_mode
                ),
                "table_rows_per_chunk": (
                    self.config.table_rows_per_chunk
                ),
                "repeat_table_headers": (
                    self.config.repeat_table_headers
                ),
                "token_counter": (
                    self.token_counter.name
                ),
                "source_table_count": len(
                    document.tables
                ),
                "tables_processed": (
                    tables_processed
                ),
                "tables_skipped": (
                    tables_skipped
                ),
            },
        )