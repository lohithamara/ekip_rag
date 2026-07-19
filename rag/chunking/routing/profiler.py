import re

from ingestion.schemas.documents import CleanDocument

from rag.chunking.routing.schemas import (
    DocumentProfile,
)
from rag.chunking.tokenization.base import (
    BaseTokenCounter,
)


class DocumentProfiler:

    MARKDOWN_HEADING = re.compile(
        r"^#{1,6}\s+\S.+$"
    )

    NUMBERED_HEADING = re.compile(
        r"^(?:\d+\.)+\s*\S.+$"
    )

    UPPERCASE_HEADING = re.compile(
        r"^[A-Z][A-Z0-9\s/&(),:'-]{2,100}$"
    )

    PARAGRAPH_PATTERN = re.compile(
        r"\S(?:.*?\S)?(?=\n\s*\n|\Z)",
        flags=re.DOTALL,
    )

    def __init__(
        self,
        token_counter: BaseTokenCounter,
        short_document_threshold: int = 512,
    ):
        if short_document_threshold <= 0:

            raise ValueError(
                "short_document_threshold "
                "must be greater than zero."
            )

        self.token_counter = token_counter

        self.short_document_threshold = (
            short_document_threshold
        )

    def profile(
        self,
        document: CleanDocument,
    ) -> DocumentProfile:

        text = document.clean_text or ""

        character_count = len(text)

        token_count = self.token_counter.count(
            text
        )

        pages = document.pages or []

        tables = document.tables or []

        page_count = len(pages)

        pages_with_text = sum(
            1
            for page in pages
            if page.text
            and page.text.strip()
        )

        table_count = len(tables)

        tables_with_rows = sum(
            1
            for table in tables
            if self._table_has_rows(table)
        )

        heading_count = (
            self._count_headings(text)
        )

        paragraph_token_counts = (
            self._get_paragraph_token_counts(
                text
            )
        )

        paragraph_count = len(
            paragraph_token_counts
        )

        if paragraph_token_counts:

            average_paragraph_tokens = (
                sum(paragraph_token_counts)
                / paragraph_count
            )

        else:

            average_paragraph_tokens = 0.0

        has_pages = pages_with_text > 0

        has_tables = tables_with_rows > 0

        structure_confidence = (
            self._calculate_structure_confidence(
                text=text,
                heading_count=heading_count,
                paragraph_count=paragraph_count,
            )
        )

        has_structure = (
            structure_confidence >= 0.5
        )

        is_short_document = (
            token_count
            <= self.short_document_threshold
        )

        return DocumentProfile(
            document_id=(
                document.metadata.document_id
            ),
            file_type=(
                document.metadata.file_type.lower()
            ),
            character_count=character_count,
            token_count=token_count,
            page_count=page_count,
            pages_with_text=pages_with_text,
            table_count=table_count,
            tables_with_rows=tables_with_rows,
            heading_count=heading_count,
            paragraph_count=paragraph_count,
            average_paragraph_tokens=(
                average_paragraph_tokens
            ),
            has_pages=has_pages,
            has_tables=has_tables,
            has_structure=has_structure,
            is_short_document=is_short_document,
            structure_confidence=(
                structure_confidence
            ),
            metadata={
                "profiler_version": "1.0",
                "token_counter": (
                    self.token_counter.name
                ),
                "short_document_threshold": (
                    self.short_document_threshold
                ),
            },
        )

    @staticmethod
    def _table_has_rows(
        table,
    ) -> bool:

        rows = getattr(
            table,
            "rows",
            None,
        )

        return bool(rows)

    def _count_headings(
        self,
        text: str,
    ) -> int:

        count = 0

        for raw_line in text.splitlines():

            line = raw_line.strip()

            if not line:
                continue

            if self._is_heading(line):
                count += 1

        return count

    def _is_heading(
        self,
        line: str,
    ) -> bool:

        if self.MARKDOWN_HEADING.match(line):

            return True

        if self.NUMBERED_HEADING.match(line):

            return True

        if (
            self.UPPERCASE_HEADING.match(line)
            and self._is_reasonable_heading(line)
        ):

            return True

        return False

    @staticmethod
    def _is_reasonable_heading(
        text: str,
    ) -> bool:

        words = text.split()

        return (
            1 <= len(words) <= 15
            and len(text) <= 120
        )

    def _get_paragraph_token_counts(
        self,
        text: str,
    ) -> list[int]:

        counts = []

        for match in (
            self.PARAGRAPH_PATTERN.finditer(
                text
            )
        ):

            paragraph = (
                match.group().strip()
            )

            if not paragraph:
                continue

            counts.append(
                self.token_counter.count(
                    paragraph
                )
            )

        return counts

    @staticmethod
    def _calculate_structure_confidence(
        text: str,
        heading_count: int,
        paragraph_count: int,
    ) -> float:

        if not text.strip():
            return 0.0

        score = 0.0

        if heading_count >= 1:
            score += 0.25

        if heading_count >= 3:
            score += 0.25

        if paragraph_count >= 2:
            score += 0.20

        if paragraph_count >= 5:
            score += 0.15

        if (
            heading_count > 0
            and paragraph_count > heading_count
        ):
            score += 0.15

        return min(score, 1.0)