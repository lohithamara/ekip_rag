from pathlib import Path
from typing import Iterator

from docx import Document
from docx.document import Document as DocumentObject
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

from ingestion.parsers.base import BaseParser
from ingestion.schemas.documents import (
    DocumentMetadata,
    ExtractedTable,
    RawDocument,
)


class DOCXParser(BaseParser):

    @property
    def supported_extensions(self) -> set[str]:
        return {".docx"}

    def parse(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> RawDocument:

        document = self._read_document(file_path)
        self._extract_core_properties(document, metadata)

        sections = []
        tables = []
        paragraph_count = 0
        heading_count = 0
        list_item_count = 0

        for block in self._iter_block_items(document):

            if isinstance(block, Paragraph):
                text = self._normalize_text(block.text)

                if not text:
                    continue

                paragraph_count += 1
                style_name = (
                    block.style.name
                    if block.style is not None
                    else ""
                )

                if style_name.lower().startswith("heading"):
                    heading_count += 1
                    text = self._serialize_heading(
                        text,
                        style_name,
                    )

                elif self._is_list_paragraph(block):
                    list_item_count += 1
                    text = f"- {text}"

                sections.append(text)

            else:
                table = self._extract_table(
                    block,
                    len(tables),
                )

                tables.append(table)
                sections.append(
                    self._serialize_table(table)
                )

        metadata.extra.update(
            {
                "paragraph_count": paragraph_count,
                "heading_count": heading_count,
                "list_item_count": list_item_count,
                "table_count": len(tables),
            }
        )

        warnings = []

        if not sections:
            warnings.append(
                "DOCX document contains no extractable text or tables."
            )

        return RawDocument(
            metadata=metadata,
            raw_text="\n\n".join(sections),
            tables=tables,
            parser_name="DOCXParser",
            parser_version="1.0",
            warnings=warnings,
        )

    @staticmethod
    def _read_document(
        file_path: Path,
    ) -> DocumentObject:

        try:
            return Document(file_path)

        except Exception as exc:
            raise ValueError(
                f"Unable to read DOCX file: {file_path}"
            ) from exc

    @staticmethod
    def _iter_block_items(
        parent: DocumentObject | _Cell,
    ) -> Iterator[Paragraph | Table]:

        parent_element = parent.element.body

        for child in parent_element.iterchildren():
            if child.tag.endswith("}p"):
                yield Paragraph(child, parent)

            elif child.tag.endswith("}tbl"):
                yield Table(child, parent)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _serialize_heading(
        text: str,
        style_name: str,
    ) -> str:

        try:
            level = int(style_name.split()[-1])
        except (ValueError, IndexError):
            level = 1

        level = min(max(level, 1), 6)

        return f"{'#' * level} {text}"

    @staticmethod
    def _is_list_paragraph(
        paragraph: Paragraph,
    ) -> bool:

        style_name = (
            paragraph.style.name.lower()
            if paragraph.style is not None
            else ""
        )

        numbering = (
            paragraph._p.pPr is not None
            and paragraph._p.pPr.numPr is not None
        )

        return "list" in style_name or numbering

    def _extract_table(
        self,
        table: Table,
        table_index: int,
    ) -> ExtractedTable:

        matrix = [
            [
                self._normalize_text(cell.text)
                for cell in row.cells
            ]
            for row in table.rows
        ]

        headers = matrix[0] if matrix else []
        rows = matrix[1:] if matrix else []

        return ExtractedTable(
            table_id=f"table_{table_index}",
            page_number=None,
            headers=headers,
            rows=rows,
            metadata={
                "source_type": "docx",
                "row_count": len(rows),
                "column_count": len(headers),
            },
        )

    @staticmethod
    def _serialize_table(
        table: ExtractedTable,
    ) -> str:

        lines = [
            f"Table: {table.table_id}",
            f"Columns: {' | '.join(table.headers)}",
        ]

        lines.extend(
            f"Row {row_number}: "
            + " | ".join(
                f"{header}: {value}"
                for header, value in zip(
                    table.headers,
                    row,
                )
            )
            for row_number, row in enumerate(
                table.rows,
                start=1,
            )
        )

        return "\n".join(lines)

    @staticmethod
    def _extract_core_properties(
        document: DocumentObject,
        metadata: DocumentMetadata,
    ) -> None:

        properties = document.core_properties

        if properties.title:
            metadata.title = properties.title

        if properties.author:
            metadata.author = properties.author

        metadata.extra.update(
            {
                "subject": properties.subject or None,
                "keywords": properties.keywords or None,
                "category": properties.category or None,
            }
        )