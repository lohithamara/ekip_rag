import os
from io import BytesIO
from pathlib import Path

import fitz
import pytesseract
from dotenv import load_dotenv
from PIL import Image

from ingestion.parsers.base import BaseParser
from ingestion.schemas.documents import (
    DocumentMetadata,
    ExtractedPage,
    ExtractedTable,
    RawDocument,
)


load_dotenv()


class PDFParser(BaseParser):

    MIN_NATIVE_TEXT_CHARS = 50
    OCR_DPI = 300

    def __init__(self):
        tesseract_cmd = os.getenv("TESSERACT_CMD")

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = (
                tesseract_cmd
            )

    @property
    def supported_extensions(self) -> set[str]:
        return {".pdf"}

    def parse(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> RawDocument:

        document = self._open_document(file_path)

        try:
            self._extract_pdf_metadata(
                document,
                metadata,
            )

            pages = []
            tables = []
            warnings = []
            ocr_page_count = 0

            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                page_number = page_index + 1

                extracted_page = self._extract_page(
                    page=page,
                    page_number=page_number,
                    warnings=warnings,
                )

                if extracted_page.ocr_used:
                    ocr_page_count += 1

                page_tables = self._extract_page_tables(
                    page=page,
                    page_number=page_number,
                    starting_table_index=len(tables),
                    warnings=warnings,
                )

                extracted_page.metadata["table_count"] = (
                    len(page_tables)
                )

                pages.append(extracted_page)
                tables.extend(page_tables)

            metadata.page_count = document.page_count
            metadata.ocr_used = ocr_page_count > 0

            metadata.extra.update(
                {
                    "ocr_page_count": ocr_page_count,
                    "native_page_count": (
                        document.page_count
                        - ocr_page_count
                    ),
                    "table_count": len(tables),
                }
            )

            raw_text = "\n\n".join(
                self._serialize_page(page)
                for page in pages
            )

            if not raw_text.strip():
                warnings.append(
                    "PDF contains no extractable text."
                )

            return RawDocument(
                metadata=metadata,
                raw_text=raw_text,
                pages=pages,
                tables=tables,
                parser_name="PDFParser",
                parser_version="1.0",
                warnings=warnings,
            )

        finally:
            document.close()

    @staticmethod
    def _open_document(
        file_path: Path,
    ) -> fitz.Document:

        try:
            return fitz.open(file_path)

        except Exception as exc:
            raise ValueError(
                f"Unable to open PDF: {file_path}"
            ) from exc

    def _extract_page(
        self,
        page: fitz.Page,
        page_number: int,
        warnings: list[str],
    ) -> ExtractedPage:

        native_text = self._normalize_text(
            page.get_text("text")
        )

        ocr_required = self._requires_ocr(
            native_text
        )

        page_text = native_text
        ocr_used = False
        ocr_attempted = False
        ocr_succeeded = False
        page_warnings = []

        if ocr_required:
            ocr_attempted = True

            try:
                ocr_text = self._normalize_text(
                    self._perform_ocr(page)
                )

                if ocr_text:
                    page_text = ocr_text
                    ocr_used = True
                    ocr_succeeded = True

                else:
                    warning = (
                        "OCR returned no text on page "
                        f"{page_number}"
                    )

                    warnings.append(warning)
                    page_warnings.append(warning)

            except Exception as exc:
                warning = (
                    f"OCR failed on page "
                    f"{page_number}: {exc}"
                )

                warnings.append(warning)
                page_warnings.append(warning)

        return ExtractedPage(
            page_number=page_number,
            text=page_text,
            ocr_used=ocr_used,
            metadata={
                "native_text_char_count": len(
                    native_text
                ),
                "final_text_char_count": len(
                    page_text
                ),
                "ocr_required": ocr_required,
                "ocr_attempted": ocr_attempted,
                "ocr_succeeded": ocr_succeeded,
                "table_count": 0,
                "warnings": page_warnings,
            },
        )

    @classmethod
    def _requires_ocr(
        cls,
        native_text: str,
    ) -> bool:

        meaningful_char_count = sum(
            character.isalnum()
            for character in native_text
        )

        return (
            meaningful_char_count
            < cls.MIN_NATIVE_TEXT_CHARS
        )

    @classmethod
    def _perform_ocr(
        cls,
        page: fitz.Page,
    ) -> str:

        pixmap = page.get_pixmap(
            dpi=cls.OCR_DPI,
            alpha=False,
        )

        with Image.open(
            BytesIO(pixmap.tobytes("png"))
        ) as image:
            return pytesseract.image_to_string(
                image
            )

    def _extract_page_tables(
        self,
        page: fitz.Page,
        page_number: int,
        starting_table_index: int,
        warnings: list[str],
    ) -> list[ExtractedTable]:

        try:
            detected_tables = page.find_tables().tables

        except Exception as exc:
            warnings.append(
                f"Table detection failed on page "
                f"{page_number}: {exc}"
            )
            return []

        tables = []

        for local_index, detected_table in enumerate(
            detected_tables
        ):
            try:
                matrix = detected_table.extract()

                if not matrix:
                    continue

                matrix = [
                    [
                        self._normalize_cell(cell)
                        for cell in row
                    ]
                    for row in matrix
                ]

                headers = matrix[0]
                rows = matrix[1:]

                tables.append(
                    ExtractedTable(
                        table_id=(
                            f"table_"
                            f"{starting_table_index + local_index}"
                        ),
                        page_number=page_number,
                        headers=headers,
                        rows=rows,
                        metadata={
                            "source_type": "pdf",
                            "row_count": len(rows),
                            "column_count": len(headers),
                        },
                    )
                )

            except Exception as exc:
                warnings.append(
                    f"Table extraction failed on page "
                    f"{page_number}: {exc}"
                )

        return tables

    @staticmethod
    def _normalize_cell(value) -> str:
        return (
            ""
            if value is None
            else " ".join(str(value).split())
        )

    @staticmethod
    def _normalize_text(text: str) -> str:

        return "\n".join(
            normalized_line
            for line in text.splitlines()
            if (
                normalized_line
                := " ".join(line.split())
            )
        )

    @staticmethod
    def _serialize_page(
        page: ExtractedPage,
    ) -> str:

        return (
            f"Page {page.page_number}\n"
            f"{page.text}"
        )

    @staticmethod
    def _extract_pdf_metadata(
        document: fitz.Document,
        metadata: DocumentMetadata,
    ) -> None:

        pdf_metadata = document.metadata or {}

        if pdf_metadata.get("title"):
            metadata.title = pdf_metadata["title"]

        if pdf_metadata.get("author"):
            metadata.author = pdf_metadata["author"]

        metadata.extra.update(
            {
                "subject": pdf_metadata.get("subject"),
                "keywords": pdf_metadata.get("keywords"),
                "creator": pdf_metadata.get("creator"),
                "producer": pdf_metadata.get("producer"),
                "pdf_format": pdf_metadata.get("format"),
            }
        )