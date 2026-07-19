from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag

from ingestion.parsers.base import BaseParser
from ingestion.schemas.documents import (
    DocumentMetadata,
    ExtractedTable,
    RawDocument,
)


class HTMLParser(BaseParser):

    NOISE_TAGS = (
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
    )

    BLOCK_TAGS = (
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "li",
        "pre",
        "blockquote",
    )

    ENCODINGS = (
        "utf-8",
        "utf-8-sig",
        "latin-1",
    )

    @property
    def supported_extensions(self) -> set[str]:
        return {".html", ".htm"}

    def parse(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> RawDocument:

        warnings = []

        html = self._read_html(file_path, warnings)
        soup = BeautifulSoup(html, "lxml")

        self._extract_metadata(soup, metadata)
        self._remove_noise(soup)

        tables = self._extract_tables(soup)

        for table_tag in soup.find_all("table"):
            table_tag.decompose()

        text_sections = self._extract_text_blocks(soup)

        raw_text = "\n\n".join(
            text_sections
            + [
                self._serialize_table(table)
                for table in tables
            ]
        )

        metadata.extra.update(
            {
                "heading_count": len(
                    soup.find_all(
                        ["h1", "h2", "h3", "h4", "h5", "h6"]
                    )
                ),
                "paragraph_count": len(soup.find_all("p")),
                "table_count": len(tables),
                "link_count": len(soup.find_all("a")),
            }
        )

        if not raw_text.strip():
            warnings.append(
                "HTML document contains no extractable content."
            )

        return RawDocument(
            metadata=metadata,
            raw_text=raw_text,
            tables=tables,
            parser_name="HTMLParser",
            parser_version="1.0",
            warnings=warnings,
        )

    @classmethod
    def _read_html(
        cls,
        file_path: Path,
        warnings: list[str],
    ) -> str:

        last_exception = None

        for encoding in cls.ENCODINGS:
            try:
                content = file_path.read_text(
                    encoding=encoding
                )

                if encoding != "utf-8":
                    warnings.append(
                        "HTML decoded using fallback "
                        f"encoding: {encoding}"
                    )

                return content

            except UnicodeDecodeError as exc:
                last_exception = exc

        raise ValueError(
            f"Unable to decode HTML file: {file_path}"
        ) from last_exception

    @staticmethod
    def _extract_metadata(
        soup: BeautifulSoup,
        metadata: DocumentMetadata,
    ) -> None:

        if soup.title and soup.title.string:
            metadata.title = soup.title.string.strip()

        html_tag = soup.find("html")

        if html_tag and html_tag.get("lang"):
            metadata.language = html_tag["lang"].strip()

        description = soup.find(
            "meta",
            attrs={"name": "description"},
        )

        if description:
            metadata.extra["description"] = (
                description.get("content")
            )

        author = soup.find(
            "meta",
            attrs={"name": "author"},
        )

        if author and author.get("content"):
            metadata.author = author["content"].strip()

    @classmethod
    def _remove_noise(
        cls,
        soup: BeautifulSoup,
    ) -> None:

        for tag in soup.find_all(cls.NOISE_TAGS):
            tag.decompose()

    def _extract_tables(
        self,
        soup: BeautifulSoup,
    ) -> list[ExtractedTable]:

        return [
            self._parse_table(table_tag, table_index)
            for table_index, table_tag
            in enumerate(soup.find_all("table"))
        ]

    def _parse_table(
        self,
        table_tag: Tag,
        table_index: int,
    ) -> ExtractedTable:

        matrix = [
            [
                self._normalize_text(
                    cell.get_text(
                        separator=" ",
                        strip=True,
                    )
                )
                for cell in row_tag.find_all(["th", "td"])
            ]
            for row_tag in table_tag.find_all("tr")
        ]

        matrix = [
            row
            for row in matrix
            if row
        ]

        if not matrix:
            headers = []
            rows = []

        elif table_tag.find("th"):
            headers = matrix[0]
            rows = matrix[1:]

        else:
            column_count = max(map(len, matrix))

            headers = [
                f"column_{index}"
                for index in range(1, column_count + 1)
            ]

            rows = matrix

        return ExtractedTable(
            table_id=f"table_{table_index}",
            page_number=None,
            headers=headers,
            rows=rows,
            metadata={
                "source_type": "html",
                "row_count": len(rows),
                "column_count": len(headers),
            },
        )

    def _extract_text_blocks(
        self,
        soup: BeautifulSoup,
    ) -> list[str]:

        body = soup.body or soup
        sections = []

        for tag in body.find_all(self.BLOCK_TAGS):
            text = self._normalize_text(
                tag.get_text(
                    separator=" ",
                    strip=True,
                )
            )

            if not text:
                continue

            if tag.name.startswith("h"):
                text = f"{'#' * int(tag.name[1])} {text}"

            elif tag.name == "li":
                text = f"- {text}"

            elif tag.name == "pre":
                text = f"Code Block:\n{text}"

            elif tag.name == "blockquote":
                text = f"Quote: {text}"

            sections.append(text)

        return sections

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split())

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