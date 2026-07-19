from pathlib import Path

from ingestion.parsers.base import BaseParser
from ingestion.schemas.documents import (
    DocumentMetadata,
    RawDocument,
)


class MarkdownParser(BaseParser):

    @property
    def supported_extensions(self) -> set[str]:
        return {".md"}

    def parse(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> RawDocument:

        text = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        return RawDocument(
            metadata=metadata,
            raw_text=text,
            parser_name="MarkdownParser",
            parser_version="1.0",
        )