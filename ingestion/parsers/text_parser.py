from pathlib import Path

from ingestion.parsers.base import BaseParser
from ingestion.schemas.documents import (
    DocumentMetadata,
    RawDocument,
)


class TextParser(BaseParser):

    @property
    def supported_extensions(self) -> set[str]:
        return {".txt"}

    def parse(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> RawDocument:

        text = file_path.read_text(
            encoding="utf-8"
        )

        return RawDocument(
            metadata=metadata,
            raw_text=text,
            parser_name="TextParser",
            parser_version="1.0",
        )