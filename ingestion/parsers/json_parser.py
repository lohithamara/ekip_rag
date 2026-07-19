import json
from pathlib import Path

from ingestion.parsers.base import BaseParser
from ingestion.schemas.documents import (
    DocumentMetadata,
    RawDocument,
)


class JSONParser(BaseParser):

    @property
    def supported_extensions(self) -> set[str]:
        return {".json"}

    def parse(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> RawDocument:

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        formatted_text = json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )

        metadata.extra["json_root_type"] = type(data).__name__

        if isinstance(data, dict):
            metadata.extra["top_level_keys"] = list(data.keys())

        return RawDocument(
            metadata=metadata,
            raw_text=formatted_text,
            parser_name="JSONParser",
            parser_version="1.0",
        )