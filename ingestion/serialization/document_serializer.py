from importlib import metadata
import json
from dataclasses import asdict
from pathlib import Path

from ingestion.schemas.documents import (
    CleanDocument,
    DocumentMetadata,
    ExtractedPage,
    ExtractedTable,
    RawDocument,
)


class DocumentSerializer:

    @staticmethod
    def save_raw_document(
        document: RawDocument,
        output_path: Path,
    ) -> None:

        DocumentSerializer._save_document(
            document=document,
            output_path=output_path,
        )

    @staticmethod
    def save_clean_document(
        document: CleanDocument,
        output_path: Path,
    ) -> None:

        DocumentSerializer._save_document(
            document=document,
            output_path=output_path,
        )

    @staticmethod
    def _save_document(
        document,
        output_path: Path,
    ) -> None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                asdict(document),
                file,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def load_raw_document(
        input_path: Path,
    ) -> RawDocument:

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        metadata = DocumentMetadata(
            **data["metadata"]
        )

        pages = [
            ExtractedPage(**page)
            for page in data.get("pages", [])
        ]

        tables = [
            ExtractedTable(**table)
            for table in data.get("tables", [])
        ]

        return RawDocument(
            metadata=metadata,
            raw_text=data["raw_text"],
            pages=pages,
            tables=tables,
            parser_name=data.get("parser_name"),
            parser_version=data.get("parser_version"),
            warnings=data.get("warnings", []),
        )
    @staticmethod
    def load_clean_document(
        input_path: Path,
    ) -> CleanDocument:

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

            metadata = DocumentMetadata(
            **data["metadata"]
            )

            pages = [
                ExtractedPage(**page)
                for page in data.get("pages", [])
            ]

            tables = [
                ExtractedTable(**table)
                for table in data.get("tables", [])
            ]

        return CleanDocument(
            metadata=metadata,
            clean_text=data["clean_text"],
            pages=pages,
            tables=tables,
            cleaning_steps=data.get(
                "cleaning_steps",
                [],
            ),
            warnings=data.get(
                "warnings",
                [],
            ),
        )