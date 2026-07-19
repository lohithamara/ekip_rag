from pathlib import Path
from typing import Any

import pandas as pd

from ingestion.parsers.base import BaseParser
from ingestion.schemas.documents import (
    DocumentMetadata,
    ExtractedTable,
    RawDocument,
)


class CSVParser(BaseParser):

    ENCODINGS = (
        "utf-8",
        "utf-8-sig",
        "latin-1",
    )

    @property
    def supported_extensions(self) -> set[str]:
        return {".csv"}

    def parse(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> RawDocument:

        warnings = []
        dataframe = self._read_csv(file_path, warnings)
        dataframe = dataframe.map(self._normalize_value)

        headers = list(map(str, dataframe.columns))
        rows = dataframe.values.tolist()

        table = ExtractedTable(
            table_id="table_0",
            page_number=None,
            headers=headers,
            rows=rows,
            metadata={
                "source_type": "csv",
                "row_count": len(rows),
                "column_count": len(headers),
            },
        )

        metadata.extra.update(
            {
                "row_count": len(rows),
                "column_count": len(headers),
                "column_names": headers,
            }
        )

        return RawDocument(
            metadata=metadata,
            raw_text=self._build_text(
                headers=headers,
                rows=rows,
            ),
            tables=[table],
            parser_name="CSVParser",
            parser_version="1.0",
            warnings=warnings,
        )

    @classmethod
    def _read_csv(
        cls,
        file_path: Path,
        warnings: list[str],
    ) -> pd.DataFrame:

        last_exception = None

        for encoding in cls.ENCODINGS:
            try:
                dataframe = pd.read_csv(
                    file_path,
                    encoding=encoding,
                )

                if encoding != "utf-8":
                    warnings.append(
                        "CSV decoded using fallback "
                        f"encoding: {encoding}"
                    )

                return dataframe

            except UnicodeDecodeError as exc:
                last_exception = exc

        raise ValueError(
            f"Unable to decode CSV file: {file_path}"
        ) from last_exception

    @staticmethod
    def _normalize_value(value: Any) -> Any:

        if pd.isna(value):
            return None

        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, AttributeError):
                pass

        return value

    @staticmethod
    def _build_text(
        headers: list[str],
        rows: list[list[Any]],
    ) -> str:

        columns = " | ".join(headers)

        if not rows:
            return (
                "CSV Table\n"
                f"Columns: {columns}\n"
                "No data rows."
            )

        lines = [
            "CSV Table",
            f"Columns: {columns}",
            "",
        ]

        lines.extend(
            f"Row {row_number}: "
            + " | ".join(
                f"{header}: "
                f"{'' if value is None else value}"
                for header, value in zip(headers, row)
            )
            for row_number, row in enumerate(
                rows,
                start=1,
            )
        )

        return "\n".join(lines)