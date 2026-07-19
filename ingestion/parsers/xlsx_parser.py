from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd

from ingestion.parsers.base import BaseParser
from ingestion.schemas.documents import (
    DocumentMetadata,
    ExtractedTable,
    RawDocument,
)


class XLSXParser(BaseParser):

    @property
    def supported_extensions(self) -> set[str]:
        return {".xlsx"}

    def parse(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> RawDocument:

        worksheets = self._read_workbook(file_path)

        tables = []
        text_sections = []
        warnings = []

        metadata.extra.update(
            {
                "sheet_count": len(worksheets),
                "sheet_names": list(worksheets),
            }
        )

        for sheet_index, (sheet_name, dataframe) in enumerate(
            worksheets.items()
        ):
            dataframe = self._prepare_dataframe(dataframe)

            if dataframe.empty and not len(dataframe.columns):
                warnings.append(
                    f"Skipped empty worksheet: {sheet_name}"
                )
                continue

            headers = [
                self._normalize_header(column, index)
                for index, column in enumerate(
                    dataframe.columns,
                    start=1,
                )
            ]

            dataframe.columns = headers
            rows = dataframe.values.tolist()

            table = ExtractedTable(
                table_id=f"sheet_{sheet_index}",
                page_number=None,
                headers=headers,
                rows=rows,
                metadata={
                    "source_type": "xlsx",
                    "sheet_name": sheet_name,
                    "sheet_index": sheet_index,
                    "row_count": len(rows),
                    "column_count": len(headers),
                },
            )

            tables.append(table)

            text_sections.append(
                self._build_text(
                    sheet_name=sheet_name,
                    headers=headers,
                    rows=rows,
                )
            )

        metadata.extra.update(
            {
                "extracted_table_count": len(tables),
                "total_data_rows": sum(
                    len(table.rows)
                    for table in tables
                ),
            }
        )

        if not tables:
            warnings.append(
                "Workbook contains no non-empty worksheets."
            )

        return RawDocument(
            metadata=metadata,
            raw_text="\n\n".join(text_sections),
            tables=tables,
            parser_name="XLSXParser",
            parser_version="1.0",
            warnings=warnings,
        )

    @staticmethod
    def _read_workbook(
        file_path: Path,
    ) -> dict[str, pd.DataFrame]:

        try:
            return pd.read_excel(
                file_path,
                sheet_name=None,
                engine="openpyxl",
                dtype=object,
            )

        except Exception as exc:
            raise ValueError(
                f"Unable to read XLSX workbook: {file_path}"
            ) from exc

    @classmethod
    def _prepare_dataframe(
        cls,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        dataframe = (
            dataframe
            .dropna(axis=0, how="all")
            .dropna(axis=1, how="all")
            .reset_index(drop=True)
        )

        return dataframe.map(cls._normalize_value)

    @staticmethod
    def _normalize_header(
        header: Any,
        column_number: int,
    ) -> str:

        if pd.isna(header):
            return f"unnamed_column_{column_number}"

        header = str(header).strip()

        return (
            header
            or f"unnamed_column_{column_number}"
        )

    @staticmethod
    def _normalize_value(
        value: Any,
    ) -> Any:

        if pd.isna(value):
            return None

        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, AttributeError):
                pass

        if isinstance(value, (datetime, date, time)):
            return value.isoformat()

        return value

    @staticmethod
    def _build_text(
        sheet_name: str,
        headers: list[str],
        rows: list[list[Any]],
    ) -> str:

        lines = [
            f"Worksheet: {sheet_name}",
            f"Columns: {' | '.join(headers)}",
            "",
        ]

        if not rows:
            lines.append("No data rows.")
            return "\n".join(lines)

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