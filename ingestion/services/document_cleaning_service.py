from copy import deepcopy

from ingestion.cleaning.pipeline import (
    TextCleaningPipeline,
)
from ingestion.schemas.documents import (
    CleanDocument,
    RawDocument,
)


class DocumentCleaningService:

    def __init__(
        self,
        cleaning_pipeline: TextCleaningPipeline,
    ):
        self.cleaning_pipeline = cleaning_pipeline

    def clean_document(
        self,
        raw_document: RawDocument,
    ) -> CleanDocument:

        metadata = deepcopy(
            raw_document.metadata
        )

        pages = deepcopy(
            raw_document.pages
        )

        tables = deepcopy(
            raw_document.tables
        )

        cleaned_pages = []

        for page in pages:

            cleaned_text, _ = (
                self.cleaning_pipeline.clean(
                    page.text
                )
            )

            page.text = cleaned_text

            cleaned_pages.append(page)

        for table in tables:

            table.headers = [
                self.cleaning_pipeline.clean(
                    str(header)
                )[0]
                for header in table.headers
            ]

            table.rows = [
                [
                    self.cleaning_pipeline.clean(
                        str(value)
                    )[0]
                    if value is not None
                    else None
                    for value in row
                ]
                for row in table.rows
            ]

        if cleaned_pages:

            clean_text = "\n\n".join(
                (
                    f"Page {page.page_number}\n"
                    f"{page.text}"
                )
                for page in cleaned_pages
                if page.text
            )

            cleaning_steps = self.cleaning_pipeline.step_names

        else:

            clean_text, cleaning_steps = (
                self.cleaning_pipeline.clean(
                    raw_document.raw_text
                )
            )

        metadata.extra["cleaning_version"] = "1.0"

        return CleanDocument(
            metadata=metadata,
            clean_text=clean_text,
            pages=cleaned_pages,
            tables=tables,
            cleaning_steps=cleaning_steps,
            warnings=deepcopy(
                raw_document.warnings
            ),
        )