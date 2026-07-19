from abc import ABC, abstractmethod
from pathlib import Path

from ingestion.schemas.documents import (
    DocumentMetadata,
    RawDocument,
)


class BaseParser(ABC):

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """
        File extensions supported by this parser.
        """

    @abstractmethod
    def parse(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> RawDocument:
        """
        Extract content from a document and return
        a standardized RawDocument.
        """