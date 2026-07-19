from abc import ABC, abstractmethod

from ingestion.schemas.documents import CleanDocument
from rag.chunking.schemas import ChunkingResult


class BaseChunkingStrategy(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def chunk(
        self,
        document: CleanDocument,
    ) -> ChunkingResult:
        pass