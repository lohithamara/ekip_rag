from abc import ABC
from abc import abstractmethod

from rag.documents.schemas import (
    DocumentRecord,
)

class DocumentRepository(ABC):

    @abstractmethod
    def create(
        self,
        document: DocumentRecord,
    ) -> DocumentRecord:
        ...

    @abstractmethod
    def get_by_id(
        self,
        document_id: int,
    ) -> DocumentRecord | None:
        ...

    @abstractmethod
    def get_by_filename(
        self,
        tenant_id: int,
        filename: str,
    ) -> DocumentRecord | None:
        ...

    @abstractmethod
    def latest_version(
        self,
        tenant_id: int,
        filename: str,
    ) -> DocumentRecord | None:
        ...

    @abstractmethod
    def list_documents(
        self,
        tenant_id: int,
        department_id: int | None = None,
    ) -> list[DocumentRecord]:
        ...

    @abstractmethod
    def exists(
        self,
        tenant_id: int,
        content_hash: str,
    ) -> bool:
        ...

    @abstractmethod
    def update(
        self,
        document: DocumentRecord,
    ) -> DocumentRecord:
        ...

    @abstractmethod
    def update_status(
        self,
        document_id: int,
        status: str,
    ) -> None:
        ...

    @abstractmethod
    def delete(
        self,
        document_id: int,
    ) -> None:
        ...