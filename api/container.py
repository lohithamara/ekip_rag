from dataclasses import dataclass

from ingestion.workers.document_ingestion_worker import DocumentIngestionWorker
from rag.documents.service import DocumentService
from rag.service.service import RAGService
from rag.vector_store.qdrant_store import QdrantVectorStore
from security.authorization.service import AuthorizationService

@dataclass
class ApplicationContainer:

    rag_service: RAGService

    vector_store: QdrantVectorStore

    document_service: DocumentService

    document_worker: DocumentIngestionWorker

    def close(self) -> None:

        self.vector_store.close()