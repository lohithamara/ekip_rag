from pathlib import Path


class DocumentIngestionPipeline:

    def __init__(
        self,
        s3_client,
        ingestion_processor,
        cleaning_service,
        chunking_service,
        embedding_service,
        vector_store,
    ):

        self.s3_client = s3_client

        self.ingestion_processor = (
            ingestion_processor
        )

        self.cleaning_service = (
            cleaning_service
        )

        self.chunking_service = (
            chunking_service
        )

        self.embedding_service = (
            embedding_service
        )

        self.vector_store = (
            vector_store
        )

    def process(
        self,
        *,
        local_file: Path,
        tenant_id: str,
        department: str,
        content_hash: str,
    ):

        raise NotImplementedError