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
        s3_key = self._upload_original(
            local_file,
            tenant_id,
            department,
            content_hash,
        )

        raw_document = self._extract_raw(
            s3_key,
            tenant_id,
            department,
            content_hash,
        )

        clean_document = self._clean(
            raw_document,
        )

        chunks = self._chunk(
            clean_document,
        )

        embeddings = self._embed(
            chunks,
        )

        self._index(
            chunks,
            embeddings,
        )

        