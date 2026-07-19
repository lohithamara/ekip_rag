from pathlib import Path

from ingestion.parsers.registry import ParserRegistry
from ingestion.schemas.documents import (
    DocumentMetadata,
    RawDocument,
)
from ingestion.storage.s3_client import S3Client
from ingestion.workspace.temporary import TemporaryWorkspace


class IngestionProcessor:

    def __init__(
        self,
        s3_client: S3Client,
        parser_registry: ParserRegistry,
    ):
        self.s3_client = s3_client
        self.parser_registry = parser_registry

    def process(
        self,
        s3_key: str,
        tenant_id: str,
        department: str,
        content_hash: str,
        document_id: int,
    ) -> RawDocument:

        filename = Path(s3_key).name
        file_path = Path(filename)

        metadata = DocumentMetadata(
            document_id=str(document_id),
            tenant_id=tenant_id,
            department=department,
            filename=filename,
            file_type=file_path.suffix.lower(),
            content_type=None,
            original_s3_key=s3_key,
            content_hash=content_hash,
        )

        with TemporaryWorkspace() as workspace:
            local_file = workspace / filename

            self.s3_client.download_file(
                s3_key=s3_key,
                local_path=str(local_file),
            )

            parser = self.parser_registry.get_parser(local_file)

            return parser.parse(
                file_path=local_file,
                metadata=metadata,
            )