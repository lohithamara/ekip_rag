from pathlib import Path

from security.authentication.schemas import (
    AuthenticatedUser,
)

from rag.documents.service import (
    DocumentService,
)

from rag.documents.upload_schema import (
    UploadDocumentRequest,
    UploadDocumentResponse,
)

from rag.documents.schemas import (
    DocumentInfo,
)


class DocumentController:

    def __init__(
        self,
        service: DocumentService,
    ):
        self.service = service

    def upload_document(
        self,
        user: AuthenticatedUser,
        file_path: Path,
        original_filename: str,
        department: str,
    ) -> UploadDocumentResponse:

        request = UploadDocumentRequest(

            user=user,
            file_path=file_path,
            original_filename=original_filename,
            department=department,
        )

        return self.service.upload_document(
            request
        )

    def get_document(
        self,
        user: AuthenticatedUser,
        document_id: int,
    ) -> DocumentInfo:

        return self.service.get_document(
            user=user,
            document_id=document_id,
        )


    def download_document(
        self,
        user: AuthenticatedUser,
        document_id: int,
    ):

        return self.service.download(
            user=user,
            document_id=document_id,
        )


    def list_documents(
        self,
        user: AuthenticatedUser,
        department: str | None = None,
    ) -> list[DocumentInfo]:

        return self.service.list(
            user=user,
            department=department,
        )


    def delete_document(
        self,
        user: AuthenticatedUser,
        document_id: int,
    ) -> None:

        self.service.delete(
            user=user,
            document_id=document_id,
        )

    def latest_version(
        self,
        user: AuthenticatedUser,
        filename: str,
    ) -> DocumentInfo:

        return self.service.latest(
            user=user,
            filename=filename,
        )