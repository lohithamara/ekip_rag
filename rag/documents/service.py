from pathlib import Path

from database.repositories.user_repository import ( 
UserRepository,
)

from ingestion.storage.s3_client import (
    S3Client,
)

from rag.documents.exceptions import (
    DepartmentAccessDeniedError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    DepartmentNotFoundError
)

from rag.documents.repository import (
    DocumentRepository,
)

from rag.documents.schemas import (
    DocumentRecord,
    DocumentInfo
)

from rag.documents.upload_schema import (
    UploadDocumentRequest,
    UploadDocumentResponse,
)

from rag.vector_store.qdrant_store import (
    QdrantVectorStore,
)

from database.repositories.department_repository import (
    DepartmentRepository,
)

import hashlib
import tempfile


from ingestion.workers.document_ingestion_worker import (
    DocumentIngestionWorker,
)

class DocumentService:

    def __init__(
        self,
        repository: DocumentRepository,
        department_repository: DepartmentRepository,
        user_repository: UserRepository,
        s3_client: S3Client,
        worker: DocumentIngestionWorker,
        vector_store: QdrantVectorStore,
    ):
        self.repository = repository
        self.department_repository = department_repository
        self.user_repository = user_repository
        self.s3_client = s3_client
        self.worker = worker
        self.vector_store = vector_store

    def _check_document_access(
        self,
        user,
        document: DocumentRecord,
    ) -> None:

        # Never allow cross-tenant access
        if document.tenant_id != int(user.tenant_id):
            raise DepartmentAccessDeniedError(
                "Document access denied"
            )

        # Admin and knowledge admin can access
        # all departments within their tenant
        if user.role in (
            "admin",
            "knowledge_admin",
        ):
            return

        # Other roles can access only
        # their own department
        department = self.department_repository.get_by_id(
            document.department_id
        )

        if (
            department is None
            or department.name != user.department
        ):
            raise DepartmentAccessDeniedError(
                "Document access denied"
            )

    def get_document(
        self,
        user,
        document_id: int,
    ) -> DocumentInfo:

        document = self.repository.get_by_id(
            document_id
        )

        if document is None:
            raise DocumentNotFoundError(
                document_id
            )

        self._check_document_access(
            user,
            document,
        )

        return self._to_document_info(
            document
        )

    def upload_document(
        self,
        request: UploadDocumentRequest,
    ) -> UploadDocumentResponse:

        if request.user.role in (
            "admin",
            "knowledge_admin",
        ):
            target_department = request.department

        else:

            if request.department != request.user.department:
                raise DepartmentAccessDeniedError(
                    request.department
                )

            target_department = request.user.department


        content_hash = self._compute_hash(
            request.file_path
        )

        self._check_duplicate(

            tenant_id=int(
                request.user.tenant_id
            ),

            content_hash=content_hash,

            filename=request.original_filename,
        )

        s3_key = self._build_s3_key(

            tenant_id=int(
                request.user.tenant_id
            ),

            department=target_department,

            content_hash=content_hash,

            filename=request.original_filename,
        )

        document = self._create_document_record(

            request=request,
            target_department=target_department,
            content_hash=content_hash,
            s3_key=s3_key,
        )

        try:

            self._upload_to_s3(

                file_path=request.file_path,

                s3_key=s3_key,
            )

            self.repository.update_status(

                document.document_id,

                "PROCESSING",
            )

        except Exception:

            self.repository.update_status(

                document.document_id,

                "FAILED",
            )

            raise

        return UploadDocumentResponse(

            document_id=document.document_id,

            filename=document.filename,

            department=target_department,

            status="PROCESSING",
        )

    def list(
        self,
        user,
        department: str | None = None,
    ) -> list[DocumentInfo]:

        tenant_id = int(
            user.tenant_id
        )

        department_id = None

        if user.role in (
            "admin",
            "knowledge_admin",
        ):

            if department is not None:

                dept = self.department_repository.get_by_name(
                    tenant_id=tenant_id,
                    name=department,
                )

                if dept is None:
                    return []

                department_id = dept.id

        else:

            # All other roles are restricted
            # to their own department
            dept = self.department_repository.get_by_name(
                tenant_id=tenant_id,
                name=user.department,
            )

            if dept is None:
                return []

            department_id = dept.id

        documents = self.repository.list_documents(
            tenant_id=tenant_id,
            department_id=department_id,
        )

        return [
            self._to_document_info(document)
            for document in documents
        ]

    def latest(
        self,
        tenant_id: int,
        filename: str,
    ) -> DocumentInfo:

        document = self.repository.latest_version(

            tenant_id=tenant_id,

            filename=filename,
        )

        if document is None:

            raise DocumentNotFoundError(
                filename
            )

        return self._to_document_info(
            document
        )

    def download(
        self,
        user,
        document_id: int,
    ) -> Path:

        document = self.repository.get_by_id(
            document_id
        )

        if document is None:
            raise DocumentNotFoundError(
                document_id
            )

        self._check_document_access(
            user,
            document,
        )

        temp_dir = Path(
            tempfile.gettempdir()
        )

        destination = (
            temp_dir /
            document.filename
        )

        self.s3_client.download_file(
            s3_key=document.s3_key,
            local_path=str(destination),
        )

        return destination

    def _compute_hash(
        self,
        file_path: Path,
    ) -> str:

        sha256 = hashlib.sha256()

        with open(
            file_path,
            "rb",
        ) as file:

            while True:

                block = file.read(
                    8192
                )

                if not block:
                    break

                sha256.update(
                    block
                )

        return sha256.hexdigest()
    
    def delete(
        self,
        user,
        document_id: int,
    ) -> None:

        if user.role not in (
            "admin",
            "knowledge_admin",
        ):
            raise DepartmentAccessDeniedError(
                "You do not have permission to delete documents"
            )

        document = self.repository.get_by_id(
            document_id
        )

        if document is None:
            raise DocumentNotFoundError(
                document_id
            )

        self._check_document_access(
            user,
            document,
        )

        # Delete document chunks/vectors from Qdrant
        self.vector_store.delete_document_points(
            document_id=document.document_id,
            tenant_id=document.tenant_id,
        )

        # Delete original file from S3
        if document.s3_key:
            self.s3_client.delete_file(
                document.s3_key
            )

        # Delete document record from PostgreSQL
        self.repository.delete(
            document.document_id
        )

    def _build_s3_key(
        self,
        tenant_id: int,
        department: str,
        content_hash: str,
        filename: str,
    ) -> str:

        return (
            f"original/"
            f"{tenant_id}/"
            f"{department}/"
            f"{content_hash}/"
            f"{filename}"
        )
    
    def _upload_to_s3(
        self,
        file_path: Path,
        s3_key: str,
    ):

        self.s3_client.upload_file(

            local_path=file_path,

            s3_key=s3_key,
        )

    def _check_duplicate(
        self,
        tenant_id: int,
        content_hash: str,
        filename: str,
    ) -> None:

        if self.repository.exists(

            tenant_id=tenant_id,

            content_hash=content_hash,
        ):

            raise DuplicateDocumentError(
                filename
            )
    
    def _to_document_info(
        self,
        document: DocumentRecord,
    ) -> DocumentInfo:

        department = (
            self.department_repository.get_by_id(
                document.department_id
            )
        )

        user = None

        if document.uploaded_by is not None:

            user = self.user_repository.get_by_id(
                document.uploaded_by
            )

        return DocumentInfo(

            document_id=document.document_id,

            filename=document.filename,

            file_type=document.file_type,

            file_size=document.file_size,

            department=(
                department.name
                if department
                else ""
            ),

            uploaded_by=(
                user.username
                if user
                else ""
            ),

            version=document.version,

            status=document.status,

            created_at=document.created_at,
        )

    def _create_document_record(
        self,
        request: UploadDocumentRequest,
        target_department: str,
        content_hash: str,
        s3_key: str,
    ) -> DocumentRecord:

        department = (
            self.department_repository.get_by_name(
                tenant_id=int(
                    request.user.tenant_id
                ),
                name=target_department,
            )
        )

        if department is None:

            raise DepartmentNotFoundError(
                target_department
            )

        document = DocumentRecord(

            document_id=0,

            tenant_id=int(
                request.user.tenant_id
            ),

            department_id=department.id,

            uploaded_by=int(
                request.user.user_id
            ),

            filename=request.original_filename,

            file_type=request.file_path.suffix.replace(
                ".",
                "",
            ),

            content_hash=content_hash,

            file_size=request.file_path.stat().st_size,

            s3_key=s3_key,

            version=1,

            status="PROCESSING",

            created_at=None,

            updated_at=None,
        )

        return self.repository.create(
            document
        )