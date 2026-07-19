from sqlalchemy import (
    select,
)

from sqlalchemy.orm import (
    Session,
)

from database.models.document import (
    Document,
)

from rag.documents.repository import (
    DocumentRepository,
)

from rag.documents.schemas import (
    DocumentRecord,
)


class DatabaseDocumentRepository(
    DocumentRepository,
):

    def __init__(
        self,
        session: Session,
    ):
        self.session = session

    def create(
        self,
        document: DocumentRecord,
    ) -> DocumentRecord:

        db_document = Document(

            filename=document.filename,

            file_type=document.file_type,

            content_hash=document.content_hash,

            file_size=document.file_size,

            s3_key=document.s3_key,

            tenant_id=document.tenant_id,

            department_id=document.department_id,

            uploaded_by=document.uploaded_by,

            version=document.version,

            status=document.status,
        )

        self.session.add(
            db_document
        )

        self.session.commit()

        self.session.refresh(
            db_document
        )

        return self._to_record(
            db_document
        )

    def get_by_id(
        self,
        document_id: int,
    ) -> DocumentRecord | None:

        document = self.session.get(
            Document,
            document_id,
        )

        if document is None:
            return None

        return self._to_record(
            document
        )

    def get_by_filename(
        self,
        tenant_id: int,
        filename: str,
    ) -> DocumentRecord | None:

        document = self.session.scalar(

            select(Document)

            .where(
                Document.tenant_id == tenant_id,
                Document.filename == filename,
            )
        )

        if document is None:
            return None

        return self._to_record(
            document
        )

    def latest_version(
        self,
        tenant_id: int,
        filename: str,
    ) -> DocumentRecord | None:

        document = self.session.scalar(

            select(Document)

            .where(
                Document.tenant_id == tenant_id,
                Document.filename == filename,
            )

            .order_by(
                Document.version.desc()
            )
        )

        if document is None:
            return None

        return self._to_record(
            document
        )

    def list_documents(
        self,
        tenant_id: int,
        department_id: int | None = None,
    ) -> list[DocumentRecord]:

        query = (

            select(Document)

            .where(
                Document.tenant_id == tenant_id
            )
        )

        if department_id is not None:

            query = query.where(

                Document.department_id
                == department_id
            )

        documents = self.session.scalars(
            query
        ).all()

        return [

            self._to_record(doc)

            for doc in documents
        ]

    def exists(
        self,
        tenant_id: int,
        content_hash: str,
    ) -> bool:

        document = self.session.scalar(

            select(Document)

            .where(
                Document.tenant_id == tenant_id,
                Document.content_hash == content_hash,
            )
        )

        return document is not None

    def update(
        self,
        document: DocumentRecord,
    ) -> DocumentRecord:

        db_document = self.session.get(
            Document,
            document.document_id,
        )

        if db_document is None:

            raise ValueError(
                "Document not found."
            )

        db_document.filename = document.filename
        db_document.file_type = document.file_type
        db_document.content_hash = document.content_hash
        db_document.file_size = document.file_size
        db_document.s3_key = document.s3_key
        db_document.department_id = document.department_id
        db_document.uploaded_by = document.uploaded_by
        db_document.version = document.version
        db_document.status = document.status

        self.session.commit()

        self.session.refresh(
            db_document
        )

        return self._to_record(
            db_document
        )

    def update_status(
        self,
        document_id: int,
        status: str,
    ) -> None:

        document = self.session.get(
            Document,
            document_id,
        )

        if document is None:

            raise ValueError(
                "Document not found."
            )

        document.status = status

        self.session.commit()

    def delete(
        self,
        document_id: int,
    ) -> None:

        document = self.session.get(
            Document,
            document_id,
        )

        if document is None:
            return

        self.session.delete(
            document
        )

        self.session.commit()

    def _to_record(
        self,
        document: Document,
    ) -> DocumentRecord:

        return DocumentRecord(

            document_id=document.id,

            tenant_id=document.tenant_id,

            department_id=document.department_id,

            uploaded_by=document.uploaded_by,

            filename=document.filename,

            file_type=document.file_type,

            content_hash=document.content_hash,

            file_size=document.file_size,

            s3_key=document.s3_key,

            version=document.version,

            status=document.status,

            created_at=document.created_at,

            updated_at=document.updated_at,
        )