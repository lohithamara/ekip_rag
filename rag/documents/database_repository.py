from sqlalchemy import (
    desc,
    select,
)

from sqlalchemy.orm import (
    Session,
)

from database.models.document import (
    Document as DocumentModel,
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

    @staticmethod
    def _to_record(
        model: DocumentModel,
    ) -> DocumentRecord:

        return DocumentRecord(

            document_id=model.id,

            tenant_id=model.tenant_id,

            department_id=model.department_id,

            uploaded_by=model.uploaded_by,

            filename=model.filename,

            file_type=model.file_type,

            content_hash=model.content_hash,

            file_size=model.file_size,

            s3_key=model.s3_key,

            version=model.version,

            status=model.status,

            created_at=model.created_at,

            updated_at=model.updated_at,
        )

    def create(
        self,
        document: DocumentRecord,
    ) -> DocumentRecord:

        model = DocumentModel(

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
        )

        try:

            self.session.add(
                model
            )

            self.session.commit()

            self.session.refresh(
                model
            )

            return self._to_record(
                model
            )

        except:

            self.session.rollback()

            raise

    def get_by_id(
        self,
        document_id: int,
    ) -> DocumentRecord | None:

        model = self.session.get(
            DocumentModel,
            document_id,
        )

        if model is None:

            return None

        return self._to_record(
            model
        )

    def get_by_filename(
        self,
        tenant_id: int,
        filename: str,
    ) -> DocumentRecord | None:

        model = self.session.scalar(

            select(DocumentModel)

            .where(

                DocumentModel.tenant_id == tenant_id,

                DocumentModel.filename == filename,
            )
        )

        if model is None:

            return None

        return self._to_record(
            model
        )

    def latest_version(
        self,
        tenant_id: int,
        filename: str,
    ) -> DocumentRecord | None:

        model = self.session.scalar(

            select(DocumentModel)

            .where(

                DocumentModel.tenant_id == tenant_id,

                DocumentModel.filename == filename,
            )

            .order_by(

                desc(
                    DocumentModel.version
                )
            )
        )

        if model is None:

            return None

        return self._to_record(
            model
        )

    def list_documents(
        self,
        tenant_id: int,
        department_id: int | None = None,
    ) -> list[DocumentRecord]:

        query = (

            select(
                DocumentModel
            )

            .where(

                DocumentModel.tenant_id == tenant_id
            )
        )

        if department_id is not None:

            query = query.where(

                DocumentModel.department_id
                == department_id
            )

        models = self.session.scalars(
            query
        ).all()

        return [

            self._to_record(
                model
            )

            for model

            in models
        ]

    def exists(
        self,
        tenant_id: int,
        content_hash: str,
    ) -> bool:

        return (

            self.session.scalar(

                select(DocumentModel)

                .where(

                    DocumentModel.tenant_id == tenant_id,

                    DocumentModel.content_hash
                    == content_hash,
                )
            )

            is not None
        )

    def update_status(
        self,
        document_id: int,
        status: str,
    ) -> None:

        model = self.session.get(

            DocumentModel,

            document_id,
        )

        if model is None:

            raise ValueError(
                "Document not found."
            )

        try:

            model.status = status

            self.session.commit()

        except:

            self.session.rollback()

            raise

    def delete(
        self,
        document_id: int,
    ) -> None:

        model = self.session.get(

            DocumentModel,

            document_id,
        )

        if model is None:

            return

        try:

            self.session.delete(
                model
            )

            self.session.commit()

        except:

            self.session.rollback()

            raise