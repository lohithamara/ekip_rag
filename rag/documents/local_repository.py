import hashlib
from pathlib import Path

from rag.documents.repository import (
    DocumentRepository,
)

from rag.documents.schemas import (
    Document,
)


class LocalDocumentRepository(
    DocumentRepository,
):

    def __init__(
        self,
        root: str,
    ):

        self.root = Path(root)

    def get_by_filename(
        self,
        tenant_id: str,
        filename: str,
    ) -> Document | None:

        tenant = (
            self.root
            / tenant_id
        )

        for path in tenant.rglob("*"):

            if (
                path.name
                == filename
            ):

                return Document(

                    document_id=str(path),

                    tenant_id=tenant_id,

                    department=(
                        path.parent.name
                    ),

                    filename=path.name,

                    file_type=(
                        path.suffix
                    ),

                    storage_path=str(
                        path
                    ),

                    version=1,

                    metadata={},
                )

        return None

    def latest_version(
        self,
        tenant_id: str,
        filename: str,
    ):

        return self.get_by_filename(
            tenant_id,
            filename,
        )

    def list_documents(
        self,
        tenant_id: str,
        department: str | None = None,
    ):

        tenant = (
            self.root
            / tenant_id
        )

        documents = []

        for path in tenant.rglob("*"):

            if not path.is_file():
                continue

            if (
                department
                and path.parent.name
                != department
            ):
                continue

            documents.append(

                Document(
                    document_id = hashlib.sha256(
                        str(path.relative_to(self.root)).encode()
                    ).hexdigest(),

                    tenant_id=tenant_id,

                    department=(
                        path.parent.name
                    ),

                    filename=path.name,

                    file_type=(
                        path.suffix
                    ),

                    storage_path=str(
                        path
                    ),

                    version=1,

                    metadata={},
                )
            )

        return documents