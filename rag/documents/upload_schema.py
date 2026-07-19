from dataclasses import dataclass
from pathlib import Path

from security.authentication.schemas import (
    AuthenticatedUser,
)


@dataclass(slots=True)
class UploadDocumentRequest:

    user: AuthenticatedUser

    file_path: Path

    original_filename: str

    department: str


@dataclass(slots=True)
class UploadDocumentResponse:

    document_id: int

    filename: str

    department: str

    status: str