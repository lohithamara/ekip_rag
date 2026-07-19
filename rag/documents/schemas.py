from dataclasses import dataclass
from datetime import datetime


# --------------------------------------------------
# Database Record
# --------------------------------------------------

@dataclass(slots=True)
class DocumentRecord:

    document_id: int

    tenant_id: int

    department_id: int

    uploaded_by: int | None

    filename: str

    file_type: str

    content_hash: str

    file_size: int

    s3_key: str

    version: int

    status: str

    created_at: datetime | None

    updated_at: datetime | None


# --------------------------------------------------
# API / Service DTO
# --------------------------------------------------

@dataclass(slots=True)
class DocumentInfo:

    document_id: int

    filename: str

    file_type: str

    file_size: int

    department: str

    uploaded_by: str

    version: int

    status: str

    created_at: datetime | None