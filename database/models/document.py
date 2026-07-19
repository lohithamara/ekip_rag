from sqlalchemy import (
    ForeignKey,
    Integer,
    String,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from database.models.base import (
    Base,
    TimestampMixin,
)


class Document(
    Base,
    TimestampMixin,
):

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(
        primary_key=True,
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    file_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    file_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    s3_key: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    tenant_id: Mapped[int] = mapped_column(
        ForeignKey(
            "tenants.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    department_id: Mapped[int] = mapped_column(
        ForeignKey(
            "departments.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    uploaded_by: Mapped[int] = mapped_column(
        ForeignKey(
            "users.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="PROCESSING",
        nullable=False,
    )

    tenant = relationship(
        "Tenant",
        back_populates="documents",
    )

    department = relationship(
        "Department",
        back_populates="documents",
    )

    uploader = relationship(
        "User",
        back_populates="documents",
    )