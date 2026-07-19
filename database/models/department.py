from sqlalchemy import (
    ForeignKey,
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


class Department(
    Base,
    TimestampMixin,
):

    __tablename__ = "departments"

    id: Mapped[int] = mapped_column(
        primary_key=True,
    )

    tenant_id: Mapped[int] = mapped_column(
        ForeignKey(
            "tenants.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    tenant = relationship(
        "Tenant",
        back_populates="departments",
    )

    users = relationship(
        "User",
        back_populates="department",
    )

    documents = relationship(
        "Document",
        back_populates="department",
    )