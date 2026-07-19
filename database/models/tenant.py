from sqlalchemy import (
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

class Tenant(
    Base,
    TimestampMixin,
):

    __tablename__ = "tenants"

    id: Mapped[int] = mapped_column(
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    departments = relationship(
        "Department",
        back_populates="tenant",
        cascade="all, delete-orphan",
    )

    users = relationship(
        "User",
        back_populates="tenant",
    )

    documents = relationship(
        "Document",
        back_populates="tenant",
    )