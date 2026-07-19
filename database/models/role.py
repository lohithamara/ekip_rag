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


class Role(
    Base,
    TimestampMixin,
):

    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    permissions = relationship(
        "Permission",
        back_populates="role",
        cascade="all, delete-orphan",
    )

    users = relationship(
        "User",
        back_populates="role",
    )