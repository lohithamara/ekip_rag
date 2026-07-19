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


class Permission(
    Base,
    TimestampMixin,
):

    __tablename__ = "permissions"

    id: Mapped[int] = mapped_column(
        primary_key=True,
    )

    role_id: Mapped[int] = mapped_column(
        ForeignKey(
            "roles.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    tool: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    role = relationship(
        "Role",
        back_populates="permissions",
    )