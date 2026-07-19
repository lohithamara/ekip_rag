from sqlalchemy import select

from sqlalchemy.orm import Session

from database.models.role import Role

from database.models.permission import Permission

from security.authorization.schemas import PermissionSet


class PermissionRepository:

    def __init__(
        self,
        session: Session,
    ):
        self.session = session

    def get_permissions(
        self,
        role: str,
    ) -> PermissionSet:

        role_obj = self.session.scalar(

            select(Role)

            .where(
                Role.name == role
            )
        )

        if role_obj is None:

            raise ValueError(
                f"Unknown role: {role}"
            )

        permissions = self.session.scalars(

            select(Permission)

            .where(
                Permission.role_id == role_obj.id
            )

        ).all()

        return PermissionSet(

            role=role,

            tools=tuple(

                permission.tool

                for permission

                in permissions
            ),
        )