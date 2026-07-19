from security.authorization.repository import (
    PermissionRepository,
)

from security.authorization.schemas import (
    DepartmentScope,
    PermissionSet,
)


class AuthorizationService:

    def __init__(
        self,
        repository: PermissionRepository,
    ):
        self.repository = repository

    def permissions_for_role(
        self,
        role: str,
    ) -> PermissionSet:

        return (
            self.repository.get_permissions(
                role
            )
        )

    def filter_departments(
        self,
        tenant_id: str,
        requested_departments: tuple[str, ...],
        user_department: str,
        user_role: str,
    ) -> DepartmentScope:

        # Admin / Knowledge Admin can access any requested department
        if user_role in ("admin", "knowledge_admin"):

            return DepartmentScope(
                tenant_id=tenant_id,
                departments=("*",),
            )

        # Normal users can access only their own department
        if not requested_departments or "*" in requested_departments:

            return DepartmentScope(
                tenant_id=tenant_id,
                departments=(user_department,),
            )

        if user_department in requested_departments:

            return DepartmentScope(
                tenant_id=tenant_id,
                departments=(user_department,),
            )

        return DepartmentScope(
            tenant_id=tenant_id,
            departments=(),
        )

    def can_use_tool(
        self,
        permissions: PermissionSet,
        tool: str,
    ) -> bool:

        return (

            "*"
            in permissions.tools

            or

            tool
            in permissions.tools
        )