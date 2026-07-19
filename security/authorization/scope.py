from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class DepartmentScope:
    """
    Departments the current user
    is allowed to access within
    a tenant.
    """

    tenant_id: str

    departments: tuple[
        str,
        ...
    ]