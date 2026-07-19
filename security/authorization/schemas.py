from dataclasses import dataclass


@dataclass(slots=True)
class DepartmentScope:
    """
    Departments that a request
    is allowed to search.
    """

    tenant_id: str

    departments: tuple[
        str,
        ...
    ]


@dataclass(slots=True)
class PermissionSet:

    role: str

    tools: tuple[str, ...]