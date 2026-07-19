from dataclasses import dataclass

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from sqlalchemy.orm import Session

from database.session import get_db

from database.models.department import Department

from database.repositories.department_repository import (
    DepartmentRepository,
)

from security.authentication.dependencies import (
    get_current_user,
)

from security.authentication.schemas import (
    AuthenticatedUser,
)


router = APIRouter(
    prefix="/departments",
    tags=["Departments"],
)


@dataclass
class CreateDepartmentRequest:

    name: str


@dataclass
class DepartmentResponse:

    id: int

    name: str


@router.get(
    "",
    response_model=list[DepartmentResponse],
)
def list_departments(
    current_user: AuthenticatedUser = Depends(
        get_current_user
    ),
    db: Session = Depends(
        get_db
    ),
):

    repository = DepartmentRepository(
        db
    )

    tenant_id = int(
        current_user.tenant_id
    )

    # Admin and knowledge admin need all
    # departments for management/upload UI.
    if current_user.role in (
        "admin",
        "knowledge_admin",
    ):

        departments = repository.list_by_tenant(
            tenant_id
        )

    else:

        department = repository.get_by_name(
            tenant_id=tenant_id,
            name=current_user.department,
        )

        departments = (
            [department]
            if department
            else []
        )

    return [
        DepartmentResponse(
            id=department.id,
            name=department.name,
        )
        for department in departments
    ]


@router.post(
    "",
    response_model=DepartmentResponse,
    status_code=201,
)
def create_department(
    request: CreateDepartmentRequest,
    current_user: AuthenticatedUser = Depends(
        get_current_user
    ),
    db: Session = Depends(
        get_db
    ),
):

    if current_user.role != "admin":

        raise HTTPException(
            status_code=403,
            detail="Only administrators can create departments.",
        )

    repository = DepartmentRepository(
        db
    )

    tenant_id = int(
        current_user.tenant_id
    )

    existing = repository.get_by_name(
        tenant_id=tenant_id,
        name=request.name,
    )

    if existing:

        raise HTTPException(
            status_code=409,
            detail="Department already exists.",
        )

    department = Department(
        tenant_id=tenant_id,
        name=request.name,
    )

    department = repository.create(
        department
    )

    return DepartmentResponse(
        id=department.id,
        name=department.name,
    )


@router.delete(
    "/{department_id}",
    status_code=204,
)
def delete_department(
    department_id: int,
    current_user: AuthenticatedUser = Depends(
        get_current_user
    ),
    db: Session = Depends(
        get_db
    ),
):

    if current_user.role != "admin":

        raise HTTPException(
            status_code=403,
            detail="Only administrators can delete departments.",
        )

    repository = DepartmentRepository(
        db
    )

    department = repository.get_by_id(
        department_id
    )

    if department is None:

        raise HTTPException(
            status_code=404,
            detail="Department not found.",
        )

    # Prevent cross-tenant deletion
    if department.tenant_id != int(
        current_user.tenant_id
    ):

        raise HTTPException(
            status_code=404,
            detail="Department not found.",
        )

    repository.delete(
        department
    )