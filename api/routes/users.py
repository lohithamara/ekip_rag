from dataclasses import dataclass

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from sqlalchemy import select

from sqlalchemy.orm import Session

from database.session import get_db

from database.models.user import User

from database.models.role import Role

from database.repositories.user_repository import (
    UserRepository,
)

from database.repositories.department_repository import (
    DepartmentRepository,
)

from security.authentication.dependencies import (
    get_current_user,
)

from security.authentication.schemas import (
    AuthenticatedUser,
)

from security.authentication.password import (
    PasswordService,
)


router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


@dataclass
class CreateUserRequest:

    username: str

    email: str

    password: str

    role: str

    department: str


@dataclass
class UpdateUserRequest:

    role: str | None = None

    department: str | None = None

    is_active: bool | None = None


@dataclass
class UserResponse:

    id: int

    username: str

    email: str

    role: str

    department: str

    is_active: bool


def require_admin(
    current_user: AuthenticatedUser,
):

    if current_user.role != "admin":

        raise HTTPException(
            status_code=403,
            detail="Only administrators can manage users.",
        )


def to_response(
    user: User,
) -> UserResponse:

    return UserResponse(

        id=user.id,

        username=user.username,

        email=user.email,

        role=user.role.name,

        department=user.department.name,

        is_active=user.is_active,
    )


@router.get(
    "",
    response_model=list[UserResponse],
)
def list_users(
    current_user: AuthenticatedUser = Depends(
        get_current_user
    ),
    db: Session = Depends(
        get_db
    ),
):

    require_admin(
        current_user
    )

    repository = UserRepository(
        db
    )

    users = repository.list_by_tenant(
        int(current_user.tenant_id)
    )

    return [
        to_response(user)
        for user in users
    ]


@router.post(
    "",
    response_model=UserResponse,
    status_code=201,
)
def create_user(
    request: CreateUserRequest,
    current_user: AuthenticatedUser = Depends(
        get_current_user
    ),
    db: Session = Depends(
        get_db
    ),
):

    require_admin(
        current_user
    )

    user_repository = UserRepository(
        db
    )

    department_repository = DepartmentRepository(
        db
    )

    tenant_id = int(
        current_user.tenant_id
    )

    if user_repository.get_by_username(
        request.username
    ):

        raise HTTPException(
            status_code=409,
            detail="Username already exists.",
        )

    if user_repository.get_by_email(
        request.email
    ):

        raise HTTPException(
            status_code=409,
            detail="Email already exists.",
        )

    department = department_repository.get_by_name(
        tenant_id=tenant_id,
        name=request.department,
    )

    if department is None:

        raise HTTPException(
            status_code=404,
            detail="Department not found.",
        )

    role = db.scalar(
        select(Role)
        .where(
            Role.name == request.role
        )
    )

    if role is None:

        raise HTTPException(
            status_code=404,
            detail="Role not found.",
        )

    user = User(

        username=request.username,

        email=request.email,

        password_hash=PasswordService.hash_password(
            request.password
        ),

        tenant_id=tenant_id,

        department_id=department.id,

        role_id=role.id,

        is_active=True,
    )

    user = user_repository.create(
        user
    )

    return to_response(
        user
    )


@router.patch(
    "/{user_id}",
    response_model=UserResponse,
)
def update_user(
    user_id: int,
    request: UpdateUserRequest,
    current_user: AuthenticatedUser = Depends(
        get_current_user
    ),
    db: Session = Depends(
        get_db
    ),
):

    require_admin(
        current_user
    )

    user_repository = UserRepository(
        db
    )

    department_repository = DepartmentRepository(
        db
    )

    user = user_repository.get_by_id(
        user_id
    )

    if (
        user is None
        or user.tenant_id
        != int(current_user.tenant_id)
    ):

        raise HTTPException(
            status_code=404,
            detail="User not found.",
        )

    if request.department is not None:

        department = department_repository.get_by_name(

            tenant_id=int(
                current_user.tenant_id
            ),

            name=request.department,
        )

        if department is None:

            raise HTTPException(
                status_code=404,
                detail="Department not found.",
            )

        user.department_id = department.id

    if request.role is not None:

        role = db.scalar(
            select(Role)
            .where(
                Role.name == request.role
            )
        )

        if role is None:

            raise HTTPException(
                status_code=404,
                detail="Role not found.",
            )

        user.role_id = role.id

    if request.is_active is not None:

        user.is_active = request.is_active

    user = user_repository.update(
        user
    )

    return to_response(
        user
    )