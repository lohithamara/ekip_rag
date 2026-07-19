from fastapi import (
    APIRouter,
    Depends,
)

from sqlalchemy.orm import (
    Session,
)

from database.session import (
    get_db,
)

from security.authentication.repository import (
    UserRepository,
)

from security.authentication.jwt_service import (
    JWTService,
)

from security.authentication.schemas import (
    LoginRequest,
    LoginResponse,
)

from security.authentication.service import (
    AuthenticationService,
)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


@router.post(
    "/login",
    response_model=LoginResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):

    service = AuthenticationService(

        repository=UserRepository(db),

        jwt_service=JWTService(),
    )

    return service.login(
        request
    )