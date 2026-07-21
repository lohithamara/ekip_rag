from fastapi import (
    Depends,
    HTTPException,
    status,
)



from jose import (
    JWTError,
)

from sqlalchemy.orm import (
    Session,
)

from database.session import (
    get_db,
)

from security.authentication.jwt_service import (
    JWTService,
)

from security.authentication.repository import (
    UserRepository,
)

from security.authentication.schemas import (
    AuthenticatedUser,
)

from fastapi.security import HTTPBearer
from fastapi.security import HTTPAuthorizationCredentials

security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> AuthenticatedUser:
    token = credentials.credentials
    jwt_service = JWTService()

    try:
        payload = jwt_service.verify_token(token)
        # print("JWT payload:", payload)

    except JWTError as exc:
        # print("JWT ERROR:", repr(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )

    user_id = payload.get("sub")

    if user_id is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )

    repository = UserRepository(db)

    user = repository.get_by_id(int(user_id))

    if user is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        )

    return AuthenticatedUser(
        user_id=str(user.id),
        username=user.username,
        role=user.role.name,
        tenant_id=str(user.tenant_id),
        department=user.department.name,
    )