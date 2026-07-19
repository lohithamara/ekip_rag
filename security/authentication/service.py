from security.authentication.jwt_service import (
    JWTService,
)

from security.authentication.password import (
    PasswordService,
)

from security.authentication.repository import (
    UserRepository,
)

from security.authentication.schemas import (
    LoginRequest,
    LoginResponse,
    AuthenticatedUser,
)


class AuthenticationService:

    def __init__(
        self,
        repository: UserRepository,
        jwt_service: JWTService,
    ):

        self.repository = repository

        self.jwt_service = jwt_service

    def login(
        self,
        request: LoginRequest,
    ) -> LoginResponse:

        user = (
            self.repository
            .get_by_username(
                request.username
            )
        )

        if user is None:

            raise ValueError(
                "Unknown user."
            )

        if not (
            PasswordService.verify_password(
                request.password,
                user.password_hash,
            )
        ):

            raise ValueError(
                "Invalid password."
            )

        token = (
            self.jwt_service
            .create_token(
                user.id
            )
        )

        return LoginResponse(

            access_token=token,

            user=AuthenticatedUser(

                user_id=str(user.id),

                username=user.username,

                role=user.role.name,

                tenant_id=str(user.tenant_id),

                department=user.department.name,
            ),
        )