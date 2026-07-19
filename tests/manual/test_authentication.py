from database.session import (
    SessionLocal,
)

from security.authentication.jwt_service import (
    JWTService,
)

from security.authentication.repository import (
    UserRepository,
)

from security.authentication.schemas import (
    LoginRequest,
)

from security.authentication.service import (
    AuthenticationService,
)


def main():

    db = SessionLocal()

    try:

        service = AuthenticationService(

            repository=UserRepository(db),

            jwt_service=JWTService(),
        )

        response = service.login(

            LoginRequest(

                username="admin",

                password="admin123",
            )
        )

        assert response.user.username == "admin"

        assert response.user.role == "admin"

        assert response.access_token

        print()
        print("=" * 80)
        print("AUTHENTICATION TEST")
        print("=" * 80)
        print()

        print("Database lookup      : PASS")
        print("Password verify      : PASS")
        print("JWT generation       : PASS")

        print()
        print("=" * 80)
        print("FINAL STATUS : PASS")
        print("=" * 80)

    finally:

        db.close()


if __name__ == "__main__":

    main()