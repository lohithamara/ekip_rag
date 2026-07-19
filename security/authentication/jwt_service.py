from datetime import (
    datetime,
    timedelta,
    UTC,
)

from jose import (
    jwt,
)

from security.authentication.config import (
    JWTConfig,
)


class JWTService:

    def create_token(
        self,
        user_id: str|int,
    ) -> str:

        expire = (
            datetime.now(
                UTC
            )
            + timedelta(
                minutes=JWTConfig.expire_minutes
            )
        )

        payload = {

            "sub": str(user_id),

            "exp": expire,
        }

        return jwt.encode(

            payload,

            JWTConfig.secret_key,

            algorithm=JWTConfig.algorithm,
        )

    def verify_token(
        self,
        token: str,
    ) -> dict:

        return jwt.decode(

            token,

            JWTConfig.secret_key,

            algorithms=[
                JWTConfig.algorithm
            ],
        )