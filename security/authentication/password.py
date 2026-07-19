import hashlib


class PasswordService:

    @staticmethod
    def hash_password(
        password: str,
    ) -> str:

        return hashlib.sha256(
            password.encode()
        ).hexdigest()

    @staticmethod
    def verify_password(
        password: str,
        password_hash: str,
    ) -> bool:

        return (
            PasswordService.hash_password(
                password
            )
            == password_hash
        )