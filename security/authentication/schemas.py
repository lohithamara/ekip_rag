from dataclasses import dataclass


@dataclass(slots=True)
class User:

    user_id: str

    username: str

    password_hash: str

    role: str

    tenant_id: str

    active: bool = True


@dataclass(slots=True)
class AuthenticatedUser:
    """
    Identity produced after
    successful authentication.

    This object is trusted by
    downstream services.
    """

    user_id: str

    username: str

    role: str

    tenant_id: str

    department: str


@dataclass(slots=True)
class LoginRequest:

    username: str

    password: str


@dataclass(slots=True)
class LoginResponse:

    access_token: str

    user: AuthenticatedUser