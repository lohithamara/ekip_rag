from security.authentication.schemas import (
    AuthenticatedUser,
)
from tests.test_auth import TestAuthentication


def create_test_user():

    return TestAuthentication.finance_manager()