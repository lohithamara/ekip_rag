from security.authorization.repository import (
    PermissionRepository,
)

from security.authorization.service import (
    AuthorizationService,
)
from database.session import SessionLocal

def main():

    session = SessionLocal()

    service = AuthorizationService(
        PermissionRepository(session)
    )
    permissions = (
        service.permissions_for_role(
            "department_manager"
        )
    )

    scope = service.filter_departments(

        tenant_id="tenant_1",

        requested_departments=(

            "finance",

            "engineering",
        ),

        user_department="finance",
        user_role="department_manager",
    )

    assert scope.departments == (
        "finance",
    )

    print()
    print("=" * 80)
    print("AUTHORIZATION INTEGRATION")
    print("=" * 80)
    print()

    print(
        "Department filtering : PASS"
    )

    print()

    print("=" * 80)
    print("FINAL STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()