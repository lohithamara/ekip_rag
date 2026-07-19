from security.authorization.repository import (
    PermissionRepository,
)

from security.authorization.service import (
    AuthorizationService,
)


def main():

    service = AuthorizationService(

        PermissionRepository()
    )

    finance = (
        service.permissions_for_role(
            "finance_manager"
        )
    )

    assert (
        service.can_access_department(
            finance,
            "finance",
        )
    )

    assert not (
        service.can_access_department(
            finance,
            "hr",
        )
    )

    assert (
        service.can_use_tool(
            finance,
            "analytics",
        )
    )

    assert not (
        service.can_use_tool(
            finance,
            "sql",
        )
    )

    admin = (
        service.permissions_for_role(
            "admin"
        )
    )

    assert (
        service.can_access_department(
            admin,
            "engineering",
        )
    )

    assert (
        service.can_use_tool(
            admin,
            "sql",
        )
    )

    print()
    print("=" * 80)
    print("RBAC TEST")
    print("=" * 80)
    print()

    print("Department access    : PASS")
    print("Tool access          : PASS")
    print("Admin permissions    : PASS")

    print()
    print("=" * 80)
    print("FINAL STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()