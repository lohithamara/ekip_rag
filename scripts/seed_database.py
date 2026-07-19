from sqlalchemy import (
    select,
)

from database.session import (
    SessionLocal,
)

from database.models.tenant import (
    Tenant,
)

from database.models.department import (
    Department,
)

from database.models.role import (
    Role,
)

from database.models.permission import (
    Permission,
)

from database.models.user import (
    User,
)

from security.authentication.password import (
    PasswordService,
)


TENANT_NAME = "EKIP"


DEPARTMENTS = [

    "finance",

    "hr",

    "engineering",

    "legal",

    "sales",

    "support",
]


ROLES = [

    "admin",

    "knowledge_admin",

    "department_manager",

    "employee",

    "auditor",
]


ROLE_PERMISSIONS = {

    "admin": [

        "*",
    ],

    "knowledge_admin": [

        "view",

        "upload",

        "delete",

        "reindex",

        "rag",

        "download",
    ],

    "department_manager": [

        "view",

        "rag",

        "download",

        "analytics",
    ],

    "employee": [

        "view",

        "rag",

        "download",
    ],

    "auditor": [

        "view",

        "rag",

        "download",
    ],
}


def get_or_create_tenant(
    session,
):

    tenant = session.scalar(

        select(Tenant)

        .where(

            Tenant.name == TENANT_NAME
        )
    )

    if tenant:

        return tenant

    tenant = Tenant(

        name="EKIP",
    )

    session.add(
        tenant
    )

    session.commit()

    session.refresh(
        tenant
    )

    return tenant


def create_departments(

    session,

    tenant,
):

    for name in DEPARTMENTS:

        existing = session.scalar(

            select(Department)

            .where(

                Department.name == name,

                Department.tenant_id == tenant.id,
            )
        )

        if existing:

            continue

        session.add(

            Department(

                tenant_id=tenant.id,

                name=name,
            )
        )

    session.commit()


def create_roles(
    session,
):

    for name in ROLES:

        role = session.scalar(

            select(Role)

            .where(
                Role.name == name
            )
        )

        if role:

            continue

        session.add(

            Role(
                name=name
            )
        )

    session.commit()


def create_permissions(
    session,
):

    roles = {

        role.name: role

        for role

        in session.scalars(

            select(Role)
        )
    }

    for role_name, tools in ROLE_PERMISSIONS.items():

        role = roles[
            role_name
        ]

        for tool in tools:

            exists = session.scalar(

                select(Permission)

                .where(

                    Permission.role_id == role.id,

                    Permission.tool == tool,
                )
            )

            if exists:

                continue

            session.add(

                Permission(

                    role_id=role.id,

                    tool=tool,
                )
            )

    session.commit()


def create_users(
    session,
):

    tenant = session.scalar(

        select(Tenant)

        .where(

            Tenant.name == TENANT_NAME
        )
    )

    departments = {

        department.name: department

        for department

        in session.scalars(

            select(Department)
            .where(
                Department.tenant_id == tenant.id
            )
        )
    }

    roles = {

        role.name: role

        for role

        in session.scalars(
            select(Role)
        )
    }

    users = [

        {
            "username": "admin",
            "email": "admin@ekip.ai",
            "password": "admin123",
            "role": "admin",
            "department": "engineering",
        },

        {
            "username": "knowledge_admin",
            "email": "knowledge_admin@ekip.ai",
            "password": "knowledge123",
            "role": "knowledge_admin",
            "department": "engineering",
        },

        {
            "username": "finance_manager",
            "email": "finance_manager@ekip.ai",
            "password": "finance123",
            "role": "department_manager",
            "department": "finance",
        },

        {
            "username": "hr_manager",
            "email": "hr_manager@ekip.ai",
            "password": "hr123",
            "role": "department_manager",
            "department": "hr",
        },

        {
            "username": "engineering_manager",
            "email": "engineering_manager@ekip.ai",
            "password": "engineering123",
            "role": "department_manager",
            "department": "engineering",
        },

        {
            "username": "legal_manager",
            "email": "legal_manager@ekip.ai",
            "password": "legal123",
            "role": "department_manager",
            "department": "legal",
        },

        {
            "username": "finance_employee",
            "email": "finance_employee@ekip.ai",
            "password": "finance123",
            "role": "employee",
            "department": "finance",
        },

        {
            "username": "hr_employee",
            "email": "hr_employee@ekip.ai",
            "password": "hr123",
            "role": "employee",
            "department": "hr",
        },

        {
            "username": "engineering_employee",
            "email": "engineering_employee@ekip.ai",
            "password": "engineering123",
            "role": "employee",
            "department": "engineering",
        },

        {
            "username": "legal_employee",
            "email": "legal_employee@ekip.ai",
            "password": "legal123",
            "role": "employee",
            "department": "legal",
        },

        {
            "username": "auditor",
            "email": "auditor@ekip.ai",
            "password": "audit123",
            "role": "auditor",
            "department": "finance",
        },
    ]

    for user_data in users:

        exists = session.scalar(

            select(User)

            .where(

                User.username
                == user_data["username"]
            )
        )

        if exists:

            continue

        session.add(

            User(

                username=user_data["username"],

                email=user_data["email"],

                password_hash=PasswordService.hash_password(
                    user_data["password"]
                ),

                tenant_id=tenant.id,

                department_id=departments[
                    user_data["department"]
                ].id,

                role_id=roles[
                    user_data["role"]
                ].id,

                is_active=True,
            )
        )

    session.commit()


def main():

    session = SessionLocal()

    try:

        tenant = get_or_create_tenant(
            session
        )

        create_departments(
            session,
            tenant,
        )

        create_roles(
            session
        )

        create_permissions(
            session
        )

        create_users(
            session
        )

        print()

        print("=" * 80)

        print("DATABASE SEEDED")

        print("=" * 80)

    finally:

        session.close()


if __name__ == "__main__":

    main()