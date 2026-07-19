from sqlalchemy.orm import (
    Session,
)

from database.session import (
    SessionLocal,
)

from security.authentication.repository import (
    UserRepository,
)

from security.authentication.schemas import (
    AuthenticatedUser,
)


class TestAuthentication:

    @staticmethod
    def get_user(
        username: str,
    ) -> AuthenticatedUser:

        db: Session = SessionLocal()

        try:

            repository = UserRepository(
                db
            )

            user = (
                repository.get_by_username(
                    username
                )
            )

            if user is None:

                raise ValueError(
                    f"User '{username}' not found."
                )
            
            return AuthenticatedUser(

                user_id=str(user.id),

                username=user.username,

                role=user.role.name,

                tenant_id=str(user.tenant_id),

                department=user.department.name,
            )

        finally:

            db.close()

    @staticmethod
    def admin():

        return (
            TestAuthentication.get_user(
                "admin"
            )
        )

    @staticmethod
    def finance_manager():

        return (
            TestAuthentication.get_user(
                "finance_manager"
            )
        )

    @staticmethod
    def hr_manager():

        return (
            TestAuthentication.get_user(
                "hr_manager"
            )
        )

    @staticmethod
    def engineering_manager():

        return (
            TestAuthentication.get_user(
                "engineering_manager"
            )
        )

    @staticmethod
    def legal_manager():

        return (
            TestAuthentication.get_user(
                "legal_manager"
            )
        )

    @staticmethod
    def finance_employee():

        return (
            TestAuthentication.get_user(
                "finance_employee"
            )
        )

    @staticmethod
    def hr_employee():

        return (
            TestAuthentication.get_user(
                "hr_employee"
            )
        )

    @staticmethod
    def engineering_employee():

        return (
            TestAuthentication.get_user(
                "engineering_employee"
            )
        )

    @staticmethod
    def legal_employee():

        return (
            TestAuthentication.get_user(
                "legal_employee"
            )
        )
    
    @staticmethod
    def manager_for_department(
        department: str,
    ):

        mapping = {

            "finance":
                TestAuthentication.finance_manager,

            "hr":
                TestAuthentication.hr_manager,

            "engineering":
                TestAuthentication.engineering_manager,

            "legal":
                TestAuthentication.legal_manager,
        }

        if department not in mapping:

    
            return TestAuthentication.admin()

        return mapping[department]()