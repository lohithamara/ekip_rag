from sqlalchemy import (
    select,
)

from sqlalchemy.orm import (
    Session,
)

from database.models.department import (
    Department,
)


class DepartmentRepository:

    def __init__(
        self,
        session: Session,
    ):
        self.session = session

    def get_by_name(
        self,
        tenant_id: int,
        name: str,
    ) -> Department | None:

        return self.session.scalar(

            select(Department)

            .where(

                Department.tenant_id == tenant_id,

                Department.name == name,
            )
        )

    def get_by_id(
        self,
        department_id: int,
    ) -> Department | None:

        return self.session.get(
            Department,
            department_id,
        )
    
    def list_by_tenant(
        self,
        tenant_id: int,
    ) -> list[Department]:

        return list(
            self.session.scalars(
                select(Department)
                .where(
                    Department.tenant_id == tenant_id
                )
                .order_by(
                    Department.name
                )
            )
        )


    def create(
        self,
        department: Department,
    ) -> Department:

        self.session.add(
            department
        )

        self.session.commit()

        self.session.refresh(
            department
        )

        return department


    def delete(
        self,
        department: Department,
    ) -> None:

        self.session.delete(
            department
        )

        self.session.commit()