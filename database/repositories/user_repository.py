from sqlalchemy import (
    select,
)

from sqlalchemy.orm import (
    Session,
)

from database.models.user import (
    User,
)


class UserRepository:

    def __init__(
        self,
        session: Session,
    ):
        self.session = session

    def get_by_id(
        self,
        user_id: int,
    ) -> User | None:

        return self.session.get(
            User,
            user_id,
        )

    def get_by_username(
        self,
        username: str,
    ) -> User | None:

        return self.session.scalar(

            select(User)

            .where(
                User.username == username
            )
        )
    
    def list_by_tenant(
        self,
        tenant_id: int,
    ) -> list[User]:

        return list(
            self.session.scalars(
                select(User)
                .where(
                    User.tenant_id == tenant_id
                )
                .order_by(
                    User.username
                )
            )
        )


    def get_by_email(
        self,
        email: str,
    ) -> User | None:

        return self.session.scalar(
            select(User)
            .where(
                User.email == email
            )
        )


    def create(
        self,
        user: User,
    ) -> User:

        self.session.add(
            user
        )

        self.session.commit()

        self.session.refresh(
            user
        )

        return user


    def update(
        self,
        user: User,
    ) -> User:

        self.session.commit()

        self.session.refresh(
            user
        )

        return user