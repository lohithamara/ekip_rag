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

    def get_by_id(
        self,
        user_id: int,
    ) -> User | None:

        return self.session.get(
            User,
            user_id,
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