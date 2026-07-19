from sqlalchemy import (
    text,
)

from database.session import (
    engine,
)

with engine.connect() as connection:

    result = connection.execute(
        text("SELECT version();")
    )

    print()

    print("=" * 80)
    print("DATABASE CONNECTION")
    print("=" * 80)

    print(result.scalar())

    print()

    print("=" * 80)
    print("STATUS : PASS")
    print("=" * 80)