import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models.base import Base


load_dotenv()


def get_database_url() -> str:

    # Production: Neon / Render
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        return database_url

    # Local development
    password = quote_plus(
        os.getenv("DATABASE_PASSWORD", "")
    )

    return (
        f"postgresql://"
        f"{os.getenv('DATABASE_USER')}:"
        f"{password}@"
        f"{os.getenv('DATABASE_HOST')}:"
        f"{os.getenv('DATABASE_PORT')}/"
        f"{os.getenv('DATABASE_NAME')}"
    )


DATABASE_URL = get_database_url()


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()