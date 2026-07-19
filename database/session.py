import os
from urllib.parse import quote_plus

from dotenv import load_dotenv

from sqlalchemy import (
    create_engine,
)

from sqlalchemy.orm import (
    sessionmaker,
)

from database.models.base import (
    Base,
)

# ----------------------------------
# Load environment variables
# ----------------------------------

load_dotenv()

password = quote_plus(
    os.getenv("DATABASE_PASSWORD")
)

DATABASE_URL = (
    f"postgresql://"
    f"{os.getenv('DATABASE_USER')}:"
    f"{password}@"
    f"{os.getenv('DATABASE_HOST')}:"
    f"{os.getenv('DATABASE_PORT')}/"
    f"{os.getenv('DATABASE_NAME')}"
)

# ----------------------------------
# SQLAlchemy Engine
# ----------------------------------

engine = create_engine(
    DATABASE_URL,
)

# ----------------------------------
# Session Factory
# ----------------------------------

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# ----------------------------------
# Dependency
# ----------------------------------

def get_db():

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close()