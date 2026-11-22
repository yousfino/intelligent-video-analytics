
import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


def get_database_url():
    return URL.create(
        drivername="postgresql+psycopg",
        username=os.getenv("POSTGRES_USER", "analytics"),
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=int(os.getenv("POSTGRES_PORT", "5435")),
        database=os.getenv(
            "POSTGRES_DB",
            "video_analytics",
        ),
    )


def create_db_engine():
    return create_engine(
        get_database_url(),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
    )


def check_database_connection():
    engine = create_db_engine()

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT current_database()")
            )

            return result.scalar_one()
    finally:
        engine.dispose()
