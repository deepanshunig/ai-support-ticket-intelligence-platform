"""Engine setup for MySQL and isolated SQLite fixture databases."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.pool import StaticPool

from ticket_intelligence.db.models import Base


def make_engine(database_url: str) -> Engine:
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite":
        arguments = {"connect_args": {"check_same_thread": False}, "pool_pre_ping": True}
        if url.database in (None, "", ":memory:"):
            arguments["poolclass"] = StaticPool
        engine = create_engine(url, **arguments)

        @event.listens_for(engine, "connect")
        def enable_foreign_keys(connection: object, _: object) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine
    if url.get_backend_name() == "mysql":
        return create_engine(
            url,
            connect_args={"connect_timeout": 3, "charset": "utf8mb4"},
            pool_pre_ping=True,
            pool_recycle=1800,
        )
    raise ValueError("Use a MySQL database URL, or SQLite for isolated fixtures.")


def make_migration_source_engine(database_url: str) -> Engine:
    """Create the PostgreSQL source engine used only for the one-time migration."""
    url = make_url(database_url)
    if url.get_backend_name() != "postgresql":
        raise ValueError("The migration source must be a PostgreSQL database URL.")
    return create_engine(url, connect_args={"connect_timeout": 3}, pool_pre_ping=True)


def create_schema(engine: Engine) -> None:
    """Create the initial tables; existing table contents are preserved."""
    Base.metadata.create_all(engine)
