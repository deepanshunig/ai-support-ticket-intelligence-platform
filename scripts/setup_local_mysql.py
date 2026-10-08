"""Provision the local MySQL app account and migrate configured PostgreSQL data."""

from __future__ import annotations

import getpass
import os
import shutil
import socket
import sys
from pathlib import Path
from secrets import token_urlsafe

from dotenv import dotenv_values, set_key
from sqlalchemy import text
from sqlalchemy.engine import URL

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
DATABASE = "ticket_intelligence"
APP_USER = "ticket_app"


def _configured_database_url() -> str | None:
    value = os.environ.get("TICKET_DATABASE_URL")
    if value is None:
        value = dotenv_values(ENV_FILE).get("TICKET_DATABASE_URL")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _assert_app_stopped() -> None:
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=0.3):
            raise SystemExit(
                "The API is still listening on port 8000. Run scripts/stop-local.ps1 "
                "before migrating so new PostgreSQL tickets cannot be missed."
            )
    except OSError:
        return


def _mysql_url(username: str, password: str, database: str | None = DATABASE) -> URL:
    return URL.create(
        "mysql+pymysql",
        username=username,
        password=password,
        host="127.0.0.1",
        port=3306,
        database=database,
        query={"charset": "utf8mb4"},
    )


def _verify_existing_mysql(database_url: str) -> None:
    from ticket_intelligence.db.session import create_schema, make_engine

    engine = make_engine(database_url)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        create_schema(engine)
    except Exception:
        raise SystemExit(
            "The configured MySQL database could not be verified. Check the local "
            "MySQL process and ignored .env configuration."
        ) from None
    finally:
        engine.dispose()
    print("Configured MySQL database is ready.")


def _has_app_account(connection: object) -> bool:
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT COUNT(*) FROM mysql.user WHERE User = %s AND Host = %s",
            (APP_USER, "127.0.0.1"),
        )
        return cursor.fetchone()[0] > 0
    finally:
        cursor.close()


def main() -> None:
    _assert_app_stopped()
    database_url = _configured_database_url()
    if database_url and database_url.startswith("mysql+"):
        _verify_existing_mysql(database_url)
        if not ENV_FILE.exists():
            shutil.copy2(ROOT / ".env.example", ENV_FILE)
            set_key(ENV_FILE, "TICKET_DATABASE_URL", database_url)
        values = dotenv_values(ENV_FILE)
        if not values.get("TICKET_REVIEWER_TOKEN"):
            set_key(ENV_FILE, "TICKET_REVIEWER_TOKEN", token_urlsafe(32))
        return
    if database_url and not database_url.startswith("postgresql+"):
        raise SystemExit(
            "Local setup supports MySQL or the existing PostgreSQL source only."
        )

    source_engine = None
    if database_url:
        try:
            import psycopg  # noqa: F401
        except ImportError:
            raise SystemExit(
                "Migrating existing PostgreSQL history needs the optional driver. "
                "Install it with: .\\.venv\\Scripts\\python.exe -m pip install "
                "'psycopg[binary]>=3.2,<4'"
            ) from None
        from ticket_intelligence.db.session import make_migration_source_engine

        try:
            source_engine = make_migration_source_engine(database_url)
            with source_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception:
            if source_engine is not None:
                source_engine.dispose()
            raise SystemExit(
                "The configured PostgreSQL source is unavailable; its data was not changed."
            ) from None

    try:
        admin_user = input("MySQL administrator username [root]: ").strip() or "root"
        admin_password = getpass.getpass("MySQL administrator password (hidden): ")
        import pymysql

        try:
            admin_connection = pymysql.connect(
                host="127.0.0.1",
                port=3306,
                user=admin_user,
                password=admin_password,
                database="mysql",
                charset="utf8mb4",
                autocommit=True,
            )
        except Exception:
            raise SystemExit(
                "MySQL administrator login failed. The PostgreSQL source was not changed."
            ) from None

        with admin_connection:
            cursor = admin_connection.cursor()
            try:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{DATABASE}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
                )
                app_account_exists = _has_app_account(admin_connection)
            finally:
                cursor.close()

        if app_account_exists:
            app_password = getpass.getpass("Existing ticket_app MySQL password (hidden): ")
            with pymysql.connect(
                host="127.0.0.1",
                port=3306,
                user=admin_user,
                password=admin_password,
                database="mysql",
                charset="utf8mb4",
                autocommit=True,
            ) as connection:
                cursor = connection.cursor()
                try:
                    cursor.execute(
                        f"GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX, "
                        f"REFERENCES ON `{DATABASE}`.* TO '{APP_USER}'@'127.0.0.1'"
                    )
                finally:
                    cursor.close()
            _verify_existing_mysql(
                _mysql_url(APP_USER, app_password).render_as_string(hide_password=False)
            )
        else:
            app_password = token_urlsafe(32)

        from ticket_intelligence.db.session import create_schema, make_engine

        admin_engine = make_engine(
            _mysql_url(admin_user, admin_password).render_as_string(hide_password=False)
        )
        try:
            if source_engine is not None:
                from ticket_intelligence.db.migration import migrate_postgres_to_mysql

                report = migrate_postgres_to_mysql(source_engine, admin_engine)
                total = sum(table["rows"] for table in report.values())
                print(f"Verified migration: {total} rows across {len(report)} tables.")
            else:
                create_schema(admin_engine)
        finally:
            admin_engine.dispose()

        if not app_account_exists:
            with pymysql.connect(
                host="127.0.0.1",
                port=3306,
                user=admin_user,
                password=admin_password,
                database="mysql",
                charset="utf8mb4",
                autocommit=True,
            ) as connection:
                cursor = connection.cursor()
                try:
                    cursor.execute(
                        f"CREATE USER '{APP_USER}'@'127.0.0.1' IDENTIFIED BY "
                        f"{connection.escape(app_password)}"
                    )
                finally:
                    cursor.close()

            with pymysql.connect(
                host="127.0.0.1",
                port=3306,
                user=admin_user,
                password=admin_password,
                database="mysql",
                charset="utf8mb4",
                autocommit=True,
            ) as connection:
                cursor = connection.cursor()
                try:
                    cursor.execute(
                        f"GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX, "
                        f"REFERENCES ON `{DATABASE}`.* TO '{APP_USER}'@'127.0.0.1'"
                    )
                finally:
                    cursor.close()

        app_url = _mysql_url(APP_USER, app_password).render_as_string(hide_password=False)
        _verify_existing_mysql(app_url)
        if not ENV_FILE.exists():
            shutil.copy2(ROOT / ".env.example", ENV_FILE)
        set_key(ENV_FILE, "TICKET_DATABASE_URL", app_url)
        values = dotenv_values(ENV_FILE)
        if not values.get("TICKET_REVIEWER_TOKEN"):
            set_key(ENV_FILE, "TICKET_REVIEWER_TOKEN", token_urlsafe(32))
        print("MySQL app configuration saved to the ignored local .env file.")
    finally:
        if source_engine is not None:
            source_engine.dispose()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Setup cancelled; the PostgreSQL source was left unchanged.", file=sys.stderr)
        raise SystemExit(130)
