"""Copy ticket history from PostgreSQL to an empty MySQL database safely."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Engine, select, text
from sqlalchemy.sql.schema import Table

from ticket_intelligence.db.models import Base
from ticket_intelligence.db.session import create_schema


def _utc_naive(value: datetime) -> datetime:
    """MySQL DATETIME has no timezone; persist the UTC wall time explicitly."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC).replace(tzinfo=None)
    return value.astimezone(UTC).replace(tzinfo=None)


def _mysql_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _utc_naive(value)
    if isinstance(value, dict):
        return {key: _mysql_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_mysql_value(item) for item in value]
    return value


def _canonical(value: Any) -> Any:
    if isinstance(value, datetime):
        timestamp = _utc_naive(value).replace(tzinfo=UTC)
        return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (float, Decimal)):
        # Drivers may render the same DOUBLE with adjacent final decimal digits.
        return {"__numeric__": format(float(value), ".15g")}
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _digest_rows(connection: Any, table: Table) -> tuple[int, str, dict[str, str]]:
    primary_key = tuple(table.primary_key.columns)
    statement = select(table).order_by(*primary_key)
    digest = hashlib.sha256()
    column_digests = {column.name: hashlib.sha256() for column in table.columns}
    count = 0
    result = connection.execute(statement)
    while rows := result.mappings().fetchmany(500):
        for row in rows:
            payload = {column.name: _canonical(row[column.name]) for column in table.columns}
            digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\n")
            for name, value in payload.items():
                column_digests[name].update(
                    json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
                )
                column_digests[name].update(b"\n")
            count += 1
    return count, digest.hexdigest(), {
        name: column_digest.hexdigest() for name, column_digest in column_digests.items()
    }


def _ensure_lossless_mysql_types(destination: Engine, tables: tuple[Table, ...]) -> None:
    """Upgrade an empty prior-attempt schema from FLOAT to lossless DOUBLE."""
    with destination.begin() as connection:
        for table in tables:
            if connection.scalar(select(table).limit(1)) is not None:
                raise ValueError(
                    f"MySQL table {table.name} is not empty; refusing to overwrite it."
                )
        column_type = connection.scalar(
            text(
                "SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'tickets' "
                "AND COLUMN_NAME = 'confidence'"
            )
        )
        if column_type is None:
            raise RuntimeError("MySQL ticket schema is missing the confidence column.")
        if column_type.lower() == "float":
            connection.execute(
                text("ALTER TABLE tickets MODIFY confidence DOUBLE NOT NULL")
            )
            column_type = connection.scalar(
                text(
                    "SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'tickets' "
                    "AND COLUMN_NAME = 'confidence'"
                )
            )
        if column_type.lower() != "double":
            raise RuntimeError("MySQL confidence must use DOUBLE to preserve score precision.")


def migrate_postgres_to_mysql(source: Engine, destination: Engine) -> dict[str, dict[str, Any]]:
    """Copy all application tables transactionally, refusing any nonempty target.

    A repeatable-read PostgreSQL snapshot keeps the copied tables consistent even
    if another client attempts writes while this function runs. Callers should
    stop the application before migration so no tickets are created after the
    snapshot and before the app is switched to MySQL.
    """
    if source.dialect.name != "postgresql":
        raise ValueError("The migration source must be PostgreSQL.")
    if destination.dialect.name != "mysql":
        raise ValueError("The migration destination must be MySQL.")

    create_schema(destination)
    tables = tuple(Base.metadata.sorted_tables)
    _ensure_lossless_mysql_types(destination, tables)
    report: dict[str, dict[str, Any]] = {}
    with source.connect().execution_options(isolation_level="REPEATABLE READ") as source_conn:
        with source_conn.begin():
            with destination.begin() as destination_conn:
                for table in tables:
                    if destination_conn.scalar(select(table).limit(1)) is not None:
                        raise ValueError(
                            f"MySQL table {table.name} is not empty; refusing to overwrite it."
                        )

                for table in tables:
                    source_digest = hashlib.sha256()
                    source_column_digests = {
                        column.name: hashlib.sha256() for column in table.columns
                    }
                    source_confidences: dict[str, float] = {}
                    source_count = 0
                    primary_key = tuple(table.primary_key.columns)
                    result = source_conn.execute(select(table).order_by(*primary_key))
                    while rows := result.mappings().fetchmany(500):
                        mysql_rows = []
                        for row in rows:
                            if table.name == "tickets":
                                source_confidences[str(row["id"])] = float(row["confidence"])
                            record = {
                                column.name: _mysql_value(row[column.name])
                                for column in table.columns
                            }
                            canonical = {
                                column.name: _canonical(row[column.name])
                                for column in table.columns
                            }
                            source_digest.update(
                                json.dumps(
                                    canonical, sort_keys=True, separators=(",", ":")
                                ).encode()
                            )
                            source_digest.update(b"\n")
                            for name, value in canonical.items():
                                source_column_digests[name].update(
                                    json.dumps(
                                        value, sort_keys=True, separators=(",", ":")
                                    ).encode()
                                )
                                source_column_digests[name].update(b"\n")
                            mysql_rows.append(record)
                            source_count += 1
                        if mysql_rows:
                            destination_conn.execute(table.insert(), mysql_rows)

                    destination_count, destination_digest, destination_columns = _digest_rows(
                        destination_conn, table
                    )
                    expected_digest = source_digest.hexdigest()
                    if source_count != destination_count or expected_digest != destination_digest:
                        differing_columns = [
                            name
                            for name, column_digest in source_column_digests.items()
                            if column_digest.hexdigest() != destination_columns[name]
                        ]
                        details = (
                            f" Differing columns: {', '.join(differing_columns)}."
                            if differing_columns
                            else ""
                        )
                        if table.name == "tickets" and source_confidences:
                            target_confidences = {
                                str(ticket_id): float(confidence)
                                for ticket_id, confidence in destination_conn.execute(
                                    select(table.c.id, table.c.confidence)
                                )
                            }
                            deltas = [
                                abs(value - target_confidences[ticket_id])
                                for ticket_id, value in source_confidences.items()
                            ]
                            details += f" Maximum confidence delta: {max(deltas):.12g}."
                        raise RuntimeError(
                            f"Verification failed for {table.name}.{details} "
                            "MySQL transaction rolled back."
                        )
                    report[table.name] = {
                        "rows": source_count,
                        "sha256": expected_digest,
                    }
    return report
