"""Checks that the shared schema emits the intended MySQL types and guarantees."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy.schema import CreateTable

from ticket_intelligence.db.migration import _canonical, _mysql_value
from ticket_intelligence.db.models import Base, Ticket
from ticket_intelligence.db.session import make_engine


def test_mysql_engine_and_schema_use_mysql_native_types() -> None:
    engine = make_engine(
        "mysql+pymysql://ticket_app:example@127.0.0.1:3306/ticket_intelligence"
    )
    try:
        assert engine.dialect.name == "mysql"
        ddl = {
            table.name: str(CreateTable(table).compile(engine)).upper()
            for table in Base.metadata.sorted_tables
        }
        assert "DATETIME(6)" in ddl["tickets"]
        assert "JSON NOT NULL" in ddl["tickets"]
        assert "CONFIDENCE DOUBLE NOT NULL" in ddl["tickets"]
        confidence_type = Ticket.__table__.c.confidence.type.dialect_impl(engine.dialect)
        assert confidence_type.asdecimal is False
        assert "CHECK" in ddl["tickets"]
        for statement in ddl.values():
            assert "ENGINE=INNODB" in statement
            assert "CHARSET=UTF8MB4" in statement
        assert set(Base.metadata.tables) == {"tickets", "ticket_events", "ticket_reviews"}
    finally:
        engine.dispose()


def test_migration_keeps_timezone_values_as_utc_in_mysql() -> None:
    source_time = datetime(2026, 10, 8, 12, 30, tzinfo=UTC)
    stored = _mysql_value(source_time)
    assert stored == datetime(2026, 10, 8, 12, 30)
    assert _canonical(stored) == "2026-10-08T12:30:00.000000Z"


def test_migration_rejects_non_utc_timestamp_offsets_without_shifting_time() -> None:
    source_time = datetime.fromisoformat("2026-10-08T18:00:00+05:30")
    assert _mysql_value(source_time) == datetime(2026, 10, 8, 12, 30)


def test_migration_hashes_equivalent_float_and_decimal_values_identically() -> None:
    assert _canonical(0.83) == _canonical(Decimal("0.83000000000000002"))


def test_postgresql_is_not_accepted_as_the_runtime_database() -> None:
    with pytest.raises(ValueError, match="MySQL database URL"):
        make_engine("postgresql+psycopg://user:password@127.0.0.1/ticket_intelligence")
