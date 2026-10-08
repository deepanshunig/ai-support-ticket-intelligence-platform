"""Initial relational schema for ticket state and its immutable audit history."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.mysql import DATETIME as MySQLDateTime
from sqlalchemy.dialects.mysql import DOUBLE as MySQLDouble
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

UTC_DATETIME = DateTime(timezone=True).with_variant(MySQLDateTime(fsp=6), "mysql")
CONFIDENCE_FLOAT = Float().with_variant(MySQLDouble(asdecimal=False), "mysql")


class Base(DeclarativeBase):
    pass


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (
        CheckConstraint("status IN ('needs_review', 'assigned', 'resolved')", name="ticket_status"),
        CheckConstraint("version >= 1", name="ticket_positive_version"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ticket_confidence_bounds"),
        Index("ix_ticket_status_created_at", "status", "created_at"),
        Index("ix_ticket_assigned_team_created_at", "assigned_team", "created_at"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTC_DATETIME, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    prediction: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    current_category: Mapped[str] = mapped_column(String(128), nullable=False)
    assigned_team: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[str] = mapped_column(String(256), nullable=False)
    confidence: Mapped[float] = mapped_column(CONFIDENCE_FLOAT, nullable=False)
    events: Mapped[list[TicketEvent]] = relationship(
        back_populates="ticket", order_by="(TicketEvent.created_at, TicketEvent.id)"
    )
    reviews: Mapped[list[TicketReview]] = relationship(
        back_populates="ticket", order_by="(TicketReview.created_at, TicketReview.id)"
    )


class TicketEvent(Base):
    __tablename__ = "ticket_events"
    __table_args__ = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(ForeignKey("tickets.id"), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, nullable=False)
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    ticket: Mapped[Ticket] = relationship(back_populates="events")


class TicketReview(Base):
    __tablename__ = "ticket_reviews"
    __table_args__ = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(ForeignKey("tickets.id"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, nullable=False)
    reviewer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False)
    before_state: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    after_state: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    ticket: Mapped[Ticket] = relationship(back_populates="reviews")
