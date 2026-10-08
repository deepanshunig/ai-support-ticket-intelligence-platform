"""Transactional ticket operations with optimistic review concurrency control."""

from __future__ import annotations

import copy
import math
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, selectinload

from ticket_intelligence.db.models import Ticket, TicketEvent, TicketReview

STATUSES = {"needs_review", "assigned", "resolved"}


class TicketNotFound(LookupError):
    """The requested ticket does not exist."""


class VersionConflict(RuntimeError):
    """Another review changed this ticket since the client fetched it."""


def _iso(timestamp: datetime) -> str:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _state(ticket: Ticket) -> dict[str, Any]:
    return {
        "current_category": ticket.current_category,
        "assigned_team": ticket.assigned_team,
        "status": ticket.status,
        "version": ticket.version,
    }


def _serialize(ticket: Ticket, *, detailed: bool = True) -> dict[str, Any]:
    result = {
        "id": ticket.id,
        "text": ticket.text,
        "created_at": _iso(ticket.created_at),
        "updated_at": _iso(ticket.updated_at),
        **_state(ticket),
        "prediction": copy.deepcopy(ticket.prediction),
        "model_version": ticket.model_version,
        "confidence": ticket.confidence,
    }
    if detailed:
        result["events"] = [
            {
                "id": event.id,
                "type": event.type,
                "created_at": _iso(event.created_at),
                "actor_id": event.actor_id,
                "payload": copy.deepcopy(event.payload),
            }
            for event in ticket.events
        ]
        result["reviews"] = [
            {
                "id": review.id,
                "created_at": _iso(review.created_at),
                "reviewer_id": review.reviewer_id,
                "note": review.note,
                "before": copy.deepcopy(review.before_state),
                "after": copy.deepcopy(review.after_state),
            }
            for review in ticket.reviews
        ]
    return result


class TicketRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, ticket_id: str) -> dict[str, Any]:
        statement = (
            select(Ticket)
            .where(Ticket.id == ticket_id)
            .options(selectinload(Ticket.events), selectinload(Ticket.reviews))
            .execution_options(populate_existing=True)
        )
        ticket = self.session.scalar(statement)
        if ticket is None:
            raise TicketNotFound(f"Ticket {ticket_id} was not found.")
        return _serialize(ticket)

    def create(self, text: str, prediction: dict[str, Any]) -> dict[str, Any]:
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Ticket text must be nonempty.")
            category = prediction.get("predicted_category")
            model_version = prediction.get("model_version")
            confidence = float(prediction["confidence"])
            action = prediction.get("routing_action")
            if not isinstance(category, str) or not category.strip():
                raise ValueError("The prediction must include a category.")
            if not isinstance(model_version, str) or not model_version.strip():
                raise ValueError("The prediction must include a model version.")
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Confidence must be finite and between zero and one.")
            if action not in {"needs_review", "assigned"}:
                raise ValueError("Routing action must be needs_review or assigned.")
            assigned_team = prediction.get("assigned_team") if action == "assigned" else None
            if action == "assigned" and (
                not isinstance(assigned_team, str) or not assigned_team.strip()
            ):
                raise ValueError("Assigned tickets require an assigned team.")
            now = datetime.now(UTC)
            ticket = Ticket(
                id=str(uuid4()),
                text=text,
                created_at=now,
                updated_at=now,
                status=action,
                version=1,
                prediction=copy.deepcopy(prediction),
                current_category=category,
                assigned_team=assigned_team,
                model_version=model_version,
                confidence=confidence,
            )
            self.session.add(ticket)
            self.session.add(
                TicketEvent(
                    ticket=ticket,
                    type="triaged",
                    created_at=now,
                    actor_id="system",
                    payload={"prediction": copy.deepcopy(prediction), "state": _state(ticket)},
                )
            )
            self.session.commit()
            self.session.expire_all()
            return self.get(ticket.id)
        except Exception:
            self.session.rollback()
            raise

    def list(
        self, status: str | None = None, team: str | None = None, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        if status is not None and status not in STATUSES:
            raise ValueError("Unknown ticket status.")
        if not 1 <= limit <= 200 or offset < 0:
            raise ValueError("Limit must be 1..200 and offset must be nonnegative.")
        filters = []
        if status is not None:
            filters.append(Ticket.status == status)
        if team is not None:
            filters.append(
                Ticket.assigned_team.is_(None)
                if team == "unassigned"
                else Ticket.assigned_team == team
            )
        total = self.session.scalar(select(func.count()).select_from(Ticket).where(*filters))
        statement = (
            select(Ticket)
            .where(*filters)
            .order_by(Ticket.created_at.desc(), Ticket.id.desc())
            .limit(limit)
            .offset(offset)
            .execution_options(populate_existing=True)
        )
        return {
            "items": [
                _serialize(ticket, detailed=False) for ticket in self.session.scalars(statement)
            ],
            "total": int(total or 0),
        }

    def review(
        self,
        ticket_id: str,
        *,
        expected_version: int,
        category: str,
        team: str,
        status: str,
        note: str,
        reviewer_id: str,
    ) -> dict[str, Any]:
        try:
            if type(expected_version) is not int or expected_version < 1:
                raise ValueError("Expected version must be a positive integer.")
            if status not in STATUSES:
                raise ValueError("Unknown review status.")
            if any(not isinstance(value, str) or not value.strip()
                   for value in (category, team, reviewer_id)):
                raise ValueError("Category, team and reviewer identity must be nonempty.")
            if not isinstance(note, str):
                raise ValueError("Review note must be a string.")
            ticket = self.session.get(Ticket, ticket_id, populate_existing=True)
            if ticket is None:
                raise TicketNotFound(f"Ticket {ticket_id} was not found.")
            before = _state(ticket)
            now = datetime.now(UTC)
            assigned_team = None if status == "needs_review" else team
            after = {
                "current_category": category,
                "assigned_team": assigned_team,
                "status": status,
                "version": expected_version + 1,
            }
            changed = self.session.execute(
                update(Ticket)
                .where(Ticket.id == ticket_id, Ticket.version == expected_version)
                .values(**after, updated_at=now)
                .execution_options(synchronize_session=False)
            )
            if changed.rowcount != 1:
                raise VersionConflict("The ticket changed; reload it before submitting the review.")
            review = TicketReview(
                ticket_id=ticket_id,
                created_at=now,
                reviewer_id=reviewer_id,
                note=note,
                before_state=before,
                after_state=after,
            )
            self.session.add(review)
            self.session.flush()
            self.session.add(
                TicketEvent(
                    ticket_id=ticket_id,
                    type="reviewed",
                    created_at=now,
                    actor_id=reviewer_id,
                    payload={
                        "review_id": review.id, "before": before, "after": after, "note": note
                    },
                )
            )
            self.session.commit()
            self.session.expire_all()
            return self.get(ticket_id)
        except Exception:
            self.session.rollback()
            raise

    def metrics(self) -> dict[str, Any]:
        by_status = {status: 0 for status in sorted(STATUSES)}
        for status, count in self.session.execute(
            select(Ticket.status, func.count()).group_by(Ticket.status)
        ):
            by_status[status] = int(count)
        by_team = {"unassigned": 0}
        for team, count in self.session.execute(
            select(Ticket.assigned_team, func.count()).group_by(Ticket.assigned_team)
        ):
            by_team[team if team is not None else "unassigned"] = int(count)
        review_count = self.session.scalar(select(func.count()).select_from(TicketReview))
        return {
            "total_tickets": sum(by_status.values()),
            "by_status": by_status,
            "by_team": by_team,
            "review_count": int(review_count or 0),
        }
