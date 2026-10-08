"""SQLite fixture tests for history preservation and atomic review updates."""

from __future__ import annotations

import copy
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, event, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ticket_intelligence.db.models import Ticket, TicketEvent
from ticket_intelligence.db.repository import TicketNotFound, TicketRepository, VersionConflict
from ticket_intelligence.db.session import create_schema, make_engine


@pytest.fixture
def engine() -> Iterator[Engine]:
    database = make_engine("sqlite+pysqlite:///:memory:")
    create_schema(database)
    yield database
    database.dispose()


@pytest.fixture
def prediction() -> dict:
    return {
        "predicted_category": "card_arrival",
        "proposed_team": "cards",
        "confidence": 0.83,
        "model_version": "logistic_regression-fixture",
        "candidates": [{"category": "card_arrival", "probability": 0.83}],
        "artifact_sha256": "a" * 64,
        "routing_action": "needs_review",
        "assigned_team": None,
        "policy_version": "manual-review-v1",
    }


def review(repository: TicketRepository, ticket_id: str, *, version: int = 1) -> dict:
    return repository.review(
        ticket_id,
        expected_version=version,
        category="cash_transfer",
        team="payments",
        status="assigned",
        note="The request describes a transfer.",
        reviewer_id="demo-reviewer",
    )


def test_creation_and_review_preserve_original_prediction(engine: Engine, prediction: dict) -> None:
    assert isinstance(engine.pool, StaticPool)
    original = copy.deepcopy(prediction)
    with Session(engine, expire_on_commit=False) as session:
        repository = TicketRepository(session)
        created = repository.create("Where is my card?", prediction)
        assert created["status"] == "needs_review"
        assert created["assigned_team"] is None
        assert created["version"] == 1
        assert created["created_at"].endswith("Z")
        assert len(created["events"]) == 1
        assert created["events"][0]["type"] == "triaged"
        prediction["candidates"][0]["probability"] = 0.0
        created["prediction"]["candidates"][0]["probability"] = 0.1
        corrected = review(repository, created["id"])
        assert corrected["prediction"] == original
        assert corrected["events"][0]["payload"]["prediction"] == original
        assert corrected["current_category"] == "cash_transfer"
        assert corrected["assigned_team"] == "payments"
        assert corrected["version"] == 2
        assert corrected["confidence"] == original["confidence"]
        assert corrected["model_version"] == original["model_version"]
        assert [entry["type"] for entry in corrected["events"]] == ["triaged", "reviewed"]
        assert corrected["events"][1]["actor_id"] == "demo-reviewer"
        assert corrected["reviews"][0]["before"]["version"] == 1
        assert corrected["reviews"][0]["after"]["version"] == 2
        assert corrected["reviews"][0]["reviewer_id"] == "demo-reviewer"
    with Session(engine) as session:
        persisted = TicketRepository(session).get(created["id"])
        assert persisted["prediction"] == original
        assert len(persisted["reviews"]) == 1


def test_stale_review_does_not_add_correction_or_event(engine: Engine, prediction: dict) -> None:
    with Session(engine) as first, Session(engine) as second:
        first_repository, second_repository = TicketRepository(first), TicketRepository(second)
        created = first_repository.create("A transfer is pending", prediction)
        stale = second_repository.get(created["id"])
        review(first_repository, created["id"])
        with pytest.raises(VersionConflict):
            review(second_repository, created["id"], version=stale["version"])
        latest = second_repository.get(created["id"])
        assert latest["version"] == 2
        assert len(latest["reviews"]) == 1
        assert len(latest["events"]) == 2
        assert second_repository.metrics()["review_count"] == 1


def test_event_failure_rolls_back_review_and_ticket_update(
    engine: Engine, prediction: dict
) -> None:
    with Session(engine) as session:
        repository = TicketRepository(session)
        created = repository.create("A pending transfer", prediction)

        def fail_review_event(current: Session, _: object, __: object) -> None:
            if any(isinstance(record, TicketEvent) and record.type == "reviewed"
                   for record in current.new):
                raise RuntimeError("Injected event persistence failure")

        event.listen(session, "before_flush", fail_review_event)
        with pytest.raises(RuntimeError, match="Injected event"):
            review(repository, created["id"])
        event.remove(session, "before_flush", fail_review_event)
        unchanged = repository.get(created["id"])
        assert unchanged["version"] == 1
        assert unchanged["current_category"] == "card_arrival"
        assert unchanged["reviews"] == []
        assert len(unchanged["events"]) == 1
        assert repository.metrics()["review_count"] == 0


def test_event_failure_rolls_back_creation(engine: Engine, prediction: dict) -> None:
    with Session(engine) as session:
        repository = TicketRepository(session)

        def fail_event(current: Session, _: object, __: object) -> None:
            if any(isinstance(record, TicketEvent) for record in current.new):
                raise RuntimeError("Injected triage event failure")

        event.listen(session, "before_flush", fail_event)
        with pytest.raises(RuntimeError, match="Injected triage"):
            repository.create("A new request", prediction)
        event.remove(session, "before_flush", fail_event)
        assert repository.metrics()["total_tickets"] == 0
        assert repository.list()["items"] == []


def test_filters_pagination_and_metrics_use_current_state(engine: Engine, prediction: dict) -> None:
    with Session(engine) as session:
        repository = TicketRepository(session)
        first = repository.create("First request", prediction)
        repository.create("Second request", prediction)
        assigned = {**prediction, "routing_action": "assigned", "assigned_team": "cards"}
        repository.create("Third request", assigned)
        review(repository, first["id"])
        first_page = repository.list(status="assigned", limit=1)
        second_page = repository.list(status="assigned", limit=1, offset=1)
        assert first_page["total"] == second_page["total"] == 2
        assert first_page["items"][0]["id"] != second_page["items"][0]["id"]
        assert repository.list(team="payments")["total"] == 1
        assert repository.list(team="unassigned")["total"] == 1
        assert repository.metrics() == {
            "total_tickets": 3,
            "by_status": {"assigned": 2, "needs_review": 1, "resolved": 0},
            "by_team": {"unassigned": 1, "cards": 1, "payments": 1},
            "review_count": 1,
        }


@pytest.mark.parametrize(
    "field,value", [("status", "invalid"), ("version", 0), ("confidence", 1.1)]
)
def test_sql_constraints_reject_invalid_state(
    engine: Engine, prediction: dict, field: str, value: object
) -> None:
    with Session(engine) as session:
        repository = TicketRepository(session)
        created = repository.create("Constraint fixture", prediction)
        with pytest.raises(IntegrityError):
            session.execute(update(Ticket).where(Ticket.id == created["id"]).values({field: value}))
            session.commit()
        session.rollback()
        assert repository.get(created["id"])["version"] == 1
        assert repository.metrics()["review_count"] == 0


def test_missing_ticket_errors_leave_session_usable(engine: Engine, prediction: dict) -> None:
    with Session(engine) as session:
        repository = TicketRepository(session)
        with pytest.raises(TicketNotFound):
            repository.get("missing-id")
        with pytest.raises(TicketNotFound):
            review(repository, "missing-id")
        assert repository.create("Valid request", prediction)["version"] == 1


def test_nonfinite_confidence_is_rejected_without_persistence(
    engine: Engine, prediction: dict
) -> None:
    with Session(engine) as session:
        repository = TicketRepository(session)
        with pytest.raises(ValueError, match="finite"):
            repository.create("Invalid confidence", {**prediction, "confidence": float("nan")})
        assert repository.metrics()["total_tickets"] == 0
