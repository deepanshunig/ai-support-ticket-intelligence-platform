"""Exercise real PostgreSQL persistence and simultaneous review conflict handling."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from sqlalchemy import text
from sqlalchemy.orm import Session

from ticket_intelligence.core.config import Settings
from ticket_intelligence.db.repository import TicketRepository, VersionConflict
from ticket_intelligence.db.session import create_schema, make_engine
from ticket_intelligence.inference.predictor import Predictor


def main() -> None:
    settings = Settings()
    if settings.database_url is None:
        raise ValueError("Configure PostgreSQL first.")
    engine = make_engine(settings.database_url.get_secret_value())
    if engine.dialect.name != "postgresql":
        raise ValueError("This check must use real PostgreSQL.")
    create_schema(engine)
    predictor = Predictor.from_artifacts(settings.artifact_dir, settings.team_mapping_path)
    prediction = predictor.predict("My card payment was declined in a shop.") | {
        "routing_action": "needs_review", "assigned_team": None,
        "policy_version": "manual-review-v1", "review_reason": "policy_not_verified",
    }
    with Session(engine) as session:
        ticket = TicketRepository(session).create(
            "My card payment was declined in a shop.", prediction
        )
    barrier = Barrier(2)

    def review() -> str:
        with Session(engine) as session:
            barrier.wait(timeout=5)
            try:
                TicketRepository(session).review(
                    ticket["id"], expected_version=1, category="declined_card_payment",
                    team="card_operations", status="assigned", note="Confirmed demo card issue.",
                    reviewer_id="integration-probe",
                )
                return "committed"
            except VersionConflict:
                return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: review(), range(2)))
    assert sorted(results) == ["committed", "conflict"]
    with Session(engine) as session:
        stored = TicketRepository(session).get(ticket["id"])
        assert stored["prediction"] == prediction
        assert stored["version"] == 2 and len(stored["reviews"]) == 1
        assert len(stored["events"]) == 2
        assert stored["assigned_team"] == "card_operations"
        is_superuser = session.execute(
            text("SELECT rolsuper FROM pg_roles WHERE rolname=current_user")
        ).scalar()
        assert is_superuser is False
        version = session.execute(text("SHOW server_version")).scalar()
    report = {
        "database": "PostgreSQL", "version": version, "demo_ticket_id": ticket["id"],
        "checks": ["schema", "persist across sessions", "original prediction unchanged",
                   "simultaneous reviews: one commit, one conflict", "one correction/event pair",
                   "non-superuser application role"],
    }
    directory = Path("reports/integration")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "postgres_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    engine.dispose()


if __name__ == "__main__":
    main()
