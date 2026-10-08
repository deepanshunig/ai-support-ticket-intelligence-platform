"""Authenticated baseline predictions and durable human-review ticket workflow."""

from __future__ import annotations

import hmac
import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ticket_intelligence import __version__
from ticket_intelligence.api.schemas import ReviewInput, TicketInput
from ticket_intelligence.core.config import Settings
from ticket_intelligence.db.repository import TicketNotFound, TicketRepository, VersionConflict
from ticket_intelligence.db.session import create_schema, make_engine
from ticket_intelligence.inference.predictor import ArtifactError, Predictor
from ticket_intelligence.inference.routing import RoutingEngine
from ticket_intelligence.integrations.webhooks import WebhookNotifier, ticket_event

logger = logging.getLogger(__name__)


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _manual_policy_metadata() -> dict[str, Any]:
    return {
        "mode": "manual_review",
        "policy_version": "manual-review-v1",
        "active": False,
        "temperature": None,
        "threshold": None,
        "margin": None,
        "guard_version": None,
    }


def create_app(
    settings: Settings | None = None,
    *,
    predictor: Predictor | None = None,
    engine: Engine | None = None,
    webhook_notifier: WebhookNotifier | None = None,
) -> FastAPI:
    settings = settings or Settings()
    if webhook_notifier is None:
        webhook_notifier = WebhookNotifier(
            settings.webhook_url.get_secret_value() if settings.webhook_url else None,
            settings.webhook_secret.get_secret_value() if settings.webhook_secret else None,
            environment=settings.environment,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=settings.log_level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        logger.info("Starting API environment=%s phase=ticket_workflow", settings.environment)
        owned_engine = None
        app.state.router = None
        app.state.routing_unavailable = False
        app.state.routing_policy = _manual_policy_metadata()
        try:
            loaded = predictor or Predictor.from_artifacts(
                settings.artifact_dir, settings.team_mapping_path
            )
            configuration = json.loads(settings.team_mapping_path.read_text(encoding="utf-8"))
            mapping, teams = configuration["intent_to_team"], configuration["teams"]
            team_ids = {team["id"] for team in teams}
            if set(mapping) != set(loaded.classes) or any(
                team not in team_ids for team in mapping.values()
            ):
                raise ValueError("Model and team mapping coverage do not agree.")
            app.state.teams = [{"id": team["id"], "name": team["name"]} for team in teams]
            app.state.intent_to_team = dict(mapping)
            app.state.predictor = loaded
        except Exception as error:
            logger.warning("Model startup unavailable error_type=%s", type(error).__name__)
            app.state.predictor = None
        if settings.policy_path is not None and settings.policy_path.exists():
            try:
                if app.state.predictor is None:
                    raise ArtifactError("The model required by the routing policy is unavailable.")
                app.state.router = RoutingEngine.from_policy(
                    app.state.predictor, settings.policy_path, settings.team_mapping_path
                )
                app.state.routing_policy = app.state.router.policy_metadata()
            except Exception as error:
                logger.warning("Routing startup unavailable error_type=%s", type(error).__name__)
                app.state.router = None
                app.state.routing_unavailable = True
                app.state.routing_policy = {**_manual_policy_metadata(), "mode": "unavailable"}
        if engine is not None or settings.database_url is not None:
            try:
                database_engine = engine
                if database_engine is None:
                    database_engine = make_engine(settings.database_url.get_secret_value())
                    owned_engine = database_engine
                create_schema(database_engine)
                with database_engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
                app.state.engine = database_engine
            except Exception as error:
                logger.warning("Database startup unavailable error_type=%s", type(error).__name__)
                app.state.engine = None
        try:
            yield
        finally:
            if owned_engine is not None:
                owned_engine.dispose()
            logger.info("API stopped")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Banking-support prototype with baseline prediction and authenticated human review. "
            "A verified frozen policy enables calibrated routing; absent policy uses human review."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.predictor = None
    app.state.router = None
    app.state.routing_unavailable = False
    app.state.routing_policy = _manual_policy_metadata()
    app.state.engine = None
    app.state.webhook_notifier = webhook_notifier
    app.state.intent_to_team = {}
    app.state.teams = []

    def authenticate(
        supplied: Annotated[str | None, Header(alias="X-Reviewer-Token")] = None,
    ) -> None:
        configured = settings.reviewer_token
        if configured is None or not configured.get_secret_value().strip():
            raise _error(
                503, "reviewer_auth_unconfigured", "Reviewer authentication is unavailable."
            )
        if supplied is None or not hmac.compare_digest(
            supplied.encode(), configured.get_secret_value().encode()
        ):
            raise _error(401, "unauthorized", "A valid reviewer token is required.")

    def require_runtime(_: None = Depends(authenticate)) -> None:
        if app.state.routing_unavailable:
            raise _error(
                503, "service_unavailable", "The configured routing policy is unavailable."
            )
        if app.state.predictor is None or app.state.engine is None:
            raise _error(
                503, "service_unavailable", "Model and database dependencies must be ready."
            )

    def repository(_: None = Depends(require_runtime)) -> Iterator[TicketRepository]:
        with Session(app.state.engine) as session:
            yield TicketRepository(session)

    def triage(value: str) -> dict[str, Any]:
        if app.state.router is not None:
            return app.state.router.predict(value)
        return {
            **app.state.predictor.predict(value),
            "routing_action": "needs_review",
            "assigned_team": None,
            "review_reason": "policy_not_verified",
            "policy_version": "manual-review-v1",
        }

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "detail": {
                    "code": "invalid_request",
                    "message": "Request data did not pass validation.",
                    "errors": [
                        {"field": ".".join(map(str, item["loc"])), "message": item["msg"]}
                        for item in error.errors()
                    ],
                }
            },
        )

    @app.exception_handler(TicketNotFound)
    async def missing_ticket(_request, _error_value) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"detail": {"code": "ticket_not_found", "message": "Ticket was not found."}},
        )

    @app.exception_handler(VersionConflict)
    async def conflicting_review(_request, _error_value) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": {
                    "code": "version_conflict",
                    "message": "Ticket changed; reload it before submitting another review.",
                }
            },
        )

    @app.exception_handler(SQLAlchemyError)
    @app.exception_handler(ArtifactError)
    async def unavailable_dependency(_request, error) -> JSONResponse:
        logger.warning("Request dependency unavailable error_type=%s", type(error).__name__)
        return JSONResponse(
            status_code=503,
            content={
                "detail": {
                    "code": "service_unavailable",
                    "message": "A dependency is unavailable.",
                }
            },
        )

    @app.get("/", tags=["Service"])
    def service_info() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": __version__,
            "phase": "ticket_workflow",
            "docs": "/docs",
        }

    @app.get("/health", tags=["Service"])
    def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name, "version": __version__}

    @app.get("/ready", tags=["Service"])
    def ready(response: Response) -> dict[str, Any]:
        database_status = "ready" if app.state.engine is not None else "not_configured"
        if app.state.engine is None and (engine is not None or settings.database_url is not None):
            database_status = "unavailable"
        if app.state.engine is not None:
            try:
                with app.state.engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
            except SQLAlchemyError:
                database_status = "unavailable"
        checks = {
            "api": "ready",
            "model": "ready" if app.state.predictor is not None else "unavailable",
            "database": database_status,
            "reviewer_auth": (
                "ready"
                if settings.reviewer_token and settings.reviewer_token.get_secret_value().strip()
                else "not_configured"
            ),
            "routing_policy": (
                "unavailable"
                if app.state.routing_unavailable
                else "ready"
                if app.state.router is not None
                else "manual_review"
            ),
        }
        is_ready = not app.state.routing_unavailable and all(
            value == "ready" for name, value in checks.items() if name != "routing_policy"
        )
        response.status_code = 200 if is_ready else 503
        return {
            "status": "ready" if is_ready else "not_ready",
            "phase": "ticket_workflow",
            "checks": checks,
        }

    @app.get("/metadata", tags=["Tickets"], dependencies=[Depends(require_runtime)])
    def metadata() -> dict[str, Any]:
        return {
            "categories": list(app.state.predictor.classes),
            "teams": app.state.teams,
            "routing_policy": app.state.routing_policy,
        }

    @app.post("/predict", tags=["Predictions"], dependencies=[Depends(require_runtime)])
    def predict(request: TicketInput) -> dict[str, Any]:
        return triage(request.text)

    @app.post("/tickets", tags=["Tickets"], status_code=201)
    def create_ticket(
        request: TicketInput,
        background_tasks: BackgroundTasks,
        repo: TicketRepository = Depends(repository),
    ):
        ticket = repo.create(text=request.text, prediction=triage(request.text))
        if webhook_notifier.enabled:
            background_tasks.add_task(
                webhook_notifier.notify, ticket_event(ticket, "ticket.created")
            )
        return ticket

    @app.get("/tickets", tags=["Tickets"])
    def list_tickets(
        status: Literal["needs_review", "assigned", "resolved"] | None = None,
        team: str | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
        repo: TicketRepository = Depends(repository),
    ):
        if team is not None and team not in {item["id"] for item in app.state.teams}:
            raise _error(400, "invalid_team", "Team must be a configured demonstration team.")
        return repo.list(status=status, team=team, limit=limit, offset=offset)

    @app.get("/tickets/{ticket_id}", tags=["Tickets"])
    def get_ticket(ticket_id: str, repo: TicketRepository = Depends(repository)):
        return repo.get(ticket_id)

    @app.post("/tickets/{ticket_id}/reviews", tags=["Tickets"])
    def review_ticket(
        ticket_id: str,
        request: ReviewInput,
        background_tasks: BackgroundTasks,
        repo: TicketRepository = Depends(repository),
    ):
        if request.category not in app.state.intent_to_team:
            raise _error(400, "invalid_category", "Category must be supported by the loaded model.")
        ticket = repo.review(
            ticket_id,
            expected_version=request.expected_version,
            category=request.category,
            team=app.state.intent_to_team[request.category],
            status=request.status,
            note=request.note,
            reviewer_id=settings.reviewer_id,
        )
        if webhook_notifier.enabled:
            background_tasks.add_task(
                webhook_notifier.notify, ticket_event(ticket, "ticket.reviewed")
            )
        return ticket

    @app.get("/metrics", tags=["Tickets"])
    def metrics(repo: TicketRepository = Depends(repository)):
        return repo.metrics()

    return app


def run() -> None:
    settings = Settings()
    uvicorn.run(
        create_app(settings),
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
