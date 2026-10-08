"""Workflow integration with fixture predictions and a real isolated SQL database."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.exc import SQLAlchemyError

from ticket_intelligence.api.main import create_app
from ticket_intelligence.core.config import Settings
from ticket_intelligence.db.session import make_engine
from ticket_intelligence.inference.predictor import ArtifactError
from ticket_intelligence.integrations.webhooks import WebhookNotifier

TOKEN = "fixture-reviewer-token"
AUTH = {"X-Reviewer-Token": TOKEN}


class FixturePredictor:
    """A fixed valid predictor contract; model integrity has its own fixture tests."""

    classes = ("card_issue", "login_issue")

    def predict(self, text):
        return {
            "predicted_category": "card_issue",
            "proposed_team": "card_team",
            "confidence": 0.8,
            "confidence_kind": "uncalibrated_model_probability",
            "candidates": [
                {"category": "card_issue", "probability": 0.8, "team": "card_team"},
                {"category": "login_issue", "probability": 0.2, "team": "account_team"},
            ],
            "model_version": "fixture-model-v1",
            "artifact_sha256": "a" * 64,
            "known_word_feature_count": 1,
            "no_known_vocabulary": False,
        }


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    mapping_path = tmp_path / "teams.json"
    mapping_path.write_text(
        json.dumps(
            {
                "teams": [
                    {"id": "card_team", "name": "Card Support"},
                    {"id": "account_team", "name": "Account Support"},
                ],
                "intent_to_team": {"card_issue": "card_team", "login_issue": "account_team"},
            }
        ),
        encoding="utf-8",
    )
    return Settings(
        _env_file=None,
        environment="test",
        database_url=None,
        artifact_dir=tmp_path / "missing-model",
        team_mapping_path=mapping_path,
        reviewer_token=SecretStr(TOKEN),
        reviewer_id="configured-reviewer",
        policy_path=None,
    )


@pytest.fixture
def app_bundle(settings):
    engine = make_engine("sqlite+pysqlite:///:memory:")
    app = create_app(settings, predictor=FixturePredictor(), engine=engine)
    yield app, engine
    engine.dispose()


@pytest.fixture
def client(app_bundle):
    with TestClient(app_bundle[0], headers=AUTH) as test_client:
        yield test_client


def _create(client, text="  My card is missing.  "):
    response = client.post("/tickets", json={"text": text})
    assert response.status_code == 201, response.text
    return response.json()


def _review(category="login_issue", version=1, status="assigned"):
    return {
        "expected_version": version,
        "category": category,
        "status": status,
        "note": "  Corrected and reviewed.  ",
    }


def test_end_to_end_review_preserves_original_prediction_and_audit(client):
    assert client.get("/ready").status_code == 200
    assert client.get("/metadata").json()["categories"] == list(FixturePredictor.classes)
    prediction = client.post("/predict", json={"text": "missing card"}).json()
    assert prediction["routing_action"] == "needs_review"
    assert prediction["assigned_team"] is None
    assert prediction["policy_version"] == "manual-review-v1"
    assert prediction["review_reason"] == "policy_not_verified"
    created = _create(client)
    original = created["prediction"]
    assert created["text"] == "My card is missing."
    assert created["status"] == "needs_review"
    assert created["assigned_team"] is None
    assert created["version"] == 1
    assert [event["type"] for event in created["events"]] == ["triaged"]
    ticket_id = created["id"]
    assigned = client.post(f"/tickets/{ticket_id}/reviews", json=_review())
    assert assigned.status_code == 200, assigned.text
    changed = assigned.json()
    assert changed["current_category"] == "login_issue"
    assert changed["assigned_team"] == "account_team"
    assert changed["version"] == 2
    assert changed["prediction"] == original
    review = changed["reviews"][0]
    assert review["reviewer_id"] == "configured-reviewer"
    assert review["note"] == "Corrected and reviewed."
    assert review["before"]["current_category"] == "card_issue"
    assert review["after"]["current_category"] == "login_issue"
    resolved = client.post(
        f"/tickets/{ticket_id}/reviews", json=_review(version=2, status="resolved")
    ).json()
    assert resolved["status"] == "resolved"
    assert resolved["version"] == 3
    assert resolved["prediction"] == original
    assert len(resolved["reviews"]) == 2
    assert len(resolved["events"]) == 3
    assert client.get(f"/tickets/{ticket_id}").json() == resolved
    metrics = client.get("/metrics").json()
    assert metrics["total_tickets"] == 1
    assert metrics["by_status"]["resolved"] == 1
    assert metrics["by_team"]["account_team"] == 1
    assert metrics["review_count"] == 2


def test_list_filters_and_pagination_are_consistent(client):
    first = _create(client, "first card request")
    _create(client, "second card request")
    _create(client, "third card request")
    client.post(f"/tickets/{first['id']}/reviews", json=_review())
    matching = client.get("/tickets", params={"status": "assigned", "team": "account_team"})
    assert matching.status_code == 200
    assert matching.json()["total"] == 1
    assert matching.json()["items"][0]["id"] == first["id"]
    assert "reviews" not in matching.json()["items"][0]
    first_page = client.get("/tickets", params={"limit": 1, "offset": 0}).json()
    second_page = client.get("/tickets", params={"limit": 1, "offset": 1}).json()
    assert first_page["total"] == second_page["total"] == 3
    assert first_page["items"][0]["id"] != second_page["items"][0]["id"]


def test_stale_review_does_not_append_or_mutate(client):
    ticket = _create(client)
    url = f"/tickets/{ticket['id']}/reviews"
    accepted = client.post(url, json=_review()).json()
    stale = client.post(url, json=_review(category="card_issue"))
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "version_conflict"
    assert client.get(f"/tickets/{ticket['id']}").json() == accepted
    assert client.get("/metrics").json()["review_count"] == 1


def test_ticket_and_review_events_are_sent_to_configured_webhook(app_bundle, settings):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    notifier = WebhookNotifier(
        "https://hooks.example.test/tickets",
        "fixture-webhook-secret-with-at-least-32-characters",
        transport=httpx.MockTransport(handler),
    )
    engine = app_bundle[1]
    app = create_app(
        settings,
        predictor=FixturePredictor(),
        engine=engine,
        webhook_notifier=notifier,
    )
    with TestClient(app, headers=AUTH) as integrated:
        ticket = _create(integrated, "Private request text must stay inside the app.")
        reviewed = integrated.post(
            f"/tickets/{ticket['id']}/reviews", json=_review()
        ).json()

    assert [request.headers["X-Ticket-Event-ID"] for request in requests] == [
        f"ticket.created:{ticket['id']}:1",
        f"ticket.reviewed:{ticket['id']}:2",
    ]
    assert all(b"Private request text" not in request.content for request in requests)
    assert all(b"Corrected and reviewed" not in request.content for request in requests)
    assert reviewed["version"] == 2


@pytest.mark.parametrize("token", [None, "incorrect-token"])
def test_all_sensitive_routes_require_authentication(app_bundle, token):
    headers = {} if token is None else {"X-Reviewer-Token": token}
    routes = [
        ("GET", "/metadata", None),
        ("GET", "/tickets", None),
        ("GET", "/tickets/missing", None),
        ("GET", "/metrics", None),
        ("POST", "/predict", {"text": "card request"}),
        ("POST", "/tickets", {"text": "card request"}),
        ("POST", "/tickets/missing/reviews", _review()),
    ]
    with TestClient(app_bundle[0]) as unauthenticated:
        for method, path, body in routes:
            response = unauthenticated.request(method, path, json=body, headers=headers)
            assert response.status_code == 401, (path, response.text)
            assert response.json()["detail"]["code"] == "unauthorized"
        assert unauthenticated.get("/health").status_code == 200


@pytest.mark.parametrize("value", ["", " \t\n ", "x" * 2001, None, 123])
def test_invalid_ticket_text_rejected_without_persistence(client, value):
    response = client.post("/tickets", json={"text": value})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_request"
    assert client.get("/metrics").json()["total_tickets"] == 0


def test_invalid_category_and_untrusted_review_fields_rejected(client):
    ticket = _create(client)
    url = f"/tickets/{ticket['id']}/reviews"
    assert client.post(url, json=_review(category="unknown_category")).status_code == 400
    assert client.post(url, json={**_review(), "reviewer_id": "spoofed"}).status_code == 400
    assert client.post(url, json={**_review(), "team": "card_team"}).status_code == 400
    assert client.get(f"/tickets/{ticket['id']}").json() == ticket


@pytest.mark.parametrize(
    "parameters",
    [
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
        {"status": "invalid"},
        {"team": "invalid"},
    ],
)
def test_invalid_inbox_parameters_return_400(client, parameters):
    assert client.get("/tickets", params=parameters).status_code == 400


def test_missing_ticket_returns_stable_404(client):
    response = client.get("/tickets/does-not-exist")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "ticket_not_found"
    assert client.post("/tickets/does-not-exist/reviews", json=_review()).status_code == 404


def test_missing_dependencies_keep_liveness_and_deny_workflow(settings):
    with TestClient(create_app(settings)) as unavailable:
        assert unavailable.get("/health").status_code == 200
        readiness = unavailable.get("/ready")
        assert readiness.status_code == 503
        assert readiness.json()["checks"]["model"] == "unavailable"
        assert readiness.json()["checks"]["database"] == "not_configured"
        assert (
            unavailable.post("/tickets", json={"text": "card request"}, headers=AUTH).status_code
            == 503
        )


def test_missing_auth_configuration_blocks_readiness_and_requests(settings, app_bundle):
    settings = settings.model_copy(update={"reviewer_token": None})
    app = create_app(settings, predictor=FixturePredictor(), engine=app_bundle[1])
    with TestClient(app) as unconfigured:
        assert unconfigured.get("/ready").json()["checks"]["reviewer_auth"] == "not_configured"
        assert unconfigured.get("/ready").status_code == 503
        assert unconfigured.get("/metadata", headers=AUTH).status_code == 503


def test_database_failure_updates_readiness_without_exposing_details(
    client, app_bundle, monkeypatch
):
    def unavailable_connection():
        raise SQLAlchemyError("password=private-database-secret")

    monkeypatch.setattr(app_bundle[1], "connect", unavailable_connection)
    readiness = client.get("/ready")
    assert readiness.status_code == 503
    assert readiness.json()["checks"]["database"] == "unavailable"
    assert "private-database-secret" not in readiness.text
    assert client.get("/health").status_code == 200


def test_prediction_failure_is_503_and_does_not_create_ticket(
    client, app_bundle, monkeypatch, caplog
):
    def unavailable_prediction(_text):
        raise ArtifactError("private-model-diagnostic")

    monkeypatch.setattr(app_bundle[0].state.predictor, "predict", unavailable_prediction)
    response = client.post("/tickets", json={"text": "sensitive request text"})
    assert response.status_code == 503
    assert "private-model-diagnostic" not in response.text
    assert "sensitive request text" not in caplog.text
    assert "private-model-diagnostic" not in caplog.text
    assert client.get("/metrics").json()["total_tickets"] == 0


def test_ticket_and_review_survive_engine_and_application_restart(settings, tmp_path):
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'durable.db').as_posix()}"
    first_engine = make_engine(database_url)
    with TestClient(
        create_app(settings, predictor=FixturePredictor(), engine=first_engine), headers=AUTH
    ) as first_client:
        ticket = _create(first_client)
        reviewed = first_client.post(f"/tickets/{ticket['id']}/reviews", json=_review()).json()
    first_engine.dispose()
    second_engine = make_engine(database_url)
    try:
        with TestClient(
            create_app(settings, predictor=FixturePredictor(), engine=second_engine), headers=AUTH
        ) as restarted:
            assert restarted.get(f"/tickets/{ticket['id']}").json() == reviewed
            assert restarted.get("/metrics").json()["review_count"] == 1
    finally:
        second_engine.dispose()
