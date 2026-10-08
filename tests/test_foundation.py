"""Check the service contract and configuration boundary before model integration."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ticket_intelligence.api.main import create_app
from ticket_intelligence.core.config import Settings


def test_liveness_does_not_falsely_claim_ticket_readiness(tmp_path):
    settings = Settings(_env_file=None, environment="test", artifact_dir=tmp_path / "missing")
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        readiness = client.get("/ready")
        assert readiness.status_code == 503
        assert readiness.json()["checks"]["model"] == "unavailable"
        assert readiness.json()["checks"]["database"] == "not_configured"
        assert "/tickets" in client.get("/openapi.json").json()["paths"]


def test_configuration_respects_environment_override(monkeypatch):
    monkeypatch.setenv("TICKET_APP_NAME", "Configuration smoke check")
    monkeypatch.setenv("TICKET_API_PORT", "8123")
    settings = Settings(_env_file=None)
    assert settings.app_name == "Configuration smoke check"
    assert settings.api_port == 8123


@pytest.mark.parametrize(
    ("name", "value"),
    [("TICKET_API_PORT", "70000"), ("TICKET_ENVIRONMENT", "unknown")],
)
def test_invalid_configuration_fails_early(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
