"""Privacy and authentication checks for optional outbound ticket webhooks."""

from __future__ import annotations

import hashlib
import hmac
import json

import httpx
import pytest

from ticket_intelligence.integrations.webhooks import WebhookNotifier, ticket_event

SECRET = "webhook-test-secret-with-at-least-32-characters"


def _ticket() -> dict:
    return {
        "id": "ticket-123",
        "text": "Sensitive customer request that must not leave the app.",
        "updated_at": "2026-10-08T12:00:00Z",
        "status": "assigned",
        "current_category": "declined_card_payment",
        "assigned_team": "card_operations",
        "confidence": 0.92,
        "model_version": "model-v1",
        "version": 2,
        "prediction": {"private": "original output"},
        "reviews": [{"note": "private reviewer note"}],
    }


def test_ticket_event_is_stable_and_excludes_private_free_text():
    event = ticket_event(_ticket(), "ticket.reviewed")
    encoded = json.dumps(event)
    assert event["event_id"] == "ticket.reviewed:ticket-123:2"
    assert event["ticket"]["category"] == "declined_card_payment"
    assert "Sensitive customer request" not in encoded
    assert "private reviewer note" not in encoded
    assert "original output" not in encoded


def test_webhook_delivery_signs_minimal_json_payload():
    received = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(204)

    notifier = WebhookNotifier(
        "https://hooks.example.test/tickets",
        SECRET,
        transport=httpx.MockTransport(handler),
    )
    event = ticket_event(_ticket(), "ticket.created")

    assert notifier.notify(event)
    assert len(received) == 1
    request = received[0]
    raw_body = request.content
    expected = hmac.new(SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
    assert request.headers["X-Ticket-Signature"] == f"sha256={expected}"
    assert request.headers["X-Ticket-Event-ID"] == event["event_id"]
    assert json.loads(raw_body) == event
    assert b"Sensitive customer request" not in raw_body


def test_webhook_is_disabled_without_configuration():
    notifier = WebhookNotifier(None, None)
    assert not notifier.enabled
    assert not notifier.notify(ticket_event(_ticket(), "ticket.created"))


@pytest.mark.parametrize(
    ("url", "secret", "environment"),
    [
        ("https://hooks.example.test/tickets", None, "development"),
        (None, SECRET, "development"),
        ("http://hooks.example.test/tickets", SECRET, "development"),
        ("http://127.0.0.1/tickets", SECRET, "production"),
        ("https://user:pass@hooks.example.test/tickets", SECRET, "production"),
        ("https://hooks.example.test/tickets?token=secret", SECRET, "production"),
    ],
)
def test_unsafe_or_incomplete_webhook_configuration_is_rejected(url, secret, environment):
    with pytest.raises(ValueError):
        WebhookNotifier(url, secret, environment=environment)


def test_local_http_webhook_is_allowed_only_for_development():
    notifier = WebhookNotifier("http://127.0.0.1:9000/tickets", SECRET)
    assert notifier.enabled


def test_webhook_failure_is_nonfatal_and_does_not_log_response_body(caplog):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="private downstream error body")

    notifier = WebhookNotifier(
        "https://hooks.example.test/tickets",
        SECRET,
        transport=httpx.MockTransport(handler),
    )

    assert not notifier.notify(ticket_event(_ticket(), "ticket.created"))
    assert "private downstream error body" not in caplog.text
