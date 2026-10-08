"""Privacy-limited, signed webhook delivery for support-system automations."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)


def ticket_event(ticket: dict[str, Any], event_type: str) -> dict[str, Any]:
    """Build a minimal event that excludes request text, reviewer notes, and auth data."""
    if event_type not in {"ticket.created", "ticket.reviewed"}:
        raise ValueError("Unsupported ticket webhook event type.")
    version = int(ticket["version"])
    ticket_id = str(ticket["id"])
    return {
        "event_id": f"{event_type}:{ticket_id}:{version}",
        "event_type": event_type,
        "occurred_at": ticket["updated_at"],
        "ticket": {
            "id": ticket_id,
            "version": version,
            "status": ticket["status"],
            "category": ticket["current_category"],
            "assigned_team": ticket["assigned_team"],
            "confidence": ticket["confidence"],
            "model_version": ticket["model_version"],
        },
    }


class WebhookNotifier:
    """Best-effort JSON delivery; disabled unless a URL and signing secret are configured."""

    def __init__(
        self,
        url: str | None,
        secret: str | None,
        *,
        environment: str = "development",
        timeout_seconds: float = 3.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.url = (url or "").strip() or None
        self.secret = (secret or "").encode("utf-8") or None
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        if not self.url and not self.secret:
            return
        if not self.url or not self.secret:
            raise ValueError("Webhook URL and signing secret must be configured together.")
        if len(self.secret) < 32:
            raise ValueError("Webhook signing secret must contain at least 32 characters.")
        parsed = urlsplit(self.url)
        local_http = parsed.scheme == "http" and parsed.hostname in {
            "127.0.0.1", "localhost", "::1"
        }
        if parsed.scheme != "https" and not (environment != "production" and local_http):
            raise ValueError("Webhook URL must use HTTPS (HTTP is allowed only for local dev).")
        if not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
            raise ValueError(
                "Webhook URL must have a host and no embedded credentials or fragment."
            )
        if parsed.query:
            raise ValueError("Webhook URL must not contain query parameters.")
    @property
    def enabled(self) -> bool:
        return bool(self.url and self.secret)

    def notify(self, event: dict[str, Any]) -> bool:
        """Send one sanitized event; never include response bodies or secrets in logs."""
        if not self.enabled:
            return False
        body = json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        signature = hmac.new(self.secret, body, hashlib.sha256).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-Ticket-Event-ID": str(event["event_id"]),
            "X-Ticket-Signature": f"sha256={signature}",
        }
        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
            ) as client:
                response = client.post(self.url, content=body, headers=headers)
                response.raise_for_status()
        except httpx.HTTPError as error:
            logger.warning(
                "Ticket webhook delivery failed event_id=%s error_type=%s",
                event.get("event_id"),
                type(error).__name__,
            )
            return False
        logger.info("Ticket webhook delivered event_id=%s", event.get("event_id"))
        return True
