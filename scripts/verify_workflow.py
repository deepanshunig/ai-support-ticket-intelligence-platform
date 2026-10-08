"""Check a live local ticket workflow with authored demonstration requests."""

import json
from pathlib import Path

import httpx

from ticket_intelligence.core.config import Settings


def main() -> None:
    settings = Settings()
    headers = {"X-Reviewer-Token": settings.reviewer_token.get_secret_value()}
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=15, trust_env=False) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/tickets").status_code == 401
        assert client.post("/tickets", headers=headers, json={"text": "  "}).status_code == 400
        response = client.post("/tickets", headers=headers, json={
            "text": "I forgot the passcode for my banking app."
        })
        assert response.status_code == 201
        ticket = response.json()
        assert ticket["status"] == "needs_review"
        original = ticket["prediction"]
        url = f"/tickets/{ticket['id']}"
        for version, status in [(1, "assigned"), (2, "resolved")]:
            response = client.post(url + "/reviews", headers=headers, json={
                "expected_version": version, "category": "passcode_forgotten",
                "status": status, "note": f"Demonstration only: reviewer marked {status}.",
            })
            assert response.status_code == 200
            assert response.json()["assigned_team"] == "account_support"
        stale = client.post(url + "/reviews", headers=headers, json={
            "expected_version": 1, "category": "passcode_forgotten",
            "status": "assigned", "note": "This stale review must be rejected.",
        })
        assert stale.status_code == 409
        detail = client.get(url, headers=headers).json()
        assert detail["prediction"] == original
        assert detail["version"] == 3 and len(detail["events"]) == 3
        assert len(detail["reviews"]) == 2
        assert all(r["reviewer_id"] == settings.reviewer_id for r in detail["reviews"])
        assert detail["status"] == "resolved"
    report = {"demo_ticket_id": ticket["id"], "checks": [
        "readiness", "authentication", "input validation", "prediction and persistence",
        "assigned team derived from confirmed category", "resolution", "stale review rejected",
        "original prediction retained", "server-derived reviewer identity", "event history",
    ]}
    directory = Path("reports/integration")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "http_workflow.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
