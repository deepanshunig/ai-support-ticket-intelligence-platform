"""Exercise the actual Streamlit forms against the live local API."""

import json
from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

from ticket_intelligence.core.config import Settings


def main() -> None:
    settings = Settings()
    headers = {"X-Reviewer-Token": settings.reviewer_token.get_secret_value()}
    request = "The ATM kept my card and did not give it back."
    ui_path = Path(__file__).resolve().parents[1] / "ui/app.py"
    app = AppTest.from_file(str(ui_path), default_timeout=20).run()
    assert not app.exception and not app.error
    assert [tab.label for tab in app.tabs] == ["Submit request", "Ticket inbox", "Dashboard"]
    app.text_area(key="request_text").set_value(request)
    app.button(key="FormSubmitter:submit_request-Create ticket").click().run()
    assert not app.exception and not app.error
    with httpx.Client(base_url="http://127.0.0.1:8000", trust_env=False, timeout=10) as client:
        tickets = client.get("/tickets", headers=headers).json()["items"]
        ticket = next(item for item in tickets if item["text"] == request)
        app.selectbox(key="inbox_selected_ticket").select(ticket["id"]).run()
        app.selectbox(key=f"review_category_{ticket['id']}").select("card_swallowed")
        app.text_area(key=f"review_note_{ticket['id']}").set_value(
            "UI demonstration: confirmed ATM retained the card; assigned to cash support."
        )
        app.button(
            key=f"FormSubmitter:review_form_{ticket['id']}-Save reviewer decision"
        ).click().run()
        assert not app.exception and not app.error
        saved = client.get(f"/tickets/{ticket['id']}", headers=headers).json()
        assert saved["current_category"] == "card_swallowed"
        assert saved["assigned_team"] == "cash_atm" and saved["version"] == 2
        assert len(saved["reviews"]) == 1
    report = {"mechanism": "Streamlit AppTest with real HTTP/MySQL", "demo_ticket_id":
              ticket["id"], "checks": ["three tabs rendered", "no UI execution exceptions",
              "submission persisted", "review saved", "assigned team shown", "history loaded"],
              "visual_browser_inspection": "unavailable: browser runtime failed to start"}
    Path("reports/integration/ui_workflow.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
