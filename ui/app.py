"""Internal Streamlit ticket workspace backed by the persistent ticket API.

Run from the project root with ``python -m streamlit run ui/app.py``.
The reviewer credential stays on this server and is never rendered or cached.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

API_BASE_URL = os.getenv("TICKET_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REVIEWER_TOKEN = os.getenv("TICKET_REVIEWER_TOKEN", "")
INDIA_TIME = timezone(timedelta(hours=5, minutes=30), name="IST")

EXAMPLES = {
    "Card payment": "My card payment was declined at the supermarket today.",
    "ATM issue": "The ATM kept my card and did not return it.",
    "Account access": "I forgot the passcode for my banking app.",
    "Unrelated request": "Suggest a simple dinner I can cook with chickpeas and spinach.",
}
STATUS_NAMES = {
    "needs_review": "Needs review",
    "assigned": "Assigned",
    "resolved": "Resolved",
}
REVIEW_REASON_TEXT = {
    "policy_not_verified": "Automatic assignment was not enabled.",
    "policy_inactive": "The policy kept this request in the review queue.",
    "no_known_vocabulary": "The model could not recognize the words in this request.",
    "multiple_team_signals": "The request may contain issues for more than one team.",
    "below_threshold": "The model score was below the assignment threshold.",
    "below_margin": "The leading suggestions were too close to choose one.",
}


class WorkspaceError(Exception):
    """A safe, user-facing API failure without credentials or transport details."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def api_request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use a short-lived server-side client; do not cache tickets or credentials."""
    try:
        with httpx.Client(
            timeout=httpx.Timeout(20.0, connect=5.0),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = client.request(
                method,
                f"{API_BASE_URL}{path}",
                headers={"X-Reviewer-Token": REVIEWER_TOKEN},
                json=payload,
                params=params,
            )
    except (httpx.HTTPError, ValueError):
        raise WorkspaceError(
            "The ticket service cannot be reached. Start the local service and try again."
        ) from None

    if response.is_error:
        messages = {
            400: "Check the request fields and try again.",
            401: "Reviewer access was rejected. Check the server's reviewer configuration.",
            404: "This ticket could not be found. Refresh the inbox to see current tickets.",
            409: "This ticket changed while you were reviewing it. Reload before saving again.",
            422: "Check the request fields and try again.",
            503: (
                "The ticket service is not ready. Start the API and database, check the "
                "model and reviewer configuration, then refresh this workspace."
            ),
        }
        raise WorkspaceError(
            messages.get(
                response.status_code, "The ticket service could not complete this action."
            ),
            response.status_code,
        )
    try:
        result = response.json()
    except ValueError:
        raise WorkspaceError("The ticket service returned an unreadable response.") from None
    if not isinstance(result, dict):
        raise WorkspaceError("The ticket service returned an unexpected response.")
    return result


def label(value: str | None) -> str:
    if not value:
        return "Not set"
    return value.replace("_", " ").replace("?", "").capitalize()


def team_name(value: str | None, teams: dict[str, str]) -> str:
    if not value:
        return "Unassigned"
    return teams.get(value, label(value))


def display_time(value: str | None) -> str:
    if not value:
        return "Time not provided"
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return timestamp.astimezone(INDIA_TIME).strftime("%d %b %Y, %H:%M IST")
    except (ValueError, TypeError):
        return "Time unavailable"


def score(value: Any) -> str:
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return f"{value:.1%}"
    return "Not available"


def populate_example(text: str) -> None:
    st.session_state["request_text"] = text


def reset_inbox_page() -> None:
    st.session_state["inbox_page"] = 1


def show_original_prediction(ticket: dict[str, Any], teams: dict[str, str]) -> None:
    prediction = ticket.get("prediction", {})
    confidence_kind = prediction.get("confidence_kind", "uncalibrated_model_probability")
    is_calibrated = confidence_kind == "temperature_scaled_model_probability"
    score_label = "Calibrated model score" if is_calibrated else "Raw model score"
    confidence = prediction.get("confidence", ticket.get("confidence"))
    st.subheader("Original model suggestion")
    category_col, score_col, team_col = st.columns([2, 1, 2])
    with category_col:
        st.caption("Suggested category")
        st.write(label(prediction.get("predicted_category")))
    with score_col:
        st.caption(score_label)
        st.write(score(confidence))
    with team_col:
        st.caption("Proposed team")
        st.write(team_name(prediction.get("proposed_team"), teams))
    st.caption(
        "Scores guide routing; they do not guarantee a correct category or a supported request."
    )
    reason = prediction.get("review_reason")
    action = prediction.get("routing_action", "needs_review")
    if action == "assigned":
        original_team = prediction.get("assigned_team") or prediction.get("proposed_team")
        st.success(
            f"Assigned at submission to {team_name(original_team, teams)} under the routing policy."
        )
    else:
        explanation = REVIEW_REASON_TEXT.get(
            reason or "policy_not_verified", "A reviewer needs to clarify this request."
        )
        st.info(f"At submission, this request was sent for review. {explanation}")
    with st.expander("Other suggested categories and score details"):
        candidates = prediction.get("candidates", [])
        if candidates:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Category": label(candidate.get("category")),
                            score_label: score(candidate.get("probability")),
                            "Proposed team": team_name(candidate.get("team"), teams),
                        }
                        for candidate in candidates
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        st.caption(f"Score type: {confidence_kind}")
        if is_calibrated:
            st.caption(f"Calibrated model score: {score(confidence)}")
            if prediction.get("raw_confidence") is not None:
                st.caption(
                    f"Raw model score before calibration: {score(prediction['raw_confidence'])}"
                )
        else:
            st.caption("The displayed probabilities have not been calibrated.")
        model_version = ticket.get("model_version", prediction.get("model_version", "unknown"))
        st.caption(f"Model: {model_version}")
        st.caption(f"Routing policy: {prediction.get('policy_version', 'manual-review-v1')}")
        st.caption(f"Original review reason: {reason or 'none'}")


def show_history(ticket: dict[str, Any], teams: dict[str, str]) -> None:
    st.subheader("Activity history")
    events = ticket.get("events", [])
    if not events:
        st.caption("No activity is available for this ticket yet.")
    for event in events:
        heading = f"{display_time(event.get('created_at'))} · {label(event.get('type'))}"
        with st.expander(heading):
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                before = payload.get("before")
                after = payload.get("after")
                if isinstance(before, dict) and isinstance(after, dict):
                    show_decision_change(before, after, teams)
                note = payload.get("note")
                if isinstance(note, str) and note:
                    st.caption("Reviewer note")
                    st.text(note)
            with st.expander("Stored event"):
                st.json(event)

    reviews = ticket.get("reviews", [])
    if reviews:
        st.subheader("Reviewer decisions")
        for review in reviews:
            with st.expander(display_time(review.get("created_at"))):
                show_decision_change(review.get("before", {}), review.get("after", {}), teams)
                st.caption("Reviewer note")
                st.text(str(review.get("note", "")))


def show_decision_change(
    before: dict[str, Any], after: dict[str, Any], teams: dict[str, str]
) -> None:
    fields = (
        ("Category", "category"),
        ("Team", "assigned_team"),
        ("Status", "status"),
        ("Version", "version"),
    )
    rows = []
    for heading, field in fields:
        previous = before.get(field)
        current = after.get(field)
        if field == "category":
            previous = before.get("current_category", previous)
            current = after.get("current_category", current)
        if field == "assigned_team":
            previous, current = team_name(previous, teams), team_name(current, teams)
        elif field in {"category", "status"}:
            previous, current = label(previous), label(current)
        rows.append({"Field": heading, "Before": str(previous), "After": str(current)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def show_review_form(ticket: dict[str, Any], categories: list[str], teams: dict[str, str]) -> None:
    ticket_id = str(ticket["id"])
    version_key = f"review_displayed_version_{ticket_id}"
    st.subheader("Reviewer decision")
    st.caption(
        "Confirm or correct the category. The confirmed category determines the assigned team. "
        "Leave unclear, unrelated or multiple-issue requests in review until they are clarified."
    )
    category = ticket.get("current_category") or ticket.get("prediction", {}).get(
        "predicted_category"
    )
    category_index = categories.index(category) if category in categories else 0
    with st.form(f"review_form_{ticket_id}"):
        confirmed_category = st.selectbox(
            "Confirmed category",
            options=categories,
            index=category_index,
            format_func=label,
            key=f"review_category_{ticket_id}",
        )
        next_status = st.radio(
            "Set status",
            options=["assigned", "resolved"],
            format_func=lambda value: STATUS_NAMES[value],
            horizontal=True,
            key=f"review_status_{ticket_id}",
        )
        note = st.text_area(
            "Reviewer note",
            placeholder="Explain what you confirmed or corrected and the next action.",
            max_chars=1000,
            height=110,
            key=f"review_note_{ticket_id}",
        )
        st.caption(f"Ticket version {ticket['version']} · Original model suggestion is preserved.")
        submitted = st.form_submit_button("Save reviewer decision", type="primary")
    if not submitted:
        # A form submission reruns the script. Preserve the previously displayed
        # version rather than silently adopting a fresh version fetched that run.
        st.session_state[version_key] = int(ticket["version"])
        return
    if not note.strip():
        st.warning("Add a short reviewer note before saving.")
        return
    expected_version = st.session_state.get(version_key)
    if expected_version is None:
        st.warning("Refresh the ticket before saving a reviewer decision.")
        return
    try:
        api_request(
            "POST",
            f"/tickets/{ticket_id}/reviews",
            payload={
                "expected_version": expected_version,
                "category": confirmed_category,
                "status": next_status,
                "note": note.strip(),
            },
        )
    except WorkspaceError as error:
        if error.status_code == 409:
            st.session_state["workspace_notice"] = (
                "warning",
                "This ticket changed while you were reviewing it. The latest version has "
                "been loaded. Check the category, history and note before saving again.",
            )
            st.session_state.pop(version_key, None)
            st.rerun()
        st.error(str(error))
        return
    st.session_state["workspace_notice"] = (
        "success",
        "Reviewer decision saved. The ticket and activity charts are up to date.",
    )
    st.session_state.pop(version_key, None)
    st.rerun()


def show_ticket(
    ticket: dict[str, Any], categories: list[str], teams: dict[str, str], *, allow_review: bool
) -> None:
    st.caption("Persistent ticket ID")
    st.code(str(ticket["id"]), language=None)
    status_col, team_col, version_col = st.columns([1, 2, 1])
    status_col.metric("Status", STATUS_NAMES.get(ticket.get("status"), label(ticket.get("status"))))
    team_col.metric("Assigned team", team_name(ticket.get("assigned_team"), teams))
    version_col.metric("Version", str(ticket.get("version", "")))
    st.caption(f"Created {display_time(ticket.get('created_at'))}")
    st.caption("Customer request")
    st.text(str(ticket.get("text", "")))
    st.caption("Current category")
    st.write(label(ticket.get("current_category")))
    if ticket.get("status") == "needs_review":
        st.caption("Awaiting reviewer confirmation.")
    st.divider()
    show_original_prediction(ticket, teams)
    if allow_review:
        st.divider()
        show_review_form(ticket, categories, teams)
    st.divider()
    show_history(ticket, teams)


def submit_tab(categories: list[str], teams: dict[str, str]) -> None:
    st.subheader("Start with the customer's request")
    st.caption("Choose a sample or write an English banking-support request.")
    for column, (name, text) in zip(st.columns(4), EXAMPLES.items(), strict=True):
        column.button(name, on_click=populate_example, args=(text,), width="stretch")
    with st.form("submit_request"):
        text = st.text_area(
            "Customer request",
            placeholder=(
                "Describe what happened and which card, payment or account action was involved."
            ),
            max_chars=2000,
            height=150,
            key="request_text",
        )
        st.caption(
            "Up to 2,000 characters. Use demonstration text without personal account details."
        )
        submitted = st.form_submit_button("Create ticket", type="primary")
    created = None
    if submitted:
        if not text.strip():
            st.warning("Write a customer request before creating a ticket.")
        else:
            try:
                with st.spinner("Saving the request and its model suggestion..."):
                    created = api_request("POST", "/tickets", payload={"text": text.strip()})
                st.session_state["latest_created_ticket_id"] = created["id"]
                if created.get("status") == "assigned":
                    st.success(
                        "Ticket saved and assigned to "
                        f"{team_name(created.get('assigned_team'), teams)}."
                    )
                elif created.get("status") == "needs_review":
                    st.success("Ticket saved. It is waiting for a reviewer decision.")
                else:
                    st.success("Ticket saved.")
            except WorkspaceError as error:
                st.error(str(error))
    ticket_id = st.session_state.get("latest_created_ticket_id")
    if ticket_id:
        try:
            ticket = created or api_request("GET", f"/tickets/{ticket_id}")
            st.divider()
            st.subheader("Latest submitted ticket")
            show_ticket(ticket, categories, teams, allow_review=False)
            st.caption("Open Ticket inbox to confirm the suggestion or record a correction.")
        except WorkspaceError as error:
            st.error(str(error))


def inbox_tab(categories: list[str], teams: dict[str, str]) -> None:
    st.subheader("Review the queue")
    status_col, team_col, page_col = st.columns([2, 2, 1])
    with status_col:
        status_filter = st.selectbox(
            "Status",
            options=["all", *STATUS_NAMES],
            format_func=lambda value: "All statuses" if value == "all" else STATUS_NAMES[value],
            key="inbox_status",
            on_change=reset_inbox_page,
        )
    with team_col:
        team_filter = st.selectbox(
            "Assigned team",
            options=["all", *teams],
            format_func=lambda value: "All teams" if value == "all" else teams[value],
            key="inbox_team",
            on_change=reset_inbox_page,
        )
    with page_col:
        page = st.number_input("Page", min_value=1, value=1, step=1, key="inbox_page")
    params: dict[str, Any] = {"limit": 25, "offset": (int(page) - 1) * 25}
    if status_filter != "all":
        params["status"] = status_filter
    if team_filter != "all":
        params["team"] = team_filter
    try:
        result = api_request("GET", "/tickets", params=params)
    except WorkspaceError as error:
        st.error(str(error))
        return
    tickets = result.get("items", [])
    total = int(result.get("total", 0))
    st.caption(f"{total} matching tickets · 25 per page")
    if not tickets:
        st.info(
            "No tickets on this page. Submit a request, adjust the filters or return to page 1."
        )
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Ticket": str(ticket["id"])[:8],
                    "Request": str(ticket.get("text", ""))[:90],
                    "Status": STATUS_NAMES.get(ticket.get("status"), label(ticket.get("status"))),
                    "Current category": label(ticket.get("current_category")),
                    "Assigned team": team_name(ticket.get("assigned_team"), teams),
                    "Updated": display_time(ticket.get("updated_at")),
                }
                for ticket in tickets
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    by_id = {str(ticket["id"]): ticket for ticket in tickets}
    ticket_id = st.selectbox(
        "Open a ticket",
        options=list(by_id),
        format_func=lambda value: f"{value[:8]} · {str(by_id[value].get('text', ''))[:60]}",
        key="inbox_selected_ticket",
    )
    if st.button("Reload ticket", key="reload_selected_ticket"):
        st.rerun()
    try:
        detail = api_request("GET", f"/tickets/{ticket_id}")
    except WorkspaceError as error:
        st.error(str(error))
        return
    st.divider()
    show_ticket(detail, categories, teams, allow_review=True)


def dashboard_tab(teams: dict[str, str]) -> None:
    st.subheader("Workspace activity")
    st.caption("Live counts of saved tickets and reviewer actions.")
    try:
        metrics = api_request("GET", "/metrics")
    except WorkspaceError as error:
        st.error(str(error))
        return
    by_status = metrics.get("by_status", {})
    columns = st.columns(5)
    columns[0].metric("Total tickets", int(metrics.get("total_tickets", 0)))
    columns[1].metric("Needs review", int(by_status.get("needs_review", 0)))
    columns[2].metric("Assigned", int(by_status.get("assigned", 0)))
    columns[3].metric("Resolved", int(by_status.get("resolved", 0)))
    columns[4].metric("Reviewer decisions", int(metrics.get("review_count", 0)))
    status_col, team_col = st.columns(2)
    with status_col:
        st.subheader("Tickets by status")
        status_data = pd.DataFrame(
            {"Tickets": [int(by_status.get(status, 0)) for status in STATUS_NAMES]},
            index=pd.Index(list(STATUS_NAMES.values()), name="Status"),
        )
        st.bar_chart(status_data, color="#178b87")
    with team_col:
        st.subheader("Tickets by assigned team")
        by_team = metrics.get("by_team", {})
        team_data = pd.DataFrame(
            {"Tickets": [int(by_team.get(team, 0)) for team in [*teams, "unassigned"]]},
            index=pd.Index([*teams.values(), "Unassigned"], name="Team"),
        )
        st.bar_chart(team_data, color="#224660")
    st.info(
        "These charts show workflow activity. Reviewer corrections are not an independent "
        "accuracy measurement and do not automatically retrain the model."
    )


def main() -> None:
    st.set_page_config(
        page_title="Banking Support | Review workspace", page_icon="🧭", layout="wide"
    )
    style_path = Path(__file__).with_name("styles.css")
    if style_path.is_file():
        st.markdown(
            f"<style>{style_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True
        )
    st.markdown(
        '<div class="workspace-label">BANKING SUPPORT · INTERNAL DEMO</div>',
        unsafe_allow_html=True,
    )
    st.title("Support review workspace")
    st.write("Capture a request, inspect its suggestion, and keep a clear record of the review.")
    with st.sidebar:
        st.markdown("### Banking support")
        st.caption("A calm queue. A clear next action.")
        st.divider()
        st.markdown("**Reviewer workspace**")
        st.caption("English requests · 77 supported categories")
        policy_status = st.empty()
        policy_status.caption("Routing status appears when the service connects.")
        st.divider()
        if st.button("Refresh workspace", width="stretch"):
            st.rerun()
        st.caption("Times are shown in India Standard Time.")
        st.caption("Prototype teams and demonstration requests. No bank transactions are changed.")
    notice = st.session_state.pop("workspace_notice", None)
    if notice:
        kind, message = notice
        if kind == "warning":
            st.warning(message)
        else:
            st.success(message)
    if not REVIEWER_TOKEN:
        st.warning(
            "Reviewer access is not configured on this server. Set TICKET_REVIEWER_TOKEN "
            "in the local environment, then restart the workspace."
        )
        st.stop()
    try:
        metadata = api_request("GET", "/metadata")
    except WorkspaceError as error:
        st.error(str(error))
        st.caption("Setup and demonstration steps are in docs/DEMO_WALKTHROUGH.md.")
        st.stop()
    categories = metadata.get("categories", [])
    teams = {team["id"]: team["name"] for team in metadata.get("teams", [])}
    if not categories or not teams:
        st.error("The service has not supplied its categories and teams. Check setup and refresh.")
        st.stop()
    routing_policy = metadata.get("routing_policy") or {}
    with policy_status.container():
        if routing_policy.get("mode") == "automatic_routing" and routing_policy.get("active"):
            st.caption(
                "Automatic assignment is enabled. Requests that pass the policy checks are "
                "assigned; others need review."
            )
        else:
            st.caption("All new requests need human review. Automatic assignment is not enabled.")
    st.markdown(
        '<div class="service-ready">● Connected to the ticket service</div>',
        unsafe_allow_html=True,
    )
    submit, inbox, dashboard = st.tabs(["Submit request", "Ticket inbox", "Dashboard"])
    with submit:
        submit_tab(categories, teams)
    with inbox:
        inbox_tab(categories, teams)
    with dashboard:
        dashboard_tab(teams)


if __name__ == "__main__":
    main()
