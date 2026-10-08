# Local ticket demonstration

The Streamlit workspace is an internal reviewer interface. It sends requests to
the API and displays records saved in MySQL. When the frozen routing policy
is active, requests that satisfy its checks can be assigned automatically; others
remain in review. A missing or inactive policy keeps all new requests in review.
The interface does not perform classification locally, change bank transactions,
or retrain from reviewer notes.

## Observed examples with the frozen policy

These outputs were produced by the exact policy in the final report. They are
examples of current behaviour, not hand-authored UI outcomes:

| Request | Top intent and team | Decision |
|---|---|---|
| “My debit card payment was declined at the grocery shop.” | `declined_card_payment` → Card Operations | Assigned, score 0.951 |
| “The ATM kept my card. Also, I forgot the passcode for my banking app.” | `passcode_forgotten` → Account Support | Needs review: multiple-team signals |
| “Can you suggest a recipe for pancakes?” | `fiat_currency_support` → Account Support | Needs review: below threshold, score 0.358 |
| “asdf qwerty zxcv” | `request_refund` → Card Operations | Needs review: no known vocabulary |

The unsupported recipe is still classified as one of the model's 77 banking
intents. The review action is the system's uncertainty safeguard; it does not prove
the request is out of domain or detect every unsupported request.

## Start the workspace

Run commands from the project root using the isolated environment. The locally
trained baseline artifacts must be present. To provision or start the repository's
local Windows MySQL instance, run:

```powershell
.\.venv\Scripts\python.exe scripts/setup_local_mysql.py
```

The helper saves local configuration in ignored files. If using an already
configured MySQL instance instead, start that database and set
`TICKET_DATABASE_URL` in the local environment. The API creates the MySQL schema
on startup.

Set the same nonempty `TICKET_REVIEWER_TOKEN` for the API and Streamlit server in
the local `.env` or process environment. Use a private local value; do not put it
in a screenshot, URL, demonstration request, repository or reviewer note.

The UI reads `.env` from the project root without overriding existing environment
variables. `TICKET_API_BASE_URL` defaults to `http://127.0.0.1:8000`; change that
value when the API uses a different address. The credential is attached to
server-side API calls and is never displayed or entered through a browser field.

In one terminal:

```powershell
.\.venv\Scripts\python.exe -m ticket_intelligence.api.main
```

In a second terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui/app.py --server.address 127.0.0.1
```

Open the local address printed by Streamlit, normally `http://127.0.0.1:8501`.
The green connection indicator appears when protected metadata can be retrieved.
If access is missing, the service is unavailable, or dependencies are not ready,
the workspace shows a setup message instead of fabricated ticket data.

## Demonstration sequence

1. Check the sidebar's routing status. It reflects the API's current policy mode,
   rather than assuming that automatic assignment is available. Use the following
   comparison after the frozen policy has been integrated and is active.
2. In **Submit request**, choose **Card payment** and create a ticket. Inspect the
   actual status: a sufficiently confident request that passes all checks starts
   **Assigned** with a team; otherwise it starts **Needs review** with no assigned
   team. The sample button does not force an outcome. Show its persistent ID,
   original suggestion and calibrated score. Expand the score details to compare
   the raw score before calibration and see the original policy version. Neither
   score guarantees a correct category or that the request is supported.
3. In **Ticket inbox**, open the saved ticket. Confirm or correct the category,
   choose **Assigned**, add a short reviewer note, and save. The API derives the
   assigned team from the confirmed category. Show the new version and activity
   history alongside the unchanged original prediction.
4. Record another decision with **Resolved** after explaining the hypothetical
   demonstration outcome. Show both decisions in chronological history; the
   earlier category, team, status and notes remain visible.
5. Create the **ATM issue** and **Account access** examples to demonstrate distinct
   supported categories. The actual model output should be inspected; example
   buttons do not force a particular prediction.
6. Create **Unrelated request** and compare its review outcome with the clear card
   request. The model still ranks existing banking labels because it is a
   77-class classifier. Inspect the displayed reason, such as a low score, close
   suggestions or unrecognized words. Leave an unrelated request in review; its
   top banking suggestion does not prove domain support. If the policy assigns
   it unexpectedly, record that as a routing failure instead of claiming the
   demonstration passed.
7. In **Dashboard**, show saved-ticket counts by status and assigned team, plus
   the number of reviewer decisions. These are workflow counts, not an accuracy
   score or evidence of successful customer resolutions.
8. Refresh the workspace and retrieve the ticket again to demonstrate database
   persistence. Filters and paging query the API; ticket records are not cached
   in the browser workspace.

With an inactive or missing policy, the same requests all start in review. The
workspace still shows the actual score type: old baseline tickets retain raw,
uncalibrated probabilities, while calibrated tickets show their calibrated score
and the optional original raw score separately. Existing ticket history is not
rewritten when the active policy changes.

## Requests with multiple issues

Try an independently authored example such as: "The ATM kept my card. Also, I
forgot the passcode for my banking app." Multiple-team signals can send a request
for review. This guard is a fixed heuristic, not a complete multi-intent detector:
short clauses, multiple issues for the same team, unrecognized wording and issues
beyond the inspected clauses can escape it. Inspect the actual outcome and avoid
claiming that every multi-intent request is detected.

Review reasons describe the stored decision at submission. A later reviewer may
assign or resolve that ticket, while its original review reason and suggestion
remain unchanged. Resolving a demonstration ticket records an internal workflow
decision rather than a real bank action.

## Concurrent review

Two reviewers can open the same ticket version. After one saves, the other's
stale decision receives HTTP 409. The interface reloads the latest record and
shows a notice asking the reviewer to inspect the updated category, history and
note before saving again. The API must not silently overwrite the first decision.

The form preserves the version originally displayed across its submission rerun;
fetching a fresh detail response during submission does not silently change the
version sent with the review.

## Display and evidence limits

- All example requests are fictional; avoid real personal or financial details.
- Categories come from the API's 77 supported labels; teams are project-defined.
- Times display in India Standard Time. The stored API timestamps remain intact.
- The original prediction and its score type stay distinct from the latest human
  decision. Raw and calibrated scores are labelled separately; a corrected
  category does not overwrite the original suggestion.
- Reviewer history records demonstration actions, not verified bank operations.
- Corrections do not automatically enter model training or an evaluation set.
- This local UI is not a public customer portal and adds no browser login flow.
- Screenshots and manual demonstrations show observed behaviour, not automated
  policy quality, official-test accuracy or production-scale performance.
- Policy eligibility targets are measured selection criteria, not per-ticket
  correctness guarantees. Unsupported-request and multi-intent handling remains
  limited by the small authored challenge set and the fixed heuristic guard.
