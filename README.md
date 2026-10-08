# AI Support Ticket Intelligence Platform

A working banking-support prototype: submit an English request, inspect AI intent
suggestions, confirm or correct the category, and assign it to a demo support team.
MySQL preserves the original prediction, every reviewer decision and event history.

**Steps 1-6 verified.** The evaluated prototype combines a Logistic Regression
intent model, a frozen confidence policy, explicit demo-team mapping, MySQL
ticket history and a Streamlit reviewer workspace.

## Working features

- 77 banking intent categories with ranked alternatives and model identity.
- Five explicit, project-defined team mappings.
- Temperature-scaled selective assignment; uncertain, unknown-vocabulary and
  selected multi-team requests remain in human review.
- Authenticated ticket API, persistent inbox and reviewer corrections.
- Optional provider-neutral, HMAC-signed events for support-system automations;
  ticket text and reviewer notes are excluded, and delivery is disabled by default.
- Optimistic review versions reject stale changes without duplicating history.
- Streamlit submission, filtered inbox, original suggestions, history and activity charts.
- Reproducible source/splits, measured model comparison and meaningful tests.

See [project scope](docs/PROJECT_SCOPE.md) for users, limits and build order.
See the full [project documentation](docs/PROJECT_DOCUMENTATION.md) for the
architecture, model and routing policy, database schema, API, reviewer workspace,
evaluation, setup, and known limitations.

## Setup (Windows PowerShell)

Use Python 3.11 or 3.12. This workspace is tested with Python 3.12.14.

```powershell
& .\scripts\start-local.ps1
# When finished:
& .\scripts\stop-local.ps1
```

Run from the project root. This workspace is already configured. See
[local setup](docs/LOCAL_SETUP.md) for a fresh checkout, external MySQL,
local containers and verification commands. Virtual-environment activation is
optional.

The [dashboard](http://127.0.0.1:8501) and API run on localhost:

- `/docs`: interactive API documentation.
- `/health`: HTTP 200 confirms the API process can respond.
- `/ready`: HTTP 200 only when model, database and reviewer credential are ready;
  otherwise HTTP 503.
- `/predict`, `/tickets`, `/tickets/{id}/reviews`, `/metadata`, `/metrics`: authenticated
  workflow endpoints using the `X-Reviewer-Token` header.

## Configuration

Use `.env` or `TICKET_*` environment variables, which take precedence. Preserve the
generated local `.env`; use `.env.example` for new checkouts. Secrets are ignored by Git.

| Variable | Default | Purpose |
|---|---|---|
| `TICKET_APP_NAME` | AI Support Ticket Intelligence Platform | Service name |
| `TICKET_ENVIRONMENT` | development | development, test or production |
| `TICKET_LOG_LEVEL` | INFO | DEBUG, INFO, WARNING, ERROR or CRITICAL |
| `TICKET_API_HOST` | 127.0.0.1 | Local bind address |
| `TICKET_API_PORT` | 8000 | Port from 1 to 65535 |
| `TICKET_DATABASE_URL` | unset | MySQL SQLAlchemy URL (`mysql+pymysql://...`) |
| `TICKET_REVIEWER_TOKEN` | unset | Credential required for ticket endpoints |
| `TICKET_REVIEWER_ID` | demo-reviewer | Server-derived demo review identity |
| `TICKET_ARTIFACT_DIR` | artifacts/baselines | Verified model bundle |
| `TICKET_TEAM_MAPPING_PATH` | configs/team_mapping.json | Explicit intent mapping |
| `TICKET_POLICY_PATH` | artifacts/routing/policy.json | Frozen assignment/review policy |
| `TICKET_API_BASE_URL` | http://127.0.0.1:8000 | Dashboard's API address |
| `TICKET_WEBHOOK_URL` | unset | Optional HTTPS endpoint for minimized ticket events |
| `TICKET_WEBHOOK_SECRET` | unset | Optional 32+ character secret used to sign event payloads |

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

## Structure

```text
src/ticket_intelligence/
  api/          HTTP API
  core/         settings and infrastructure
  data/         dataset audits and splits
  training/     model experiments
  inference/    model loading and routing
  db/           persistence
  integrations/ optional signed support-system webhooks
ui/             reviewer interface
configs/        labels and team mappings
data/           local datasets, ignored by Git
artifacts/      local model bundles, ignored by Git
reports/        evaluation evidence
tests/          meaningful checks
docs/           project scope and build progress
```

## Model and routing results

TF-IDF Logistic Regression and Multinomial Naive Bayes are trained and evaluated
on separate fit/development partitions. A CPU-feasible frozen DistilBERT feature
comparison is also complete. Logistic Regression is retained for its stronger
development result, smaller artifact and lower measured local latency. Calibration
and routing-policy thresholds use their own partitions.

The frozen policy selected temperature 0.586, confidence threshold 0.70 and margin
0.20. On the reserved 3,080-example official test, intent accuracy was 87.6% and
macro-F1 was 87.6%. Automatic assignment covered 82.7% of those requests at 95.1%
accepted-intent accuracy and 98.9% demo-team accuracy. The authored 52-example
challenge set exposed weaker review recall on ambiguous requests (69.2%); the
threshold was not retuned against that holdout. These are prototype benchmark
results, not production guarantees. Read the full [final evaluation](reports/final/RESULTS.md),
[routing report](reports/routing/report.json), [measured baseline results](reports/baselines/RESULTS.md),
and [frozen DistilBERT comparison](reports/transformer/RESULTS.md).

## Reproduce the dataset

```powershell
.\.venv\Scripts\python.exe -m ticket_intelligence.data.download --revision 57ec275d8078af65b7731c2a98be812d844a6d6b
.\.venv\Scripts\python.exe -m ticket_intelligence.data.prepare --raw-dir data/raw/banking77 --output-dir data/processed/banking77 --report-dir reports/data
```

The original 10,003 training examples become 7,000 fit examples and 1,001 each
for development, calibration and routing-policy selection. All partitions cover
all 77 intents, and normalized duplicate groups stay in one partition. The
official 3,080-example test split is reserved for final evaluation. Seven text
groups overlap between the publisher's train and test splits; the audit reports
this without changing the official benchmark.

See [data provenance and limitations](docs/DATA_CARD.md),
[audit evidence](reports/data/audit.json) and the explicit, project-defined
[demo team mapping](configs/team_mapping.json).

## Reproduce model selection and final evaluation

```powershell
.\.venv\Scripts\python.exe -m ticket_intelligence.training.baseline
.\.venv\Scripts\python.exe scripts/verify_baseline.py
.\.venv\Scripts\python.exe scripts/export_dev_metrics.py
```

Logistic Regression leads development macro-F1 (86.4% versus 80.4% for Naive Bayes).
The routing policy must be generated once before the final evaluation:

```powershell
.\.venv\Scripts\python.exe -m ticket_intelligence.training.routing
.\.venv\Scripts\python.exe -m ticket_intelligence.training.routing --evaluate-frozen
```

The first command uses only calibration, policy and policy-development challenge
data. The second reads the reserved official test and held-out challenges. Do not
use those final results to adjust the reported policy. Read the
[model card](docs/MODEL_CARD.md) for tradeoffs and evaluation limits.
Follow [build progress](docs/BUILD_PROGRESS.md) for verified step boundaries and
[demonstration workflow](docs/DEMO_WALKTHROUGH.md) to use the local platform.
The local Compose deployment exposes the reviewer dashboard only on loopback; it
is not configured for public or cloud access.

## Domain limits

The platform assigns prototype support teams; it does not identify the real bank,
contact bank staff, change payments or issue refunds. Banking77 contains intent
requests, not verified resolutions, urgency or SLA ground truth. Demo reviewer
activity is separate from held-out model accuracy. Restrict dashboard access before
hosting this internal reviewer prototype publicly.
