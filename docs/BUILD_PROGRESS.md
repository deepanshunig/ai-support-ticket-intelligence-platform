# Build progress

Work follows the refined plan in order. Each step is verified before the next starts.

## Step 1 - complete (8 October 2026)

- Created an installable src-layout package and an isolated Python 3.12 environment.
- Added validated environment settings and an example local configuration.
- Added FastAPI service metadata, /health, /ready and interactive documentation.
- Initialized Git; verified ignored environment, caches, datasets and artifacts.
- Saved exact tested dependencies in requirements-lock.txt.
- Verification: 4 tests passed; Ruff passed; pip dependency check passed.
- Real HTTP startup probe passed and stopped its temporary server cleanly.
- Readiness deliberately returns 503 before model/database integration.

## Step 2 - complete (8 October 2026)

- Pinned Banking77 to source revision 57ec275d8078af65b7731c2a98be812d844a6d6b;
  retained its CC BY 4.0 licence and checksums.
- Audited 10,003 training and 3,080 official test examples across 77 intents.
- Created 7,000 fit examples and 1,001 each for development, calibration and policy;
  verified all labels and zero duplicate-group overlap across generated partitions.
- Disclosed seven publisher train/test overlap groups without modifying source data.
- Explicitly mapped all intents to five project-defined demo teams.
- Added 100 authored challenge examples (48 policy development, 52 held out),
  validated unique records, group counts and exact supported-intent references.
- Verification: 13 tests passed; Ruff and dependency checks passed; actual artifact
  integrity and team mapping checks passed. Generated the data audit and charts.

## Step 3 - complete (8 October 2026)

- Trained three Logistic Regression and three Naive Bayes candidates on fit only.
- Selected Logistic Regression C=10: development accuracy 86.71%, macro-F1 86.44%.
  Best Naive Bayes alpha=0.1: accuracy 82.22%, macro-F1 80.44%.
- Saved versioned pipelines, checksums, per-class/error/confusion reports and local
  latency measurements; added top-three, demo-team, NLL and Brier diagnostics.
- Verified model reload on authored requests and valid probability columns for all
  77 classes; explicitly flagged requests with no known vocabulary features.
- Verification: 34 tests passed; Ruff passed; training fixtures verify fit-only
  vocabulary, unchanged inputs and no calibration/policy/official-test CSV reads.
- Official test and calibration/policy experiments remain reserved for later steps.

## Step 4 - complete (8 October 2026)

- Project-local PostgreSQL 18.6 on localhost:55432, application role without
  superuser privilege, generated credentials in ignored local files.
- Ticket/prediction/correction/event persistence and optimistic review versions.
- Authenticated FastAPI submission, prediction, listing, review and workflow metrics.
- Streamlit submission/inbox/review/history/dashboard with real API activity.
- Verification: 66 tests and Ruff passed; real PostgreSQL simultaneous review
  conflict check, real HTTP workflow and Streamlit AppTest workflow passed.
- Restarted API and PostgreSQL and verified ticket states and full histories remain.
- Live API http://127.0.0.1:8000; dashboard http://127.0.0.1:8501.
- Browser visual inspection unavailable because browser automation runtime failed;
  dashboard rendering and forms were verified with AppTest instead.
- Automatic assignment remains disabled pending a verified routing policy.

## Step 5 - complete (8 October 2026)

- Pinned DistilBERT's pretrained source and ran a CPU frozen-feature experiment.
  Only the scaler/linear head was trained; no encoder fine-tuning is claimed.
- Frozen development accuracy 85.81%, macro-F1 85.45%; Logistic Regression remains
  selected with 86.71%/86.44%, smaller assets and lower measured local latency.
- Saved encoder/tokenizer/head assets, hashes, errors, timing and truncation evidence.
- Verification: 73 tests passed and Ruff passed, including seven offline tensor
  fixtures for pooling, frozen/eval mode, fit-only preprocessing and artifact reload.
- Calibration, policy and official test remain unused for model choice.

## Step 6 - complete (8 October 2026)

- Fit one global temperature on the 1,001-record calibration partition, then froze
  temperature 0.58584, confidence threshold 0.70 and margin 0.20 using only the
  policy partition and 48 policy-development challenge requests.
- The selected policy accepted 830/1,001 supported development requests (82.92%),
  with 95.90% intent accuracy and 98.43% demo-team accuracy. It met the prespecified
  per-group review-recall target on policy-development examples.
- Loaded the exact frozen model, policy and team-mapping hashes for one final run.
  Official-test accuracy 87.60%, macro-F1 87.62%, top-three accuracy 96.23%, and
  mapped demo-team accuracy 96.10% across 3,080 publisher test examples.
- Automatic assignment accepted 2,547/3,080 requests (82.69% coverage): 95.09%
  accepted-intent accuracy (Wilson 95% interval 94.18%-95.87%) and 98.90%
  accepted-team accuracy (98.42%-99.24%). Full-request accuracy remains 87.60%.
- On 52 held-out authored challenges, overall review recall was 86.54%: unrelated
  100%, nearby unsupported banking 92.31%, multi-intent 84.62%, ambiguous supported
  69.23%. Seven challenge requests were automatically assigned. The ambiguous slice
  missed its 75% design target; the policy was not retuned after holdout inspection.
- Authored challenge data does not estimate review precision, production prevalence
  or staffing load. The accepted-intent confidence interval also includes values
  below the 95% point target. These are evaluation limitations, not guarantees.
- Saved exact hashes, per-class metrics, errors, Wilson intervals and challenge
  results in `reports/final`; selection evidence is in `reports/routing`.
- Verification: 27 routing engine/policy tests passed; 76 tests passed when the two
  async HTTP test modules were excluded; Ruff passed. The sandbox's Windows asyncio
  event loop hangs even for a minimal FastAPI test app before project routes run,
  so those API tests could not complete in this session. Step 4's separately logged
  real HTTP/PostgreSQL workflow verification remains recorded above.

## Step 7 - in progress

- Added a locked-dependency GitHub Actions quality workflow for Python 3.11/3.12,
  lint and the full test suite.
- Added a non-root Python application image and Compose services for MySQL 8.4,
  API and dashboard. The UI is exposed only on host localhost; the API/database
  stay inside the Compose network. Model/policy files are mounted read-only.
- Added an ignored Compose secret file template and startup, persistence and scope
  documentation.
- Verification: 76 non-async tests passed and Ruff passed. Docker is not installed
  on this machine, so image build and `docker compose config` remain unverified.
  GitHub-hosted CI has not run because this repository is not connected/pushed.
- Next: validate the Compose configuration and image on a Docker-enabled host;
  validate API HTTP tests in a working asyncio runtime.

## Step 8 - pending

Documentation and demo evidence are in the repository. A hosted cloud release is
not configured; provider, account, access-control and credential decisions remain
open. Do not treat the localhost deployment as a public reviewer application.

## Database change - complete (8 October 2026)

- Replaced the application and Compose database configuration with MySQL; kept
  PostgreSQL support isolated to the one-time source migration utility.
- Added MySQL-native UTC timestamps, InnoDB tables, utf8mb4, and a transactional
  PostgreSQL-to-MySQL copy with row-count and content-digest verification.
- Updated local setup, dependency locks and documentation. Existing PostgreSQL
  records have not been modified.
- Migrated and checksum-verified 6 tickets, 10 ticket events and 4 reviews; the
  PostgreSQL source remains intact for rollback.
- The running application uses MySQL 9.7.0 with a database-scoped `ticket_app`
  account. API readiness and dashboard returned HTTP 200 after restart.
- Live MySQL verification passed persistence and the simultaneous-review check
  (one commit, one stale conflict); the API also retrieved migrated history.
- Verification: 15 MySQL/repository tests passed, Ruff and dependency checks passed.
  One demo ticket was added by the integration probe; it is included in the live
  total of 7 tickets.
