# Run the local platform

The current workspace already has an isolated Python environment, downloaded
Banking77 data, trained baseline models, and a local MySQL server. The application
uses MySQL; the original PostgreSQL copy remains untouched as a rollback source.
Run commands from the project root.

## Start and stop

```powershell
& .\scripts\start-local.ps1
& .\scripts\stop-local.ps1
```

Start refuses occupied API/dashboard ports rather than replacing other processes.
Stop verifies saved process identities and stops only the project's API and
dashboard. Database servers remain running. Services run in hidden windows.

- Dashboard: http://127.0.0.1:8501
- API documentation: http://127.0.0.1:8000/docs
- Readiness: http://127.0.0.1:8000/ready

The dashboard sends its reviewer credential from the server to the API. The token
and database credentials are in ignored `.env`; they are not shown in the UI.
After migration, submitted examples and reviewer actions are stored in MySQL.
The PostgreSQL data directory remains available as a rollback source.

## Fresh Windows checkout

Use Python 3.11 or 3.12. Create `.venv`, install the project, then create `.env`
only if it does not already exist:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe -m ticket_intelligence.data.download --revision 57ec275d8078af65b7731c2a98be812d844a6d6b
.\.venv\Scripts\python.exe -m ticket_intelligence.data.prepare
.\.venv\Scripts\python.exe -m ticket_intelligence.training.baseline
.\.venv\Scripts\python.exe -m ticket_intelligence.training.routing
.\.venv\Scripts\python.exe -m ticket_intelligence.training.routing --evaluate-frozen
& .\scripts\start-local.ps1
```

The routing commands calibrate and freeze the policy before reading the final
test partitions. Run the final evaluation once for a new model/policy bundle.

The local setup helper expects MySQL listening on `127.0.0.1:3306`. On first setup
it asks for the administrator password through a hidden terminal prompt, creates
the `ticket_intelligence` database and a database-scoped `ticket_app` account, and
saves the application URL in ignored `.env`. It never prints either password.

The one-time migration for this workspace copied all three application tables in
one transaction and verified row counts and SHA-256 content digests before
switching the app URL. For another existing PostgreSQL source, configure its URL
in `.env`, stop the app with `scripts/stop-local.ps1`, and run the MySQL setup
helper. It refuses to migrate while the API is reachable. PostgreSQL remains
untouched as a rollback source. To migrate a PostgreSQL source in a fresh
environment, install the optional driver with
`python -m pip install 'psycopg[binary]>=3.2,<4'` before starting.

For an existing MySQL database, set `TICKET_DATABASE_URL` to a SQLAlchemy
`mysql+pymysql` URL with permission to create the schema. The initial schema is
created automatically; later schema changes require a migration.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/verify_mysql.py
.\.venv\Scripts\python.exe scripts/verify_workflow.py
.\.venv\Scripts\python.exe scripts/verify_ui.py
```

Integration verifiers deliberately add authored demonstration tickets. Their
results are in `reports/integration`; those records do not measure model accuracy.
`verify_foundation.py` separately checks health and honest readiness when model,
database and reviewer configuration are unavailable.

## Local containers

Docker Compose provides a provider-neutral local deployment with MySQL, the
API and the reviewer dashboard. It publishes only the dashboard on localhost;
the API and database remain on the private Compose network. Docker Desktop must be
installed and running, and model artifacts must already exist under `artifacts/`.
The model and frozen routing policy are mounted read-only into the application
containers. Compose waits for MySQL and the API readiness check before it
starts each dependent service.

Create separate secrets and save them in the ignored `.env.compose` file:

```powershell
Copy-Item .env.compose.example .env.compose
& .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

Run the command two more times, then replace the three example values with
different generated values. Start and inspect the local deployment:

```powershell
docker compose --env-file .env.compose up --build -d
docker compose --env-file .env.compose ps
```

Open http://127.0.0.1:8501. The browser accesses the dashboard; the dashboard
server contacts the API internally. API docs and MySQL are not published to
the host. Stop containers while preserving database records with:

```powershell
docker compose --env-file .env.compose down
```

The named `ticket_mysql_data` volume retains ticket history. Removing that volume deletes
the database. A cloud release needs a separate hosting choice, access-control
design for human reviewers, secret management, backups and a validated production
migration plan; the local Compose setup does not provide those services.

The checked-in GitHub Actions workflow runs lint and the full test suite on Python
3.11 and 3.12 for pushes and pull requests. The first remote CI result appears only
after the repository is pushed to GitHub; this workspace cannot publish it.

## Scope

Assignments target five project-defined support teams. The application does not
contact a bank, change a transaction, issue a refund or infer actual urgency.
The token-based API and shared demo reviewer are a local prototype access model.
A cloud installation must restrict access to the reviewer UI; the UI's server-side
API credential alone does not authenticate the person viewing the dashboard.
