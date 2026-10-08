"""Set up project-local PostgreSQL on Windows without installing a service."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from dotenv import dotenv_values, set_key

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
ARCHIVE_URL = (
    "https://get.enterprisedb.com/postgresql/"
    "postgresql-18.6-5-windows-x64-binaries.zip"
)
PORT = 55432


def run(arguments: list[str]) -> None:
    subprocess.run(
        arguments, check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, text=True, timeout=90,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def main() -> None:
    if os.name != "nt":
        raise SystemExit("This helper is for Windows. Use PostgreSQL or Compose on other hosts.")
    RUNTIME.mkdir(exist_ok=True)
    archive = RUNTIME / "postgresql-18.6.zip"
    bin_dir = RUNTIME / "pgsql/bin"
    if not (bin_dir / "postgres.exe").exists():
        if not archive.exists():
            print("Downloading the official PostgreSQL 18.6 archive...")
            temporary = archive.with_suffix(".part")
            with urlopen(ARCHIVE_URL, timeout=60) as response, temporary.open("wb") as target:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    target.write(chunk)
            temporary.replace(archive)
        print("Extracting server binaries, libraries and configuration files...")
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                if not entry.filename.startswith(("pgsql/bin/", "pgsql/lib/", "pgsql/share/")):
                    continue
                destination = (RUNTIME / entry.filename).resolve()
                if not destination.is_relative_to(RUNTIME.resolve()):
                    raise ValueError("Archive path escapes the project runtime.")
                bundle.extract(entry, RUNTIME)
        provenance = {
            "version": "18.6", "source_url": ARCHIVE_URL,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "origin_page": "https://www.enterprisedb.com/download-postgresql-binaries",
        }
        (RUNTIME / "postgres_provenance.json").write_text(json.dumps(provenance, indent=2))

    data_dir = RUNTIME / "pgdata"
    access_path = RUNTIME / "postgres_access.json"
    if not (data_dir / "PG_VERSION").exists():
        if data_dir.exists() and any(data_dir.iterdir()):
            raise ValueError("Existing PostgreSQL data directory is not empty; preserve it.")
        access = {"admin_password": secrets.token_urlsafe(32),
                  "app_password": secrets.token_urlsafe(32), "port": PORT}
        access_path.write_text(json.dumps(access), encoding="utf-8")
        password_file = RUNTIME / "initdb-password.txt"
        password_file.write_text(access["admin_password"], encoding="utf-8")
        try:
            run([str(bin_dir / "initdb.exe"), "-D", str(data_dir), "-U", "ticket_admin",
                 "--auth=scram-sha-256", f"--pwfile={password_file}", "--encoding=UTF8"])
        finally:
            password_file.unlink(missing_ok=True)
    access = json.loads(access_path.read_text(encoding="utf-8"))
    status = subprocess.run(
        [str(bin_dir / "pg_ctl.exe"), "-D", str(data_dir), "status"],
        capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if status.returncode != 0:
        run([str(bin_dir / "pg_ctl.exe"), "-D", str(data_dir), "-l",
             str(RUNTIME / "postgres.log"), "-o", f"-p {PORT} -h 127.0.0.1", "-w", "start"])

    import psycopg
    from psycopg import sql

    with psycopg.connect(
        host="127.0.0.1", port=PORT, user="ticket_admin",
        password=access["admin_password"], dbname="postgres", autocommit=True,
    ) as connection:
        if not connection.execute(
            "SELECT 1 FROM pg_roles WHERE rolname=%s", ("ticket_app",)
        ).fetchone():
            connection.execute(sql.SQL("CREATE ROLE ticket_app LOGIN PASSWORD {}").format(
                sql.Literal(access["app_password"])))
        if not connection.execute(
            "SELECT 1 FROM pg_database WHERE datname=%s", ("ticket_intelligence",)
        ).fetchone():
            connection.execute("CREATE DATABASE ticket_intelligence OWNER ticket_app")
        database_version = connection.execute("SHOW server_version").fetchone()[0]
    env_path = ROOT / ".env"
    env_path.touch(exist_ok=True)
    url = (f"postgresql+psycopg://ticket_app:{quote(access['app_password'])}"
           f"@127.0.0.1:{PORT}/ticket_intelligence")
    set_key(env_path, "TICKET_DATABASE_URL", url)
    if not dotenv_values(env_path).get("TICKET_REVIEWER_TOKEN"):
        set_key(env_path, "TICKET_REVIEWER_TOKEN", secrets.token_urlsafe(32))
    print(f"PostgreSQL {database_version} ready on localhost:{PORT}.")
    print("Application role has database ownership and no superuser privilege.")
    print("Local credentials saved in ignored .env and .runtime; no Windows service installed.")


if __name__ == "__main__":
    main()
