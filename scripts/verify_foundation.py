"""Exercise the documented server command over real HTTP, then stop the process."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def fetch(url):
    try:
        response = urlopen(url, timeout=2)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


def main():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = os.environ | {
        "TICKET_API_PORT": str(port), "TICKET_ENVIRONMENT": "test",
        "TICKET_ARTIFACT_DIR": str(ROOT / "tmp/foundation-missing"),
        "TICKET_DATABASE_URL": "", "TICKET_REVIEWER_TOKEN": "",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "ticket_intelligence.api.main"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited with code {process.returncode}")
            try:
                status, health = fetch(f"http://127.0.0.1:{port}/health")
                break
            except (URLError, TimeoutError):
                time.sleep(0.2)
        else:
            raise RuntimeError("Server startup timed out")
        assert status == 200 and health["status"] == "ok"
        status, readiness = fetch(f"http://127.0.0.1:{port}/ready")
        assert status == 503 and readiness["status"] == "not_ready"
        status, schema = fetch(f"http://127.0.0.1:{port}/openapi.json")
        assert status == 200 and "/health" in schema["paths"]
        print("PASS: documented server startup, HTTP health, honest readiness and OpenAPI")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
