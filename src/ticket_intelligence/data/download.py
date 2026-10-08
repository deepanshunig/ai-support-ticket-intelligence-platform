"""Download Banking77 from an immutable original-repository revision."""

import argparse
import csv
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

REPOSITORY = "https://github.com/PolyAI-LDN/task-specific-datasets"


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "AI-Support-Ticket-Student-Project"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def download(output_dir: Path, revision: str | None = None) -> dict:
    if revision is None:
        commit = json.loads(
            fetch("https://api.github.com/repos/PolyAI-LDN/task-specific-datasets/commits/master")
        )
        revision = commit["sha"]
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("Revision must be a full lowercase Git commit SHA")
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "source": REPOSITORY,
        "revision": revision,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "license": "CC BY 4.0",
        "files": {},
    }
    for name, repo_path in {
        "train.csv": "banking_data/train.csv",
        "test.csv": "banking_data/test.csv",
        "categories.json": "banking_data/categories.json",
        "LICENSE": "LICENSE",
    }.items():
        url = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{revision}/"
        payload = fetch(url + repo_path)
        if name.endswith(".csv"):
            reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
            if set(reader.fieldnames or []) != {"text", "category"}:
                raise ValueError(f"Unexpected CSV contract: {name}")
        elif name == "categories.json":
            categories = json.loads(payload)
            if len(categories) != 77 or len(set(categories)) != 77:
                raise ValueError("Expected 77 distinct categories")
        target = output_dir / name
        checksum = hashlib.sha256(payload).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Refusing to overwrite differing raw data: {target}")
        target.write_bytes(payload)
        provenance["files"][name] = {"url": url + repo_path, "sha256": checksum}
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/banking77"))
    parser.add_argument("--revision", help="Optional immutable original-repository commit SHA")
    args = parser.parse_args()
    provenance = download(args.output_dir, args.revision)
    print(json.dumps({"revision": provenance["revision"], "files": list(provenance["files"])}))


if __name__ == "__main__":
    main()
