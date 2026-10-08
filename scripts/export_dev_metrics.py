"""Supplement baseline reports without reading calibration/policy/test data."""

import json
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from ticket_intelligence.data.prepare import sha256_file
from ticket_intelligence.training.metrics import classification_metrics


def main() -> None:
    directory = Path("artifacts/baselines")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    development = pd.read_csv("data/processed/banking77/development.csv")
    mapping = json.loads(Path("configs/team_mapping.json").read_text(encoding="utf-8"))
    report = {"partition": "development", "confidence_kind": "uncalibrated", "models": {}}
    with threadpool_limits(limits=1):
        for family, entry in metadata["models"].items():
            path = directory / entry["path"]
            if sha256_file(path) != entry["sha256"]:
                raise ValueError("Baseline artifact checksum mismatch.")
            model = joblib.load(path)
            report["models"][family] = classification_metrics(
                development.category.tolist(),
                model.predict_proba(development.text.tolist()),
                model.classes_.tolist(),
                mapping["intent_to_team"],
            )
    Path("reports/baselines/operational_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
