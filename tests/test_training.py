"""Test baseline artifacts and selection without downloading data or training on test data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ticket_intelligence.training.baseline import evaluate_predictions, train_baselines


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    destination = tmp_path / "data"
    destination.mkdir()
    mapping = [{"label_id": 0, "category": "z_card"}, {"label_id": 1, "category": "a_payment"}]
    (destination / "label_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    descriptions = {
        "z_card": "plastic card delivery shipment tracking",
        "a_payment": "transfer payment money beneficiary pending",
    }
    fit_rows, development_rows = [], []
    for label, description in descriptions.items():
        for index in range(10):
            fit_rows.append({"text": f"{description} trainingtoken{index}", "category": label})
        for index in range(2):
            development_rows.append(
                {
                    "text": f"{description} untouched_development_token request{index}",
                    "category": label,
                }
            )
    pd.DataFrame(fit_rows).to_csv(destination / "fit.csv", index=False)
    pd.DataFrame(development_rows).to_csv(destination / "development.csv", index=False)
    for filename in ("official_test.csv", "calibration.csv", "policy.csv", "challenge.csv"):
        (destination / filename).write_text("FORBIDDEN LATER PHASE DATA", encoding="utf-8")
    return destination


def test_saved_pipelines_selection_and_fit_only_vocabulary(data_dir: Path, tmp_path: Path) -> None:
    original_files = {path.name: path.read_bytes() for path in data_dir.iterdir()}
    artifacts, reports = tmp_path / "artifacts", tmp_path / "reports"
    report = train_baselines(data_dir, artifacts, reports)
    metadata = json.loads((artifacts / "metadata.json").read_text(encoding="utf-8"))
    development = pd.read_csv(data_dir / "development.csv")
    assert metadata["schema_version"] == 1
    assert metadata["classes"] == ["a_payment", "z_card"]
    assert metadata["label_mapping"][0]["category"] == "z_card"
    assert len(report["candidates"]) == 6
    assert report["majority_baseline"]["accuracy"] == 0.5
    best_key = max(
        (model["development_metrics"]["macro_f1"], model["development_metrics"]["accuracy"])
        for model in report["models"].values()
    )
    selected_metrics = report["models"][report["selected_model"]]["development_metrics"]
    assert (selected_metrics["macro_f1"], selected_metrics["accuracy"]) == best_key
    for family, model_metadata in metadata["models"].items():
        artifact_path = artifacts / model_metadata["path"]
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == model_metadata["sha256"]
        model = joblib.load(artifact_path)
        assert model.classes_.tolist() == model_metadata["classes"]
        assert "untouched_development_token" not in model.named_steps["tfidf"].vocabulary_
        predictions = model.predict(development["text"])
        metrics = evaluate_predictions(development["category"], predictions, metadata["classes"])
        assert metrics["accuracy"] == model_metadata["development_metrics"]["accuracy"]
        assert metrics["macro_f1"] == model_metadata["development_metrics"]["macro_f1"]
        probabilities = model.predict_proba(development["text"])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        assert model_metadata["latency"]["repetitions"] >= 100
        assert model_metadata["latency"]["batch_size"] == 1
        assert model_metadata["latency"]["p95_ms"] >= model_metadata["latency"]["p50_ms"]
        assert model_metadata["artifact_size_mb"] > 0
        assert (reports / family / "confusion_matrix.png").read_bytes().startswith(b"\x89PNG")
        errors = pd.read_csv(reports / family / "errors.csv")
        assert len(errors) == int((development["category"].to_numpy() != predictions).sum())
    assert {path.name: path.read_bytes() for path in data_dir.iterdir()} == original_files
    assert set(metadata["data_checksums"]) == {"fit.csv", "development.csv", "label_mapping.json"}


def test_training_opens_only_fit_and_development_csvs(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = []
    original_read_csv = pd.read_csv

    def recording_read_csv(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        assert Path(path).name in {"fit.csv", "development.csv"}
        loaded.append(Path(path).name)
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recording_read_csv)
    report = train_baselines(data_dir, tmp_path / "artifacts", tmp_path / "reports")
    assert loaded == ["fit.csv", "development.csv"]
    assert report["selection"]["fit_partitions"] == ["fit"]
    assert report["selection"]["partition"] == "development"


def test_metrics_include_all_supplied_classes() -> None:
    metrics = evaluate_predictions(["a", "b", "c"], ["a", "a", "c"], ["c", "b", "a"])
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(5 / 9)
    assert metrics["classification_report"]["b"]["f1-score"] == 0.0
    assert metrics["classification_report"]["c"]["support"] == 1


def test_normalized_fit_development_overlap_is_rejected(data_dir: Path, tmp_path: Path) -> None:
    fit = pd.read_csv(data_dir / "fit.csv")
    development = pd.read_csv(data_dir / "development.csv")
    development.loc[0, "text"] = "  " + fit.loc[0, "text"].upper().replace(" ", "   ") + "\t"
    development.to_csv(data_dir / "development.csv", index=False)
    with pytest.raises(ValueError, match="overlapping normalized-text groups"):
        train_baselines(data_dir, tmp_path / "artifacts", tmp_path / "reports")


def test_unexpected_development_label_is_rejected(data_dir: Path, tmp_path: Path) -> None:
    development = pd.read_csv(data_dir / "development.csv")
    development.loc[0, "category"] = "unlisted_intent"
    development.to_csv(data_dir / "development.csv", index=False)
    with pytest.raises(ValueError, match="label mismatch"):
        train_baselines(data_dir, tmp_path / "artifacts", tmp_path / "reports")
