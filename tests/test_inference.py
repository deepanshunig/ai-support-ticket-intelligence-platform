"""Fixture-only inference checks for artifact integrity and label alignment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ticket_intelligence.inference.predictor import ArtifactError, Predictor


@pytest.fixture
def artifact_bundle(tmp_path: Path) -> tuple[Path, Path]:
    artifact_dir = tmp_path / "baselines"
    artifact_dir.mkdir()
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    ).fit(
        [
            "card stolen missing wallet",
            "lost card please replace",
            "card blocked pin problem",
            "forgot account passcode login",
            "account login password forgotten",
            "reset account password",
            "transfer payment delayed pending",
            "bank transfer not received",
            "payment transfer pending recipient",
        ],
        ["card_issue"] * 3 + ["login_issue"] * 3 + ["transfer_issue"] * 3,
    )
    model_path = artifact_dir / "logistic_regression.joblib"
    joblib.dump(pipeline, model_path)
    checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
    classes = list(pipeline.classes_)
    metadata = {
        "schema_version": 1,
        "selected_model": "logistic_regression",
        "classes": classes,
        "label_mapping": [
            {"label_id": index, "category": category}
            for index, category in enumerate(reversed(classes))
        ],
        "models": {
            "logistic_regression": {
                "path": model_path.name,
                "sha256": checksum,
                "model_version": f"logistic_regression-{checksum[:12]}",
                "classes": classes,
            }
        },
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    team_path = tmp_path / "team_mapping.json"
    team_path.write_text(
        json.dumps(
            {
                "teams": [{"id": "card_team"}, {"id": "account_team"}, {"id": "payment_team"}],
                "intent_to_team": {
                    "card_issue": "card_team",
                    "login_issue": "account_team",
                    "transfer_issue": "payment_team",
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact_dir, team_path


def _rewrite_json(path: Path, mutation) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_probability_columns_ranking_and_team_alignment(artifact_bundle) -> None:
    artifact_dir, team_path = artifact_bundle
    predictor = Predictor.from_artifacts(artifact_dir, team_path)
    texts = ["lost card stolen wallet", "reset account password", "bank transfer delayed"]
    probabilities = predictor.predict_probabilities(texts)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3))
    assert probabilities.shape == (3, 3)
    result = predictor.predict(texts[0])
    assert result["predicted_category"] == "card_issue"
    assert result["proposed_team"] == "card_team"
    candidates = result["candidates"]
    assert len(candidates) == 3
    assert [item["probability"] for item in candidates] == sorted(
        [item["probability"] for item in candidates], reverse=True
    )
    for candidate in candidates:
        index = predictor.classes.index(candidate["category"])
        assert candidate["probability"] == pytest.approx(probabilities[0, index])
    assert result["confidence_kind"] == "uncalibrated_model_probability"
    assert result["confidence"] == candidates[0]["probability"]
    assert result["known_word_feature_count"] > 0
    assert result["no_known_vocabulary"] is False
    assert len(result["artifact_sha256"]) == 64
    assert result["model_version"].endswith(result["artifact_sha256"][:12])


def test_no_known_vocabulary_is_flagged_without_claiming_ood(artifact_bundle) -> None:
    predictor = Predictor.from_artifacts(*artifact_bundle)
    result = predictor.predict("zxqv wxyzz")
    assert result["no_known_vocabulary"] is True
    assert result["known_word_feature_count"] == 0
    assert "routing_decision" not in result
    assert "out_of_domain" not in result
    assert len(result["candidates"]) == 3


def test_checksum_mismatch_rejected_before_deserialization(artifact_bundle) -> None:
    artifact_dir, team_path = artifact_bundle
    model_path = artifact_dir / "logistic_regression.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"unexpected change")
    with patch("ticket_intelligence.inference.predictor.joblib.load") as deserialize:
        with pytest.raises(ArtifactError, match="checksum mismatch"):
            Predictor.from_artifacts(artifact_dir, team_path)
        deserialize.assert_not_called()


def test_metadata_class_order_mismatch_rejected(artifact_bundle) -> None:
    artifact_dir, team_path = artifact_bundle
    _rewrite_json(
        artifact_dir / "metadata.json", lambda value: value["classes"].reverse()
    )
    with pytest.raises(ArtifactError, match="class order mismatch"):
        Predictor.from_artifacts(artifact_dir, team_path)


def test_artifact_classes_must_match_metadata(artifact_bundle) -> None:
    artifact_dir, team_path = artifact_bundle

    def reverse_classes(value):
        value["classes"].reverse()
        value["models"]["logistic_regression"]["classes"].reverse()

    _rewrite_json(artifact_dir / "metadata.json", reverse_classes)
    with pytest.raises(ArtifactError, match="Loaded model class order"):
        Predictor.from_artifacts(artifact_dir, team_path)


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "unknown_team"])
def test_mapping_coverage_and_references_are_validated(artifact_bundle, mutation) -> None:
    artifact_dir, team_path = artifact_bundle

    def change_mapping(value):
        mapping = value["intent_to_team"]
        if mutation == "missing":
            del mapping["card_issue"]
        elif mutation == "unexpected":
            mapping["untrained_issue"] = "card_team"
        else:
            mapping["card_issue"] = "unconfigured_team"

    _rewrite_json(team_path, change_mapping)
    with pytest.raises(ArtifactError, match="coverage mismatch|declared team ID"):
        Predictor.from_artifacts(artifact_dir, team_path)


def test_artifact_path_cannot_escape_bundle(artifact_bundle) -> None:
    artifact_dir, team_path = artifact_bundle

    def change_path(value):
        value["models"]["logistic_regression"]["path"] = "../outside.joblib"

    _rewrite_json(artifact_dir / "metadata.json", change_path)
    with pytest.raises(ArtifactError, match="stay inside"):
        Predictor.from_artifacts(artifact_dir, team_path)


@pytest.mark.parametrize("text", ["", "   \t", None, 123])
def test_invalid_prediction_text_is_rejected(artifact_bundle, text) -> None:
    predictor = Predictor.from_artifacts(*artifact_bundle)
    with pytest.raises(ValueError, match="nonempty strings"):
        predictor.predict(text)


def test_empty_batch_and_single_string_contract(artifact_bundle) -> None:
    predictor = Predictor.from_artifacts(*artifact_bundle)
    assert predictor.predict_probabilities([]).shape == (0, len(predictor.classes))
    with pytest.raises(TypeError, match="sequence of texts"):
        predictor.predict_probabilities("one string")
