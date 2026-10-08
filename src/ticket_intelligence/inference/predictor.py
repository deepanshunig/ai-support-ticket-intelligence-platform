"""Load trusted local baselines and expose uncalibrated intent predictions.

An artifact checksum checks consistency, not the safety of a pickle. Only load
artifacts trained locally or obtained from a source you explicitly trust.
Routing thresholds, calibration, and unfamiliar-request policies are deferred.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


class ArtifactError(ValueError):
    """The saved model, metadata, and routing configuration do not agree."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Cannot read artifact configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"{path.name} must contain a JSON object.")
    return value


def _labels(value: Any, source: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(label, str) or not label.strip() for label in value)
        or len(set(value)) != len(value)
    ):
        raise ArtifactError(f"{source} must contain unique, nonempty string classes.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ArtifactError(f"Cannot read model artifact {path}: {error}") from error
    return digest.hexdigest()


def _team_mapping(path: Path, classes: list[str]) -> dict[str, str]:
    configuration = _read_object(path)
    mapping = configuration.get("intent_to_team")
    teams = configuration.get("teams")
    if not isinstance(mapping, dict) or not isinstance(teams, list) or not teams:
        raise ArtifactError("Team mapping requires 'intent_to_team' and a nonempty 'teams' list.")
    team_ids = [team.get("id") if isinstance(team, dict) else None for team in teams]
    if (
        any(not isinstance(team, str) or not team.strip() for team in team_ids)
        or len(set(team_ids)) != len(team_ids)
    ):
        raise ArtifactError("Team mapping must declare unique, nonempty string team IDs.")
    missing = sorted(set(classes) - set(mapping))
    unexpected = sorted(set(mapping) - set(classes))
    if missing or unexpected:
        raise ArtifactError(
            f"Team mapping class coverage mismatch: missing={missing}, unexpected={unexpected}."
        )
    if any(not isinstance(team, str) or team not in team_ids for team in mapping.values()):
        raise ArtifactError("Every mapped intent must reference a declared team ID.")
    return mapping


class Predictor:
    """A fitted word-TF-IDF pipeline with versioned, proposed team assignments."""

    def __init__(
        self,
        pipeline: Pipeline,
        classes: list[str],
        intent_to_team: dict[str, str],
        model_version: str,
        artifact_sha256: str,
    ) -> None:
        self._pipeline = pipeline
        self._classes = tuple(classes)
        self._intent_to_team = dict(intent_to_team)
        self.model_version = model_version
        self.artifact_sha256 = artifact_sha256

    @property
    def classes(self) -> tuple[str, ...]:
        """Class order corresponding exactly to probability-array columns."""
        return self._classes

    @classmethod
    def from_artifacts(
        cls, artifact_dir: str | Path, team_mapping_path: str | Path
    ) -> Predictor:
        """Verify the selected artifact before unpickling a trusted local model."""
        directory = Path(artifact_dir).resolve()
        metadata = _read_object(directory / "metadata.json")
        if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
            raise ArtifactError("Unsupported baseline metadata schema_version; expected 1.")
        family = metadata.get("selected_model")
        models = metadata.get("models")
        if not isinstance(family, str) or not isinstance(models, dict) or family not in models:
            raise ArtifactError("Metadata selected_model must identify a saved model entry.")
        entry = models[family]
        if not isinstance(entry, dict):
            raise ArtifactError("Selected model metadata must be an object.")
        classes = _labels(metadata.get("classes"), "Metadata classes")
        if _labels(entry.get("classes"), "Selected model classes") != classes:
            raise ArtifactError("Top-level and selected model class order mismatch.")
        label_mapping = metadata.get("label_mapping")
        if not isinstance(label_mapping, list) or any(
            not isinstance(label, dict) for label in label_mapping
        ):
            raise ArtifactError("Metadata label_mapping must be a list of label records.")
        mapped_labels = _labels(
            [label.get("category") for label in label_mapping], "Metadata label_mapping"
        )
        if set(mapped_labels) != set(classes):
            raise ArtifactError("Metadata label_mapping class coverage mismatch.")
        if any(
            type(label.get("label_id")) is not int or label["label_id"] != index
            for index, label in enumerate(label_mapping)
        ):
            raise ArtifactError("Metadata label_mapping IDs must match ordered positions.")
        mapping = _team_mapping(Path(team_mapping_path), classes)
        filename = entry.get("path")
        checksum = entry.get("sha256")
        model_version = entry.get("model_version")
        if not isinstance(filename, str) or not filename:
            raise ArtifactError("Selected model metadata requires an artifact path.")
        if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
            raise ArtifactError("Selected model metadata requires a full lowercase SHA256.")
        if not isinstance(model_version, str) or not model_version.strip():
            raise ArtifactError("Selected model metadata requires a nonempty model_version.")
        artifact_path = (directory / filename).resolve()
        if artifact_path == directory or not artifact_path.is_relative_to(directory):
            raise ArtifactError("Model artifact path must stay inside the artifact directory.")
        actual_checksum = _sha256(artifact_path)
        if actual_checksum != checksum:
            raise ArtifactError(
                f"Model artifact checksum mismatch: expected {checksum}, got {actual_checksum}."
            )
        try:
            pipeline = joblib.load(artifact_path)
        except Exception as error:
            raise ArtifactError(f"Cannot load verified model artifact: {error}") from error
        if not isinstance(pipeline, Pipeline) or not hasattr(pipeline, "predict_proba"):
            raise ArtifactError("Model artifact must be a fitted probability-producing Pipeline.")
        vectorizer = pipeline.named_steps.get("tfidf")
        if not isinstance(vectorizer, TfidfVectorizer) or vectorizer.analyzer != "word":
            raise ArtifactError("Model pipeline requires a word-analyzer 'tfidf' vectorizer.")
        if not hasattr(vectorizer, "vocabulary_") or not hasattr(pipeline, "classes_"):
            raise ArtifactError("Model pipeline and word vectorizer must be fitted.")
        artifact_classes = _labels(list(pipeline.classes_), "Artifact classes")
        if artifact_classes != classes:
            raise ArtifactError("Loaded model class order does not match metadata classes.")
        return cls(pipeline, classes, mapping, model_version, actual_checksum)

    def predict_probabilities(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
        """Return raw, uncalibrated probabilities in ``classes`` order."""
        if isinstance(texts, str):
            raise TypeError("Pass a sequence of texts, not one text string.")
        batch = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in batch):
            raise ValueError("Prediction texts must be nonempty strings.")
        if not batch:
            return np.empty((0, len(self.classes)), dtype=float)
        probabilities = np.asarray(self._pipeline.predict_proba(batch), dtype=float)
        if (
            probabilities.shape != (len(batch), len(self.classes))
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        ):
            raise ArtifactError("Model produced invalid class probabilities.")
        return probabilities

    def known_feature_counts(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
        """Count nonzero word/ngram TF-IDF features without making predictions."""
        if isinstance(texts, str):
            raise TypeError("Pass a sequence of texts, not one text string.")
        batch = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in batch):
            raise ValueError("Prediction texts must be nonempty strings.")
        if not batch:
            return np.empty(0, dtype=int)
        vectors = self._pipeline.named_steps["tfidf"].transform(batch)
        return np.asarray(vectors.getnnz(axis=1), dtype=int).reshape(-1)

    def predict(self, text: str) -> dict[str, Any]:
        """Rank intents and propose a team; this does not decide automatic routing.

        Known-feature counts include word n-grams. A zero count flags an empty
        vocabulary representation, not a general out-of-domain detector.
        """
        probabilities = self.predict_probabilities([text])[0]
        best_indices = np.argsort(-probabilities, kind="stable")[:3]
        candidates = [
            {
                "category": self.classes[index],
                "probability": float(probabilities[index]),
                "team": self._intent_to_team[self.classes[index]],
            }
            for index in best_indices
        ]
        known_features = int(self.known_feature_counts([text])[0])
        first = candidates[0]
        return {
            "predicted_category": first["category"],
            "proposed_team": first["team"],
            "confidence": first["probability"],
            "confidence_kind": "uncalibrated_model_probability",
            "candidates": candidates,
            "known_word_feature_count": known_features,
            "no_known_vocabulary": known_features == 0,
            "model_version": self.model_version,
            "artifact_sha256": self.artifact_sha256,
        }
