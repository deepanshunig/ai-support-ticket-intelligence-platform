"""Frozen global-temperature policy shared by offline selection and serving.

The clause guard is a fixed heuristic for multiple team signals, not a reliable
multi-intent or out-of-domain detector. Thresholds are selected outside this
module; runtime never learns from requests or opens evaluation datasets.
"""

from __future__ import annotations

import hashlib
import json
import re
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from ticket_intelligence.inference.predictor import ArtifactError, Predictor

GUARD_VERSION = "multi-team-clauses-v1"
BOUNDARY_TOLERANCE = 1e-12
CLAUSE_SPLIT = re.compile(r"[;.!?]+|\b(?:and|also|plus)\b", flags=re.IGNORECASE)


class PolicyError(ArtifactError):
    """A routing policy is invalid or does not belong to the loaded bundle."""


def _probabilities(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.ndim != 2
        or result.shape[1] < 1
        or not np.isfinite(result).all()
        or (result < 0).any()
        or (result > 1).any()
        or not np.allclose(result.sum(axis=1), 1.0, atol=1e-8, rtol=0)
    ):
        raise ValueError("Expected a finite, normalized N-by-K probability array.")
    return result


def _number(value: Any, name: str, *, unit: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise PolicyError(f"Policy {name} must be a finite number.")
    result = float(value)
    if (unit and not 0 <= result <= 1) or (not unit and result <= 0):
        raise PolicyError(f"Policy {name} is outside its allowed range.")
    return result


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply softmax(log(p) / T), preserving column order and normalization."""
    values = _probabilities(probabilities)
    temperature = _number(temperature, "temperature")
    if not len(values):
        return values.copy()
    if temperature == 1.0:
        return values.copy()
    with np.errstate(divide="ignore"):
        logits = np.log(values)
    logits -= logits.max(axis=1, keepdims=True)
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(logits / temperature)
    return weights / weights.sum(axis=1, keepdims=True)


def acceptance_mask(
    probabilities: np.ndarray,
    known_feature_counts: np.ndarray,
    multi_team_flags: np.ndarray,
    threshold: float,
    margin: float,
    active: bool = True,
) -> np.ndarray:
    """Accept only known, single-team signals meeting inclusive policy limits.

    A tolerance of 1e-12 handles floating-point subtraction at exact boundaries.
    Input probabilities must already have the selected temperature applied.
    """
    values = _probabilities(probabilities)
    threshold = _number(threshold, "threshold", unit=True)
    margin = _number(margin, "margin", unit=True)
    if type(active) is not bool:
        raise PolicyError("Policy active must be a boolean.")
    counts = np.asarray(known_feature_counts, dtype=float)
    flags = np.asarray(multi_team_flags, dtype=bool)
    if (
        counts.shape != (len(values),)
        or flags.shape != (len(values),)
        or not np.isfinite(counts).all()
        or (counts < 0).any()
        or (counts != np.floor(counts)).any()
    ):
        raise ValueError("Known-feature counts and multi-team flags must match the batch.")
    sorted_values = np.sort(values, axis=1)
    top = sorted_values[:, -1]
    runner = sorted_values[:, -2] if values.shape[1] > 1 else np.zeros(len(values))
    return (
        active
        & (counts > 0)
        & ~flags
        & (top >= threshold - BOUNDARY_TOLERANCE)
        & ((top - runner) >= margin - BOUNDARY_TOLERANCE)
    )


class RoutingEngine:
    """Apply one frozen temperature, confidence threshold, and margin globally."""

    def __init__(self, predictor: Predictor, policy: dict[str, Any], team_mapping: dict) -> None:
        if type(policy.get("schema_version")) is not int or policy["schema_version"] != 1:
            raise PolicyError("Unsupported routing policy schema_version; expected 1.")
        if policy.get("guard_version") != GUARD_VERSION:
            raise PolicyError("Routing policy guard_version does not match the fixed guard.")
        if policy.get("classes") != list(predictor.classes):
            raise PolicyError("Routing policy class order does not match the model.")
        if policy.get("model_artifact_sha256") != predictor.artifact_sha256:
            raise PolicyError("Routing policy model artifact SHA256 does not match the model.")
        if policy.get("model_version") != predictor.model_version:
            raise PolicyError("Routing policy model version does not match the model.")
        if type(policy.get("active")) is not bool:
            raise PolicyError("Routing policy active must be a boolean.")
        if (
            not isinstance(policy.get("policy_version"), str)
            or not policy["policy_version"].strip()
        ):
            raise PolicyError("Routing policy requires a nonempty policy_version.")
        mapping = team_mapping.get("intent_to_team", team_mapping)
        if (
            not isinstance(mapping, dict)
            or set(mapping) != set(predictor.classes)
            or any(not isinstance(team, str) or not team.strip() for team in mapping.values())
        ):
            raise PolicyError("Routing team mapping must cover the exact model classes.")
        self.predictor = predictor
        self.classes = tuple(predictor.classes)
        self.intent_to_team = dict(mapping)
        self.temperature = _number(policy.get("temperature"), "temperature")
        self.threshold = _number(policy.get("threshold"), "threshold", unit=True)
        self.margin = _number(policy.get("margin"), "margin", unit=True)
        self.active = policy["active"]
        self.policy_version = policy["policy_version"]

    @classmethod
    def from_policy(
        cls, predictor: Predictor, policy_path: str | Path, mapping_path: str | Path
    ) -> RoutingEngine:
        """Require a frozen policy bound to the exact model and mapping bytes."""
        try:
            policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
            mapping_bytes = Path(mapping_path).read_bytes()
            mapping = json.loads(mapping_bytes)
        except (OSError, json.JSONDecodeError) as error:
            raise PolicyError(
                "Cannot read routing policy or team mapping configuration."
            ) from error
        if not isinstance(policy, dict) or not isinstance(mapping, dict):
            raise PolicyError("Routing policy and team mapping must be JSON objects.")
        if policy.get("frozen") is not True:
            raise PolicyError("Runtime routing policy must be frozen.")
        mapping_sha = hashlib.sha256(mapping_bytes).hexdigest()
        if policy.get("team_mapping_sha256") != mapping_sha:
            raise PolicyError(
                "Routing policy team mapping SHA256 does not match the configuration."
            )
        return cls(predictor, policy, mapping)

    def batch_signals(self, texts: list[str] | tuple[str, ...]) -> dict[str, np.ndarray]:
        """Prepare raw model evidence once for every offline threshold candidate."""
        if isinstance(texts, str):
            raise TypeError("Pass a sequence of texts, not one text string.")
        texts = list(texts)
        probabilities = self.predictor.predict_probabilities(texts)
        counts = self.predictor.known_feature_counts(texts)
        flags = np.zeros(len(texts), dtype=bool)
        clauses, owners = [], []
        for owner, request in enumerate(texts):
            parts = [
                part.strip()
                for part in CLAUSE_SPLIT.split(request)
                if len(re.findall(r"\b\w+\b", part)) >= 3
            ][:8]
            if len(parts) >= 2:
                clauses.extend(parts)
                owners.extend([owner] * len(parts))
        if clauses:
            clause_probabilities = self.predictor.predict_probabilities(clauses)
            clause_counts = self.predictor.known_feature_counts(clauses)
            signaled_teams: dict[int, set[str]] = {}
            for index, owner in enumerate(owners):
                category_index = int(np.argmax(clause_probabilities[index]))
                if clause_counts[index] >= 2 and clause_probabilities[index, category_index] >= 0.6:
                    team = self.intent_to_team[self.classes[category_index]]
                    signaled_teams.setdefault(owner, set()).add(team)
            for owner, teams in signaled_teams.items():
                flags[owner] = len(teams) >= 2
        return {
            "probabilities": probabilities,
            "known_feature_counts": counts,
            "multi_team_flags": flags,
        }

    def policy_metadata(self) -> dict[str, Any]:
        return {
            "mode": "automatic_routing" if self.active else "review_only",
            "policy_version": self.policy_version,
            "active": self.active,
            "temperature": self.temperature,
            "threshold": self.threshold,
            "margin": self.margin,
            "guard_version": GUARD_VERSION,
        }

    def predict(self, text: str) -> dict[str, Any]:
        signals = self.batch_signals([text])
        raw = signals["probabilities"][0]
        calibrated = temperature_scale(signals["probabilities"], self.temperature)
        accepted = bool(
            acceptance_mask(
                calibrated,
                signals["known_feature_counts"],
                signals["multi_team_flags"],
                self.threshold,
                self.margin,
                self.active,
            )[0]
        )
        order = np.argsort(-raw, kind="stable")
        top = float(calibrated[0, order[0]])
        runner = float(calibrated[0, order[1]]) if len(order) > 1 else 0.0
        count, multi = int(signals["known_feature_counts"][0]), bool(signals["multi_team_flags"][0])
        category = self.classes[order[0]]
        team = self.intent_to_team[category]
        reason = None
        if not self.active:
            reason = "policy_inactive"
        elif count == 0:
            reason = "no_known_vocabulary"
        elif multi:
            reason = "multiple_team_signals"
        elif top < self.threshold - BOUNDARY_TOLERANCE:
            reason = "below_threshold"
        elif top - runner < self.margin - BOUNDARY_TOLERANCE:
            reason = "below_margin"
        return {
            "predicted_category": category,
            "proposed_team": team,
            "confidence": top,
            "raw_confidence": float(raw[order[0]]),
            "confidence_kind": "temperature_scaled_model_probability",
            "candidates": [
                {
                    "category": self.classes[index],
                    "probability": float(calibrated[0, index]),
                    "team": self.intent_to_team[self.classes[index]],
                }
                for index in order[:3]
            ],
            "known_word_feature_count": count,
            "no_known_vocabulary": count == 0,
            "multiple_team_signals": multi,
            "model_version": self.predictor.model_version,
            "artifact_sha256": self.predictor.artifact_sha256,
            "routing_action": "assigned" if accepted else "needs_review",
            "assigned_team": team if accepted else None,
            "review_reason": reason,
            "policy_version": self.policy_version,
        }
