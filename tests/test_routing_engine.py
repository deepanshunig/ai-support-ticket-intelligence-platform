"""Fixture-only policy boundaries, bundle identity, and fixed-guard behavior."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from ticket_intelligence.inference.routing import (
    GUARD_VERSION,
    PolicyError,
    RoutingEngine,
    acceptance_mask,
    temperature_scale,
)


class FixturePredictor:
    classes = ("card_issue", "login_issue", "transfer_issue")
    model_version = "fixture-baseline-v1"
    artifact_sha256 = "a" * 64

    def predict_probabilities(self, texts):
        result = []
        for text in texts:
            text = text.casefold()
            if text.startswith("weak transfer"):
                result.append([0.21, 0.21, 0.58])
            elif "uncertain" in text:
                result.append([0.45, 0.40, 0.15])
            elif "card" in text:
                result.append([0.80, 0.10, 0.10])
            elif "login" in text:
                result.append([0.10, 0.80, 0.10])
            elif "transfer" in text:
                result.append([0.10, 0.10, 0.80])
            else:
                result.append([0.70, 0.20, 0.10])
        return np.asarray(result, dtype=float).reshape(-1, 3)

    def known_feature_counts(self, texts):
        return np.asarray([0 if "zxqv" in text else 2 for text in texts])


@pytest.fixture
def bundle(tmp_path):
    predictor = FixturePredictor()
    mapping = {
        "intent_to_team": {
            "card_issue": "cards",
            "login_issue": "accounts",
            "transfer_issue": "payments",
        }
    }
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    policy = {
        "schema_version": 1,
        "policy_version": "fixture-policy-v1",
        "model_version": predictor.model_version,
        "model_artifact_sha256": predictor.artifact_sha256,
        "team_mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        "classes": list(predictor.classes),
        "temperature": 1.0,
        "threshold": 0.6,
        "margin": 0.1,
        "active": True,
        "frozen": True,
        "guard_version": GUARD_VERSION,
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return predictor, mapping, mapping_path, policy, policy_path


def test_temperature_scaling_preserves_order_and_normalization():
    raw = np.array([[0.8, 0.15, 0.05], [0.0, 1.0, 0.0], [0.1, 0.3, 0.6]])
    scaled = temperature_scale(raw, 2.0)
    np.testing.assert_allclose(scaled.sum(axis=1), np.ones(3))
    np.testing.assert_array_equal(scaled.argmax(axis=1), raw.argmax(axis=1))
    np.testing.assert_allclose(scaled[0], np.sqrt(raw[0]) / np.sqrt(raw[0]).sum())
    np.testing.assert_allclose(temperature_scale(raw, 1.0), raw)
    assert scaled[0, 0] < raw[0, 0]
    assert np.isfinite(scaled).all()
    assert temperature_scale(np.empty((0, 3)), 1.0).shape == (0, 3)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), True])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(PolicyError, match="temperature"):
        temperature_scale(np.array([[0.8, 0.2]]), temperature)


def test_acceptance_uses_inclusive_boundaries_and_all_guards():
    probabilities = np.array(
        [
            [0.6, 0.4, 0],
            [0.6 - 1e-9, 0.4 + 1e-9, 0],
            [0.75, 0.25, 0],
            [0.75, 0.25, 0],
            [0.75, 0.25, 0],
        ]
    )
    counts = np.array([2, 2, 0, 2, 2])
    multi = np.array([False, False, False, True, False])
    expected = [True, False, False, False, True]
    np.testing.assert_array_equal(acceptance_mask(probabilities, counts, multi, 0.6, 0.2), expected)
    np.testing.assert_array_equal(
        acceptance_mask(probabilities, counts, multi, 0.6, 0.2, False), np.zeros(5, dtype=bool)
    )
    margin_boundary = np.array([[0.55, 0.45], [0.55 - 1e-9, 0.45 + 1e-9]])
    np.testing.assert_array_equal(
        acceptance_mask(margin_boundary, np.ones(2), np.zeros(2, dtype=bool), 0.5, 0.1),
        [True, False],
    )


def test_shared_acceptance_rejects_bad_probability_shape_and_evidence():
    with pytest.raises(ValueError, match="probability array"):
        temperature_scale(np.array([[0.6, 0.6]]), 1.0)
    with pytest.raises(ValueError, match="match the batch"):
        acceptance_mask(np.array([[0.8, 0.2]]), np.ones(2), np.zeros(1), 0.5, 0.1)


def test_runtime_and_batch_acceptance_agree_and_report_calibrated_confidence(bundle):
    predictor, mapping, _mapping_path, policy, _policy_path = bundle
    policy = {**policy, "temperature": 2.0, "threshold": 0.5}
    engine = RoutingEngine(predictor, policy, mapping)
    texts = ["card missing today", "login forgotten today", "transfer delayed today", "zxqv"]
    signals = engine.batch_signals(texts)
    scaled = temperature_scale(signals["probabilities"], engine.temperature)
    accepted = acceptance_mask(
        scaled,
        signals["known_feature_counts"],
        signals["multi_team_flags"],
        engine.threshold,
        engine.margin,
        engine.active,
    )
    for index, text in enumerate(texts):
        result = engine.predict(text)
        assert (result["routing_action"] == "assigned") == accepted[index]
        assert result["confidence"] == pytest.approx(scaled[index].max())
        assert result["raw_confidence"] == pytest.approx(signals["probabilities"][index].max())
        assert result["confidence_kind"] == "temperature_scaled_model_probability"
        assert result["policy_version"] == "fixture-policy-v1"
        assert result["artifact_sha256"] == predictor.artifact_sha256
        candidates = result["candidates"]
        assert sum(item["probability"] for item in candidates) == pytest.approx(1)
        for candidate in candidates:
            class_index = predictor.classes.index(candidate["category"])
            assert candidate["probability"] == pytest.approx(scaled[index, class_index])


def test_oov_and_multi_team_signals_force_review(bundle):
    predictor, mapping, _mapping_path, policy, _policy_path = bundle
    engine = RoutingEngine(predictor, policy, mapping)
    unknown = engine.predict("zxqv wxyzz unknown")
    assert unknown["review_reason"] == "no_known_vocabulary"
    assert unknown["assigned_team"] is None
    multi = engine.predict("card missing today and transfer delayed today")
    assert multi["multiple_team_signals"] is True
    assert multi["routing_action"] == "needs_review"
    assert multi["review_reason"] == "multiple_team_signals"
    same_team = engine.predict("card missing today; card pin blocked")
    assert same_team["multiple_team_signals"] is False
    assert same_team["routing_action"] == "assigned"
    weak = engine.predict("card missing today; weak transfer delayed today")
    assert weak["multiple_team_signals"] is False


def test_guard_uses_fixed_minimum_words_and_maximum_eight_clauses(bundle):
    predictor, mapping, _mapping_path, policy, _policy_path = bundle
    engine = RoutingEngine(predictor, policy, mapping)
    assert not engine.predict("card missing; transfer delayed today")["multiple_team_signals"]
    first_eight = "; ".join(["card missing today"] * 8)
    assert not engine.predict(first_eight + "; transfer delayed today")["multiple_team_signals"]


def test_threshold_margin_and_inactive_reasons_are_deterministic(bundle):
    predictor, mapping, _mapping_path, policy, _policy_path = bundle
    assert RoutingEngine(predictor, policy, mapping).predict("uncertain request")[
        "review_reason"
    ] == ("below_threshold")
    margin_policy = {**policy, "threshold": 0.4, "margin": 0.1}
    assert RoutingEngine(predictor, margin_policy, mapping).predict("uncertain request")[
        "review_reason"
    ] == ("below_margin")
    inactive = RoutingEngine(predictor, {**policy, "active": False}, mapping)
    assert inactive.predict("card missing today")["review_reason"] == "policy_inactive"
    assert inactive.policy_metadata()["mode"] == "review_only"


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("model_artifact_sha256", "b" * 64, "model artifact SHA256"),
        ("model_version", "another-model", "model version"),
        ("team_mapping_sha256", "b" * 64, "team mapping SHA256"),
        ("classes", ["login_issue", "card_issue", "transfer_issue"], "class order"),
        ("frozen", False, "must be frozen"),
        ("guard_version", "another-guard", "guard_version"),
        ("temperature", 0, "temperature"),
        ("threshold", True, "threshold"),
        ("margin", -0.1, "margin"),
        ("active", "yes", "boolean"),
    ],
)
def test_mismatched_or_invalid_policy_rejected_before_serving(bundle, key, value, message):
    predictor, _mapping, mapping_path, policy, policy_path = bundle
    policy_path.write_text(json.dumps({**policy, key: value}), encoding="utf-8")
    with pytest.raises(PolicyError, match=message):
        RoutingEngine.from_policy(predictor, policy_path, mapping_path)


def test_mapping_byte_change_invalidates_policy_and_valid_policy_loads(bundle):
    predictor, _mapping, mapping_path, _policy, policy_path = bundle
    engine = RoutingEngine.from_policy(predictor, policy_path, mapping_path)
    assert engine.predict("card missing today")["assigned_team"] == "cards"
    mapping_path.write_bytes(mapping_path.read_bytes() + b"\n")
    with pytest.raises(PolicyError, match="team mapping SHA256"):
        RoutingEngine.from_policy(predictor, policy_path, mapping_path)
