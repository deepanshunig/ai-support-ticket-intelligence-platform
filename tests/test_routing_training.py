"""Policy fitting and final reporting boundaries."""

import numpy as np

from ticket_intelligence.training.routing import (
    choose_policy,
    fit_temperature,
    wilson_interval,
)


def test_temperature_fit_uses_declared_column_order_and_improves_nll():
    classes = ["z_transfer", "a_card"]
    probabilities = np.array([[0.40, 0.60], [0.60, 0.40], [0.40, 0.60], [0.60, 0.40]])
    truth = ["a_card", "z_transfer", "a_card", "z_transfer"]
    result = fit_temperature(probabilities, truth, classes)
    assert result["temperature"] < 1
    assert result["calibration_fitted_nll"] < result["uncalibrated_nll"]


def test_selection_requires_every_challenge_group_and_prefers_eligible_coverage():
    classes = ["a_card", "z_transfer"]
    teams = {"a_card": "cards", "z_transfer": "payments"}
    truth = ["a_card"] * 30 + ["z_transfer"] * 30
    probabilities = np.array([[0.98, 0.02]] * 30 + [[0.02, 0.98]] * 30)
    known = np.full(60, 2)
    multi = np.zeros(60, dtype=bool)
    group_order = [
        "clearly_unrelated",
        "nearby_unsupported_banking",
        "ambiguous_supported",
        "multi_intent",
    ]
    group_flags = np.array([False] * 36 + [True] * 12)
    challenge = np.array([[0.5, 0.5]] * 48)
    selection, candidates = choose_policy(
        probabilities,
        truth,
        known,
        multi,
        challenge,
        np.full(48, 2),
        group_flags,
        [group for group in group_order for _ in range(12)],
        teams,
        classes=classes,
        confidence_grid=(0.5, 0.9),
        margin_grid=(0.05, 0.1),
    )
    assert selection["active"] is True
    assert selection["policy_development"]["accepted"] == 60
    assert selection["threshold"] == 0.9
    assert selection["policy_development"]["intent_accuracy"] == 1
    assert len(candidates) == 4


def test_selection_falls_back_to_review_when_no_candidate_meets_targets():
    classes = ["a_card", "z_transfer"]
    teams = {"a_card": "cards", "z_transfer": "payments"}
    probabilities = np.array([[0.9, 0.1]] * 49)
    challenges = np.array([[0.5, 0.5]] * 48)
    groups = [name for name in ["a", "b", "c", "d"] for _ in range(12)]
    selection, _ = choose_policy(
        probabilities,
        ["a_card"] * 49,
        np.ones(49),
        np.zeros(49, dtype=bool),
        challenges,
        np.ones(48),
        np.zeros(48, dtype=bool),
        groups,
        teams,
        classes=classes,
        confidence_grid=(0.5, 0.9),
        margin_grid=(0.1,),
    )
    assert selection["active"] is False
    assert selection["threshold"] == selection["margin"] == 1.0


def test_wilson_interval_handles_no_accepted_records_and_bounds():
    assert wilson_interval(0, 0) is None
    assert wilson_interval(50, 50)["lower"] < 1
    assert wilson_interval(50, 50)["upper"] == 1
