import numpy as np
import pytest

from ticket_intelligence.training.metrics import classification_metrics


def test_metrics_use_probability_column_order_and_separate_team_correctness():
    result = classification_metrics(
        ["z", "a"], np.array([[0.8, 0.2], [0.6, 0.4]]), ["z", "a"], {"z": "x", "a": "x"}
    )
    assert result["intent_accuracy"] == 0.5
    assert result["demo_team_accuracy"] == 1
    assert result["negative_log_likelihood"] == pytest.approx(-np.log([0.8, 0.4]).mean())


def test_metrics_reject_probability_shape_and_normalization_errors():
    with pytest.raises(ValueError, match="sum to one"):
        classification_metrics(["a"], np.array([[0.4, 0.4]]), ["a", "b"], {"a": "t", "b": "t"})
