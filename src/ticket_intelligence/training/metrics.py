"""Classification diagnostics with explicit class-column alignment."""

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def classification_metrics(
    truth: Sequence[str],
    probabilities: np.ndarray,
    classes: Sequence[str],
    intent_to_team: Mapping[str, str],
) -> dict[str, float | int]:
    labels = list(classes)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(truth), len(labels)) or not len(truth):
        raise ValueError("Probability rows and class columns must match nonempty labels.")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative.")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Each probability row must sum to one.")
    if set(truth) - set(labels) or set(labels) - set(intent_to_team):
        raise ValueError("Every label must have a model column and a demo team.")
    indices = {label: index for index, label in enumerate(labels)}
    target = np.array([indices[label] for label in truth])
    winner = p.argmax(axis=1)
    predicted = [labels[index] for index in winner]
    top = np.argsort(-p, axis=1, kind="stable")[:, : min(3, len(labels))]
    one_hot = np.eye(len(labels))[target]
    return {
        "records": len(truth),
        "intent_accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "top_3_accuracy": float((top == target[:, None]).any(axis=1).mean()),
        "demo_team_accuracy": float(
            np.mean(
                [intent_to_team[a] == intent_to_team[b] for a, b in zip(truth, predicted)]
            )
        ),
        # Integer targets prevent sklearn from silently sorting our column labels.
        "negative_log_likelihood": float(log_loss(target, p, labels=list(range(len(labels))))),
        "multiclass_brier_sum": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
    }
