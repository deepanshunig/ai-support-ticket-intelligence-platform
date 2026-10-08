"""Fit a frozen confidence policy on calibration/policy data, then evaluate once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import classification_report, log_loss

from ticket_intelligence.inference.predictor import Predictor
from ticket_intelligence.inference.routing import (
    GUARD_VERSION,
    RoutingEngine,
    acceptance_mask,
    temperature_scale,
)
from ticket_intelligence.training.baseline import _sha256_file
from ticket_intelligence.training.metrics import classification_metrics

CONFIDENCE_GRID = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999)
MARGIN_GRID = (0.05, 0.1, 0.2)
MIN_ACCEPTED = 50
MIN_INTENT_ACCURACY = 0.95
MIN_TEAM_ACCURACY = 0.95
MIN_CHALLENGE_REVIEW_RECALL = 0.75


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _ordered_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] < 2
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError("Expected normalized N-by-K finite class probabilities.")
    return values


def _class_ids(truth: list[str], classes: list[str]) -> np.ndarray:
    if not truth or set(truth) - set(classes):
        raise ValueError("Labels must be nonempty and present in the declared model classes.")
    order = {name: index for index, name in enumerate(classes)}
    return np.asarray([order[name] for name in truth], dtype=int)


def fit_temperature(
    probabilities: np.ndarray, truth: list[str], classes: list[str]
) -> dict[str, float]:
    """Fit one global temperature by calibration-set multiclass negative log likelihood."""
    values = _ordered_probabilities(probabilities)
    if len(truth) != len(values) or len(set(classes)) != len(classes):
        raise ValueError("Truth and probability rows must agree; classes must be unique.")
    y = _class_ids(truth, classes)
    logits = np.log(np.clip(values, np.finfo(float).tiny, 1.0))

    def loss(log_temperature: float) -> float:
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]))

    result = minimize_scalar(loss, bounds=(-2.0, 2.0), method="bounded", options={"xatol": 1e-10})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Global temperature fit failed: {result.message}")
    temperature = float(np.exp(result.x))
    return {
        "temperature": temperature,
        "uncalibrated_nll": loss(0.0),
        "calibration_fitted_nll": float(result.fun),
        "optimizer_success": 1.0,
    }


def _candidate_metrics(
    accepted: np.ndarray,
    truth: list[str],
    probabilities: np.ndarray,
    classes: list[str],
    intent_to_team: dict[str, str],
) -> dict[str, Any]:
    indices = np.flatnonzero(accepted)
    count = len(indices)
    if not count:
        return {
            "accepted": 0,
            "coverage": 0.0,
            "intent_accuracy": None,
            "team_accuracy": None,
            "intent_correct": 0,
            "team_correct": 0,
        }
    metrics = classification_metrics(
        [truth[index] for index in indices], probabilities[indices], classes, intent_to_team
    )
    return {
        "accepted": count,
        "coverage": float(count / len(truth)),
        "intent_accuracy": metrics["intent_accuracy"],
        "team_accuracy": metrics["demo_team_accuracy"],
        "intent_correct": int(round(float(metrics["intent_accuracy"]) * count)),
        "team_correct": int(round(float(metrics["demo_team_accuracy"]) * count)),
    }


def choose_policy(
    probabilities: np.ndarray,
    truth: list[str],
    known_feature_counts: np.ndarray,
    multi_team_flags: np.ndarray,
    challenge_probabilities: np.ndarray,
    challenge_counts: np.ndarray,
    challenge_multi_flags: np.ndarray,
    challenge_groups: list[str],
    intent_to_team: dict[str, str],
    *,
    classes: list[str] | None = None,
    confidence_grid: tuple[float, ...] = CONFIDENCE_GRID,
    margin_grid: tuple[float, ...] = MARGIN_GRID,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select the highest coverage among fixed-grid candidates meeting dev targets."""
    calibrated = _ordered_probabilities(probabilities)
    challenge = _ordered_probabilities(challenge_probabilities)
    if len(truth) != len(calibrated) or len(challenge_groups) != len(challenge):
        raise ValueError("Policy-development labels and evidence have inconsistent lengths.")
    class_count = calibrated.shape[1]
    classes = list(classes or intent_to_team)
    if len(classes) != class_count or set(classes) != set(intent_to_team) or not truth:
        raise ValueError("Policy class order/mapping must match the development evidence.")
    _class_ids(truth, classes)
    group_ids = sorted(set(challenge_groups))
    if not group_ids or not set(challenge_groups):
        raise ValueError("Policy selection requires separately authored challenge examples.")
    candidate_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    for threshold in confidence_grid:
        for margin in margin_grid:
            accepted = acceptance_mask(
                calibrated, known_feature_counts, multi_team_flags, threshold, margin, True
            )
            supported = _candidate_metrics(accepted, truth, calibrated, classes, intent_to_team)
            challenge_accept = acceptance_mask(
                challenge, challenge_counts, challenge_multi_flags, threshold, margin, True
            )
            by_group = {}
            for group in group_ids:
                selected = np.asarray([name == group for name in challenge_groups], dtype=bool)
                reviewed = int((~challenge_accept[selected]).sum())
                support = int(selected.sum())
                by_group[group] = {
                    "records": support,
                    "reviewed": reviewed,
                    "automatically_assigned": support - reviewed,
                    "review_recall": reviewed / support if support else 0.0,
                }
            eligible = (
                supported["accepted"] >= MIN_ACCEPTED
                and supported["intent_accuracy"] is not None
                and supported["intent_accuracy"] >= MIN_INTENT_ACCURACY
                and supported["team_accuracy"] >= MIN_TEAM_ACCURACY
                and all(
                    values["review_recall"] >= MIN_CHALLENGE_REVIEW_RECALL
                    for values in by_group.values()
                )
            )
            row = {
                "threshold": float(threshold),
                "margin": float(margin),
                **supported,
                "challenge_groups": by_group,
                "eligible": bool(eligible),
            }
            candidate_rows.append(row)
            if eligible:
                eligible_rows.append(row)
                key = (supported["coverage"], float(threshold), float(margin))
                if best_key is None or key > best_key:
                    best_key = key

    if best_key is None:
        return {
            "active": False,
            "threshold": 1.0,
            "margin": 1.0,
            "reason": "No policy-grid candidate met all prespecified selection targets.",
            "selection_targets": {
                "minimum_accepted": MIN_ACCEPTED,
                "minimum_accepted_intent_accuracy": MIN_INTENT_ACCURACY,
                "minimum_accepted_team_accuracy": MIN_TEAM_ACCURACY,
                "minimum_challenge_group_review_recall": MIN_CHALLENGE_REVIEW_RECALL,
            },
            "eligible_candidates": 0,
            "candidate_count": len(candidate_rows),
        }, candidate_rows
    selected = next(
        row
        for row in eligible_rows
        if (row["coverage"], row["threshold"], row["margin"]) == best_key
    )
    return {
        "active": True,
        "threshold": selected["threshold"],
        "margin": selected["margin"],
        "reason": "Highest supported-request coverage meeting every prespecified target.",
        "selection_targets": {
            "minimum_accepted": MIN_ACCEPTED,
            "minimum_accepted_intent_accuracy": MIN_INTENT_ACCURACY,
            "minimum_accepted_team_accuracy": MIN_TEAM_ACCURACY,
            "minimum_challenge_group_review_recall": MIN_CHALLENGE_REVIEW_RECALL,
        },
        "eligible_candidates": len(eligible_rows),
        "candidate_count": len(candidate_rows),
        "policy_development": selected,
    }, candidate_rows


def _read_policy_challenges(path: Path) -> list[dict[str, Any]]:
    development = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        partition = row.get("partition")
        if partition == "policy_development":
            development.append(row)
        elif partition != "heldout":
            raise ValueError("Challenge record has an unknown partition.")
    return development


def train_routing_policy(
    data_dir: Path = Path("data/processed/banking77"),
    artifact_dir: Path = Path("artifacts/baselines"),
    mapping_path: Path = Path("configs/team_mapping.json"),
    challenge_path: Path = Path("data/challenge_requests.jsonl"),
    routing_artifact_dir: Path = Path("artifacts/routing"),
    report_dir: Path = Path("reports/routing"),
) -> dict[str, Any]:
    """Fit calibration and policy without reading official test or held-out challenge text."""
    routing_artifact_dir, report_dir = Path(routing_artifact_dir), Path(report_dir)
    policy_path = routing_artifact_dir / "policy.json"
    if policy_path.exists():
        raise FileExistsError(f"A frozen policy already exists at {policy_path}; preserve it.")
    metadata = _json(Path(artifact_dir) / "metadata.json")
    if metadata.get("selected_model") != "logistic_regression":
        raise ValueError(
            "Step 6 is frozen to the development-selected Logistic Regression baseline."
        )
    predictor = Predictor.from_artifacts(artifact_dir, mapping_path)
    mapping_bytes = Path(mapping_path).read_bytes()
    mapping = json.loads(mapping_bytes)
    if set(predictor.classes) != set(mapping["intent_to_team"]):
        raise ValueError("Selected model and team mapping classes disagree.")
    classes = list(predictor.classes)
    intent_to_team = mapping["intent_to_team"]

    calibration_path = Path(data_dir) / "calibration.csv"
    calibration = pd.read_csv(calibration_path, dtype=str, keep_default_na=False)
    if set(calibration["category"]) != set(classes):
        raise ValueError("Calibration partition must contain every selected-model class.")
    calibration_signals = predictor.predict_probabilities(calibration["text"].tolist())
    temperature_fit = fit_temperature(
        calibration_signals, calibration["category"].tolist(), classes
    )
    temperature = temperature_fit["temperature"]

    policy_path_data = Path(data_dir) / "policy.csv"
    policy_data = pd.read_csv(policy_path_data, dtype=str, keep_default_na=False)
    if set(policy_data["category"]) != set(classes):
        raise ValueError("Policy partition must contain every selected-model class.")
    intent_router = RoutingEngine(
        predictor,
        {
            "schema_version": 1,
            "policy_version": "policy-development-only",
            "model_version": predictor.model_version,
            "model_artifact_sha256": predictor.artifact_sha256,
            "classes": classes,
            "temperature": temperature,
            "threshold": 0.0,
            "margin": 0.0,
            "active": True,
            "guard_version": GUARD_VERSION,
        },
        mapping,
    )
    policy_text = policy_data["text"].tolist()
    policy_signals = intent_router.batch_signals(policy_text)
    policy_probabilities = temperature_scale(policy_signals["probabilities"], temperature)

    challenge_policy = _read_policy_challenges(Path(challenge_path))
    if not challenge_policy or any(
        row.get("expected_action") != "needs_review" for row in challenge_policy
    ):
        raise ValueError("Policy development challenges must all require review.")
    challenge_signals = intent_router.batch_signals([row["text"] for row in challenge_policy])
    challenge_probabilities = temperature_scale(challenge_signals["probabilities"], temperature)
    selection, candidates = choose_policy(
        policy_probabilities,
        policy_data["category"].tolist(),
        policy_signals["known_feature_counts"],
        policy_signals["multi_team_flags"],
        challenge_probabilities,
        challenge_signals["known_feature_counts"],
        challenge_signals["multi_team_flags"],
        [row["group"] for row in challenge_policy],
        intent_to_team,
        classes=classes,
    )
    policy_partition_hashes = {
        "calibration.csv": _sha256_file(calibration_path),
        "policy.csv": _sha256_file(policy_path_data),
        "label_mapping.json": _sha256_file(Path(data_dir) / "label_mapping.json"),
        "policy_development_challenge_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row["id"] for row in challenge_policy)).encode()
        ).hexdigest(),
    }
    frozen = {
        "schema_version": 1,
        "frozen": True,
        "policy_version": "pending",
        "model_version": predictor.model_version,
        "model_artifact_sha256": predictor.artifact_sha256,
        "team_mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
        "classes": classes,
        "temperature": temperature,
        "threshold": selection["threshold"],
        "margin": selection["margin"],
        "active": selection["active"],
        "guard_version": GUARD_VERSION,
        "selection_reason": selection["reason"],
        "selection_partition": "policy.csv + policy_development challenges",
        "partition_sha256": policy_partition_hashes,
        "selection_targets": selection["selection_targets"],
    }
    canonical = json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    frozen["policy_version"] = f"routing-v1-{hashlib.sha256(canonical.encode()).hexdigest()[:12]}"
    grid_path = report_dir / "policy_candidates.csv"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "challenge_groups"}
            | {"challenge_groups_json": json.dumps(row["challenge_groups"], sort_keys=True)}
            for row in candidates
        ]
    ).to_csv(grid_path, index=False, lineterminator="\n")

    raw_nll = float(
        log_loss(policy_data["category"], policy_signals["probabilities"], labels=classes)
    )
    policy_nll = float(log_loss(policy_data["category"], policy_probabilities, labels=classes))
    policy_brier = float(
        classification_metrics(
            policy_data["category"].tolist(), policy_probabilities, classes, intent_to_team
        )["multiclass_brier_sum"]
    )
    calibrated_max = policy_probabilities.max(axis=1)
    accuracy = (
        policy_probabilities.argmax(axis=1)
        == np.asarray([classes.index(value) for value in policy_data["category"]])
    ).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    reliability = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        selected = (calibrated_max >= lower) & (
            calibrated_max <= upper if upper == bins[-1] else calibrated_max < upper
        )
        if selected.any():
            reliability.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "records": int(selected.sum()),
                    "mean_confidence": float(calibrated_max[selected].mean()),
                    "observed_accuracy": float(accuracy[selected].mean()),
                }
            )
    report = {
        "schema_version": 1,
        "policy": frozen,
        "temperature_fit": temperature_fit,
        "policy_development_selection": selection,
        "policy_candidates": len(candidates),
        "eligible_candidates": selection["eligible_candidates"],
        "calibration": {
            "partition": "calibration.csv",
            "records": len(calibration),
            "fitted_global_temperature": temperature,
            "fitted_sample_nll_before": temperature_fit["uncalibrated_nll"],
            "fitted_sample_nll_after": temperature_fit["calibration_fitted_nll"],
            "warning": (
                "Temperature was optimized on these same records; these are not "
                "held-out calibration-quality estimates."
            ),
        },
        "policy_partition_probabilistic_metrics": {
            "partition": "policy.csv",
            "records": len(policy_data),
            "raw_nll": raw_nll,
            "temperature_scaled_nll": policy_nll,
            "temperature_scaled_multiclass_brier_sum": policy_brier,
            "top_label_reliability_bins": reliability,
        },
        "challenge_policy_development": {
            "records": len(challenge_policy),
            "by_group": selection.get("policy_development", {}).get("challenge_groups", {}),
            "measure": "review recall only; all authored examples require review",
        },
        "heldout_challenges_and_official_test": "not read in the policy-fitting stage",
        "notes": [
            (
                "One global temperature preserves the top-intent ranking; it is not "
                "an out-of-domain detector."
            ),
            (
                "Threshold and margin were selected once on the policy partition "
                "and policy-development challenges."
            ),
            "No candidate meeting all targets creates an inactive review-only frozen policy.",
            (
                "The fixed multi-team clause detector is a heuristic and cannot "
                "detect every multi-intent request."
            ),
            "The authored challenges are diagnostic and not representative of a customer stream.",
        ],
    }
    # The policy becomes visible to runtime only after all fitting/report inputs finish.
    _write_json(report_dir / "report.json", report)
    _write_json(policy_path, frozen)
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="#65758b", label="Perfect reliability")
    axis.plot(
        [row["mean_confidence"] for row in reliability],
        [row["observed_accuracy"] for row in reliability],
        "o-",
        color="#0f766e",
    )
    axis.set(
        xlabel="Mean temperature-scaled top-label score",
        ylabel="Observed intent accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Policy partition reliability by top-score bin",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(report_dir / "policy_reliability.png", dpi=150)
    plt.close(figure)
    return report


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> dict[str, float] | None:
    if total <= 0:
        return None
    if not 0 <= successes <= total:
        raise ValueError("Wilson successes must be between zero and total.")
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return {
        "confidence": 0.95,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
    }


def _accepted_metrics(
    accepted: np.ndarray,
    truth: list[str],
    probabilities: np.ndarray,
    classes: list[str],
    intent_to_team: dict[str, str],
) -> dict[str, Any]:
    indices = np.flatnonzero(accepted)
    predicted = [classes[index] for index in probabilities.argmax(axis=1)]
    count = len(indices)
    correct_intent = sum(truth[index] == predicted[index] for index in indices)
    correct_team = sum(
        intent_to_team[truth[index]] == intent_to_team[predicted[index]] for index in indices
    )
    return {
        "total": len(truth),
        "accepted": count,
        "reviewed": len(truth) - count,
        "coverage": count / len(truth) if truth else 0.0,
        "accepted_intent_correct": correct_intent,
        "accepted_intent_accuracy": correct_intent / count if count else None,
        "accepted_intent_accuracy_wilson_95": wilson_interval(correct_intent, count),
        "accepted_team_correct": correct_team,
        "accepted_team_accuracy": correct_team / count if count else None,
        "accepted_team_accuracy_wilson_95": wilson_interval(correct_team, count),
        "all_request_intent_accuracy": float(np.mean(np.asarray(truth) == np.asarray(predicted)))
        if truth
        else 0.0,
    }


def evaluate_frozen_policy(
    data_dir: Path = Path("data/processed/banking77"),
    artifact_dir: Path = Path("artifacts/baselines"),
    mapping_path: Path = Path("configs/team_mapping.json"),
    challenge_path: Path = Path("data/challenge_requests.jsonl"),
    routing_artifact_dir: Path = Path("artifacts/routing"),
    report_dir: Path = Path("reports/final"),
) -> dict[str, Any]:
    """Evaluate once using a frozen policy; never fit, search or modify it here."""
    policy_path = Path(routing_artifact_dir) / "policy.json"
    before_policy_hash = _sha256_file(policy_path)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    policy = _json(policy_path)
    if policy.get("frozen") is not True:
        raise ValueError("Final evaluation requires a frozen routing policy.")
    predictor = Predictor.from_artifacts(artifact_dir, mapping_path)
    router = RoutingEngine.from_policy(predictor, policy_path, mapping_path)
    classes = list(predictor.classes)
    metadata = _json(Path(artifact_dir) / "metadata.json")
    if metadata["selected_model"] != "logistic_regression":
        raise ValueError("The final model must match the frozen development choice.")
    team_mapping = _json(mapping_path)["intent_to_team"]

    test_path = Path(data_dir) / "official_test.csv"
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False)
    truth = test["category"].tolist()
    raw = predictor.predict_probabilities(test["text"].tolist())
    probabilities = temperature_scale(raw, router.temperature)
    metrics = classification_metrics(truth, probabilities, classes, team_mapping)
    predicted_ids = probabilities.argmax(axis=1)
    predicted = [classes[index] for index in predicted_ids]
    all_report = classification_report(
        truth, predicted, labels=classes, output_dict=True, zero_division=0
    )
    signals = router.batch_signals(test["text"].tolist())
    accepted = acceptance_mask(
        probabilities,
        signals["known_feature_counts"],
        signals["multi_team_flags"],
        router.threshold,
        router.margin,
        router.active,
    )
    selective = _accepted_metrics(accepted, truth, probabilities, classes, team_mapping)
    errors = test.loc[:, ["text", "category"]].copy()
    errors["predicted_category"] = predicted
    errors["temperature_scaled_confidence"] = probabilities.max(axis=1)
    errors["automatically_assigned"] = accepted
    errors.loc[errors["category"] != errors["predicted_category"]].to_csv(
        Path(report_dir) / "official_test_errors.csv", index=False, lineterminator="\n"
    )

    heldout = []
    for line in Path(challenge_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("partition") == "heldout":
                heldout.append(row)
    if not heldout or any(row.get("expected_action") != "needs_review" for row in heldout):
        raise ValueError("Final challenge evaluation requires held-out review-only examples.")
    challenge_text = [row["text"] for row in heldout]
    challenge_signals = router.batch_signals(challenge_text)
    challenge_probabilities = temperature_scale(
        challenge_signals["probabilities"], router.temperature
    )
    challenge_accept = acceptance_mask(
        challenge_probabilities,
        challenge_signals["known_feature_counts"],
        challenge_signals["multi_team_flags"],
        router.threshold,
        router.margin,
        router.active,
    )
    challenges_by_group = {}
    for group in sorted({row["group"] for row in heldout}):
        positions = [index for index, row in enumerate(heldout) if row["group"] == group]
        review_count = sum(not challenge_accept[index] for index in positions)
        challenges_by_group[group] = {
            "records": len(positions),
            "expected_review": len(positions),
            "sent_to_review": review_count,
            "expected_review_recall": review_count / len(positions),
            "automatically_assigned": len(positions) - review_count,
            "interpretation": (
                "review recall only; the set cannot estimate review precision or workload"
            ),
        }
    audit_path = Path("reports/data/audit.json")
    audit = _json(audit_path)
    task_overlap = len(audit["official_train_test_hash_overlap"]["group_hashes"])
    report = {
        "schema_version": 1,
        "policy_version": router.policy_version,
        "policy_artifact_sha256": before_policy_hash,
        "model": {
            "selected": metadata["selected_model"],
            "artifact_sha256": predictor.artifact_sha256,
            "version": predictor.model_version,
        },
        "official_test": {
            "partition": "official_test.csv",
            "records": len(test),
            "raw_model_metrics": metrics,
            "classification_report": all_report,
            "accepted_automatic_routing": selective,
            "overlapping_normalized_train_test_text_groups_disclosed": task_overlap,
            "publisher_test_records_matching_train_text": audit["official_train_test_hash_overlap"][
                "test_records"
            ],
            "test_csv_sha256": _sha256_file(test_path),
            "errors_file": "official_test_errors.csv",
        },
        "authored_heldout_challenge": {
            "records": len(heldout),
            "by_group": challenges_by_group,
            "expected_action": "all needs_review",
            "overall_review_recall": float(np.mean(~challenge_accept)),
            "records_automatically_assigned": int(challenge_accept.sum()),
            "interpretation": (
                "authored balanced diagnostic examples, not production prevalence "
                "or a review precision estimate"
            ),
        },
        "policy": {
            "active": router.active,
            "threshold": router.threshold,
            "margin": router.margin,
            "temperature": router.temperature,
            "selection_targets": policy["selection_targets"],
            "policy_version": router.policy_version,
        },
        "warning": (
            "This is one final test of a policy selected on separate data. Publisher "
            "split has disclosed normalized duplicate overlap; it remains unchanged."
        ),
    }
    if _sha256_file(policy_path) != before_policy_hash:
        raise RuntimeError("Frozen policy bytes changed during final evaluation.")
    _write_json(Path(report_dir) / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluate-frozen",
        action="store_true",
        help="Read official test and held-out challenge after policy freeze.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/banking77"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--mapping", type=Path, default=Path("configs/team_mapping.json"))
    parser.add_argument("--challenge", type=Path, default=Path("data/challenge_requests.jsonl"))
    parser.add_argument("--routing-artifact-dir", type=Path, default=Path("artifacts/routing"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/routing"))
    args = parser.parse_args(argv)
    if args.evaluate_frozen:
        if not (args.routing_artifact_dir / "policy.json").is_file():
            parser.error("No frozen policy exists; refusing to access final held-out data.")
        report = evaluate_frozen_policy(
            args.data_dir,
            args.artifact_dir,
            args.mapping,
            args.challenge,
            args.routing_artifact_dir,
            Path("reports/final"),
        )
        official = report["official_test"]
        print(
            json.dumps(
                {
                    "policy_version": report["policy_version"],
                    "policy_active": report["policy"]["active"],
                    "official_test": {
                        "accuracy": official["raw_model_metrics"]["intent_accuracy"],
                        "macro_f1": official["raw_model_metrics"]["macro_f1"],
                        "accepted": official["accepted_automatic_routing"]["accepted"],
                        "coverage": official["accepted_automatic_routing"]["coverage"],
                        "accepted_intent_accuracy": official["accepted_automatic_routing"][
                            "accepted_intent_accuracy"
                        ],
                    },
                    "report": "reports/final/report.json",
                },
                indent=2,
            )
        )
    else:
        report = train_routing_policy(
            args.data_dir,
            args.artifact_dir,
            args.mapping,
            args.challenge,
            args.routing_artifact_dir,
            args.report_dir,
        )
        print(
            json.dumps(
                {
                    "frozen_policy": report["policy"],
                    "calibration": report["calibration"],
                    "selection": report["policy_development_selection"],
                    "report": str(args.report_dir / "report.json"),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
