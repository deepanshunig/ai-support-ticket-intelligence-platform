"""Select two TF-IDF baselines using fit and development data exclusively."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

FAMILY_GRIDS = {"logistic_regression": (1.0, 4.0, 10.0), "naive_bayes": (0.1, 0.5, 1.0)}
TFIDF_PARAMETERS = {
    "lowercase": True,
    "analyzer": "word",
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 1,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    mapping = json.loads((data_dir / "label_mapping.json").read_text(encoding="utf-8"))
    if not isinstance(mapping, list) or not mapping:
        raise ValueError("label_mapping.json must contain a nonempty ordered list.")
    for index, item in enumerate(mapping):
        if (
            not isinstance(item, dict)
            or item.get("label_id") != index
            or not isinstance(item.get("category"), str)
            or not item["category"].strip()
        ):
            raise ValueError("Label mapping requires sequential IDs and nonempty category names.")
    labels = [item["category"] for item in mapping]
    if len(set(labels)) != len(labels) or len(labels) < 2:
        raise ValueError("The mapping must contain at least two unique categories.")
    frames = []
    for name in ("fit", "development"):
        frame = pd.read_csv(data_dir / f"{name}.csv", dtype=str, keep_default_na=False)
        if not {"text", "category"}.issubset(frame.columns):
            raise ValueError(f"{name}.csv must include text and category columns.")
        if frame.empty or frame["text"].str.strip().eq("").any():
            raise ValueError(f"{name}.csv contains empty text or no records.")
        observed = set(frame["category"])
        if observed != set(labels):
            raise ValueError(
                f"{name}.csv label mismatch: unknown={sorted(observed - set(labels))}; "
                f"missing={sorted(set(labels) - observed)}."
            )
        frames.append(frame.reset_index(drop=True))
    fit, development = frames
    normalized = [set(frame["text"].map(lambda text: " ".join(text.lower().split())))
                  for frame in frames]
    if normalized[0] & normalized[1]:
        raise ValueError("Fit and development contain overlapping normalized-text groups.")
    return fit, development, mapping


def evaluate_predictions(
    truth: list[str] | np.ndarray | pd.Series,
    predictions: list[str] | np.ndarray | pd.Series,
    labels: list[str],
) -> dict[str, Any]:
    """Compute intent metrics with the complete supplied label set."""
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=labels, average="macro")),
        "classification_report": classification_report(
            truth, predictions, labels=labels, output_dict=True, zero_division=0
        ),
    }


def _make_pipeline(family: str, value: float) -> tuple[Pipeline, dict[str, Any]]:
    if family == "logistic_regression":
        parameters = {"C": value, "max_iter": 1000, "random_state": 42, "solver": "lbfgs"}
        classifier = LogisticRegression(**parameters)
    elif family == "naive_bayes":
        parameters = {"alpha": value}
        classifier = MultinomialNB(**parameters)
    else:
        raise ValueError(f"Unknown model family: {family}.")
    return Pipeline(
        [("tfidf", TfidfVectorizer(**TFIDF_PARAMETERS)), ("classifier", classifier)]
    ), parameters


def _measure_latency(
    model: Pipeline, texts: list[str], *, repetitions: int = 100
) -> dict[str, Any]:
    if repetitions < 100 or not texts:
        raise ValueError("Latency measurement requires text and at least 100 repetitions.")
    for index in range(5):
        model.predict([texts[index % len(texts)]])
    elapsed = []
    for index in range(repetitions):
        started = time.perf_counter()
        model.predict([texts[index % len(texts)]])
        elapsed.append((time.perf_counter() - started) * 1000)
    return {
        "batch_size": 1,
        "repetitions": repetitions,
        "warmup_repetitions": 5,
        "p50_ms": float(np.percentile(elapsed, 50)),
        "p95_ms": float(np.percentile(elapsed, 95)),
        "minimum_ms": float(np.min(elapsed)),
        "maximum_ms": float(np.max(elapsed)),
        "scope": "Local Pipeline.predict, including TF-IDF; excludes HTTP and model loading.",
    }


def _write_diagnostics(
    development: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    destination: Path,
) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    incorrect = development["category"].to_numpy() != predictions
    row_ids = (
        development["row_id"]
        if "row_id" in development
        else pd.Series([f"development:{index:06d}" for index in range(len(development))])
    )
    errors = pd.DataFrame(
        {
            "row_id": row_ids,
            "text": development["text"],
            "true_category": development["category"],
            "predicted_category": predictions,
            "uncalibrated_confidence": probabilities.max(axis=1),
        }
    ).loc[incorrect]
    errors.to_csv(destination / "errors.csv", index=False, lineterminator="\n")
    matrix = confusion_matrix(development["category"], predictions, labels=labels)
    pairs = [
        {"true_category": labels[row], "predicted_category": labels[column], "count": int(count)}
        for row in range(len(labels))
        for column in range(len(labels))
        if row != column and (count := matrix[row, column]) > 0
    ]
    confusion_pairs = pd.DataFrame(pairs, columns=["true_category", "predicted_category", "count"])
    confusion_pairs.sort_values(
        ["count", "true_category", "predicted_category"], ascending=[False, True, True]
    ).to_csv(destination / "confusion_pairs.csv", index=False, lineterminator="\n")
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(destination / "confusion_matrix.csv")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axis = plt.subplots(figsize=(12, 10))
    plotted = axis.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(plotted, ax=axis, label="Development records")
    ids = np.arange(len(labels))
    axis.set_xticks(ids, ids, fontsize=6)
    axis.set_yticks(ids, ids, fontsize=6)
    axis.set_xlabel("Predicted label ID (publisher mapping)")
    axis.set_ylabel("True label ID (publisher mapping)")
    axis.set_title("Development intent confusion matrix")
    fig.tight_layout()
    fig.savefig(destination / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    return {
        "errors": str(destination / "errors.csv"),
        "confusion_pairs": str(destination / "confusion_pairs.csv"),
        "confusion_matrix": str(destination / "confusion_matrix.csv"),
        "confusion_plot": str(destination / "confusion_matrix.png"),
    }


def train_baselines(
    data_dir: Path = Path("data/processed/banking77"),
    artifact_dir: Path = Path("artifacts/baselines"),
    report_dir: Path = Path("reports/baselines"),
) -> dict[str, Any]:
    """Fit six candidates on fit data and select by development macro-F1/accuracy."""
    data_dir, artifact_dir, report_dir = (
        Path(data_dir).resolve(),
        Path(artifact_dir).resolve(),
        Path(report_dir).resolve(),
    )
    if artifact_dir.is_relative_to(data_dir) or report_dir.is_relative_to(data_dir):
        raise ValueError("Artifacts and reports must be outside the prepared data directory.")
    fit, development, mapping = _load_data(data_dir)
    labels = [item["category"] for item in mapping]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    versions = {
        "python": platform.python_version(),
        **{name: version(name) for name in ("scikit-learn", "numpy", "pandas", "joblib")},
    }
    data_checksums = {
        name: _sha256_file(data_dir / name)
        for name in ("fit.csv", "development.csv", "label_mapping.json")
    }
    models: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    fit_text = fit["text"].tolist()
    development_text = development["text"].tolist()
    majority_label = fit["category"].mode().iloc[0]
    majority = evaluate_predictions(
        development["category"], [majority_label] * len(development), labels
    )
    majority["category"] = majority_label
    with threadpool_limits(limits=1):
        for family, grid in FAMILY_GRIDS.items():
            best: tuple[Pipeline, dict[str, Any], dict[str, Any], float] | None = None
            search_training_seconds = 0.0
            for value in grid:
                pipeline, parameters = _make_pipeline(family, value)
                started = time.perf_counter()
                pipeline.fit(fit_text, fit["category"])
                training_seconds = time.perf_counter() - started
                search_training_seconds += training_seconds
                predictions = pipeline.predict(development_text)
                metrics = evaluate_predictions(development["category"], predictions, labels)
                candidates.append(
                    {
                        "family": family,
                        "classifier_parameters": parameters,
                        "development_accuracy": metrics["accuracy"],
                        "development_macro_f1": metrics["macro_f1"],
                        "training_seconds": training_seconds,
                    }
                )
                if best is None or (metrics["macro_f1"], metrics["accuracy"]) > (
                    best[2]["macro_f1"], best[2]["accuracy"]
                ):
                    best = (pipeline, parameters, metrics, training_seconds)
            if best is None:
                raise ValueError(f"No candidate trained for {family}.")
            pipeline, parameters, metrics, training_seconds = best
            predictions = pipeline.predict(development_text)
            probabilities = pipeline.predict_proba(development_text)
            classes = pipeline.classes_.tolist()
            k = min(3, len(classes))
            top_indices = np.argsort(probabilities, axis=1)[:, -k:]
            top_labels = np.asarray(classes)[top_indices]
            metrics["top_3_accuracy"] = float(
                (top_labels == development["category"].to_numpy()[:, None]).any(axis=1).mean()
            )
            metrics["top_k_used"] = k
            artifact_path = artifact_dir / f"{family}.joblib"
            joblib.dump(pipeline, artifact_path, compress=3)
            checksum = _sha256_file(artifact_path)
            diagnostics = _write_diagnostics(
                development, predictions, probabilities, labels, report_dir / family
            )
            models[family] = {
                "path": artifact_path.name,
                "sha256": checksum,
                "model_version": f"{family}-{checksum[:12]}",
                "classes": classes,
                "parameters": {"tfidf": TFIDF_PARAMETERS, "classifier": parameters},
                "versions": versions,
                "development_metrics": metrics,
                "training_seconds": training_seconds,
                "search_training_seconds": search_training_seconds,
                "artifact_size_mb": artifact_path.stat().st_size / 1_000_000,
                "vocabulary_features": len(pipeline.named_steps["tfidf"].vocabulary_),
                "latency": _measure_latency(pipeline, development_text),
                "diagnostics": diagnostics,
            }
    selected = max(
        models,
        key=lambda name: (
            models[name]["development_metrics"]["macro_f1"],
            models[name]["development_metrics"]["accuracy"],
        ),
    )
    metadata = {
        "schema_version": 1,
        "selected_model": selected,
        "classes": models[selected]["classes"],
        "label_mapping": mapping,
        "versions": versions,
        "data_checksums": data_checksums,
        "data_counts": {"fit": len(fit), "development": len(development)},
        "models": models,
        "selection": {
            "partition": "development",
            "primary_metric": "macro_f1",
            "tie_breaker": "accuracy; then first candidate/family in declared grid order",
            "fit_partitions": ["fit"],
            "thread_limit": 1,
        },
    }
    _write_json(artifact_dir / "metadata.json", metadata)
    report = {
        **metadata,
        "majority_baseline": majority,
        "candidates": candidates,
        "notes": [
            "All vectorizers and classifiers fit only fit.csv; no fit+development refit occurs.",
            "Development scores guide selection and are not final test estimates.",
            "Calibration, policy, challenge, and official test data were not opened.",
            "Probabilities are uncalibrated; no automatic-routing threshold is defined.",
            "Latency depends on this machine; artifact MB uses 1,000,000 bytes.",
        ],
    }
    _write_json(report_dir / "report.json", report)
    pd.DataFrame(candidates).to_csv(report_dir / "candidates.csv", index=False, lineterminator="\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/banking77"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/baselines"))
    args = parser.parse_args(argv)
    try:
        report = train_baselines(args.data_dir, args.artifact_dir, args.report_dir)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.exit(1, f"Baseline training failed: {error}\n")
    print(
        json.dumps(
            {
                "selected_model": report["selected_model"],
                "models": {
                    name: {
                        "development_accuracy": model["development_metrics"]["accuracy"],
                        "development_macro_f1": model["development_metrics"]["macro_f1"],
                    }
                    for name, model in report["models"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
