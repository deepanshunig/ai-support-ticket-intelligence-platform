"""Audit Banking77 and create reproducible, grouped training partitions.

The official test split is copied to a separate artifact, never used to select
the split seed, remove training records, or make model/policy decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PARTITIONS = ("fit", "development", "calibration", "policy")
REQUIRED_COLUMNS = ("text", "category")
ARTIFACT_COLUMNS = (
    "row_id",
    "source_split",
    "source_row",
    "text",
    "category",
    "label_id",
    "group_hash",
)


def normalize_text(text: str) -> str:
    """Define duplicate groups without changing the text used by future models."""
    return " ".join(text.lower().split())


def _text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a local artifact without loading the complete file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_categories(path: Path, expected_count: int | None = 77) -> list[str]:
    """Preserve the publisher's category order as the integer-label mapping."""
    categories = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(categories, list) or not categories:
        raise ValueError("categories.json must contain a nonempty ordered list.")
    if any(not isinstance(label, str) or not label.strip() for label in categories):
        raise ValueError("Every category must be a nonempty string.")
    if len(set(categories)) != len(categories):
        raise ValueError("categories.json contains duplicate labels.")
    if expected_count is not None and len(categories) != expected_count:
        raise ValueError(f"Expected {expected_count} categories; found {len(categories)}.")
    return categories


def _check_columns(frame: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}.")


def _normalized_series(frame: pd.DataFrame) -> pd.Series:
    return frame["text"].map(lambda text: normalize_text(text) if isinstance(text, str) else "")


def _length_summary(values: pd.Series) -> dict[str, float | int]:
    if values.empty:
        return {}
    return {
        "minimum": int(values.min()),
        "maximum": int(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p05": float(values.quantile(0.05)),
        "p95": float(values.quantile(0.95)),
    }


def audit_frame(frame: pd.DataFrame, categories: list[str]) -> dict[str, Any]:
    """Summarize records, including duplicate groups with conflicting labels."""
    _check_columns(frame)
    normalized = _normalized_series(frame)
    valid_text = frame["text"].map(lambda value: isinstance(value, str) and bool(value.strip()))
    valid_category = frame["category"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )
    group_sizes = normalized.value_counts()
    grouped = pd.DataFrame({"normalized": normalized, "category": frame["category"]})
    conflicting = grouped.groupby("normalized")["category"].nunique(dropna=False)
    conflicting_keys = conflicting[conflicting > 1].index
    conflict_records = []
    for text in conflicting_keys:
        records = grouped.loc[grouped["normalized"] == text]
        conflict_records.append(
            {
                "group_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "records": int(len(records)),
                "labels": sorted(str(label) for label in records["category"].unique()),
            }
        )
    lengths = frame.loc[valid_text, "text"].str.len()
    words = normalized.loc[valid_text].str.split().str.len()
    observed = set(frame.loc[valid_category, "category"])
    return {
        "records": int(len(frame)),
        "empty_or_invalid_text_records": int((~valid_text).sum()),
        "empty_or_invalid_category_records": int((~valid_category).sum()),
        "exact_duplicate_text_excess_records": int(frame["text"].duplicated().sum()),
        "normalized_unique_groups": int(len(group_sizes)),
        "normalized_duplicate_groups": int((group_sizes > 1).sum()),
        "normalized_duplicate_excess_records": int(len(frame) - len(group_sizes)),
        "conflicting_label_groups": int(len(conflict_records)),
        "conflicting_label_details": conflict_records,
        "unknown_labels": sorted(observed - set(categories)),
        "missing_labels": [label for label in categories if label not in observed],
        "class_counts": {
            label: int((frame["category"] == label).sum()) for label in categories
        },
        "character_length": _length_summary(lengths),
        "word_length": _length_summary(words),
    }


def _validate_frame(frame: pd.DataFrame, categories: list[str], context: str) -> None:
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("The category mapping must be nonempty and contain unique labels.")
    audit = audit_frame(frame, categories)
    errors = []
    if audit["empty_or_invalid_text_records"]:
        errors.append(f"empty or invalid text: {audit['empty_or_invalid_text_records']}")
    if audit["empty_or_invalid_category_records"]:
        errors.append(f"empty or invalid categories: {audit['empty_or_invalid_category_records']}")
    if audit["unknown_labels"]:
        errors.append(f"unknown labels: {audit['unknown_labels']}")
    if audit["missing_labels"]:
        errors.append(f"missing labels: {audit['missing_labels']}")
    if errors:
        raise ValueError(f"Invalid {context} records: {'; '.join(errors)}.")


def _with_metadata(
    frame: pd.DataFrame, categories: list[str], source_split: str
) -> pd.DataFrame:
    records = frame.loc[:, list(REQUIRED_COLUMNS)].reset_index(drop=True).copy()
    records["source_row"] = np.arange(len(records), dtype=np.int64)
    records["source_split"] = source_split
    records["row_id"] = records["source_row"].map(lambda row: f"{source_split}:{row:06d}")
    records["label_id"] = records["category"].map(
        {label: index for index, label in enumerate(categories)}
    )
    records["group_hash"] = records["text"].map(_text_hash)
    return records.loc[:, list(ARTIFACT_COLUMNS)]


def split_training_frame(
    frame: pd.DataFrame, categories: list[str], *, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Assign folds 0/1/2 to dev/calibration/policy and the remaining seven to fit."""
    _validate_frame(frame, categories, "training")
    records = _with_metadata(frame, categories, "train")
    group_counts = records.groupby("category")["group_hash"].nunique()
    if (group_counts < 10).any():
        sparse = group_counts[group_counts < 10].to_dict()
        raise ValueError(f"Ten-fold grouping requires at least 10 groups per label: {sparse}.")
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fold_assignment = np.full(len(records), -1, dtype=np.int64)
    for fold, (_, held_out) in enumerate(
        splitter.split(records, records["label_id"], groups=records["group_hash"])
    ):
        fold_assignment[held_out] = fold
    assignments = {
        "development": fold_assignment == 0,
        "calibration": fold_assignment == 1,
        "policy": fold_assignment == 2,
        "fit": fold_assignment >= 3,
    }
    result = {name: records.loc[assignments[name]].copy() for name in PARTITIONS}
    seen_groups: set[str] = set()
    for name, partition in result.items():
        _validate_frame(partition, categories, name)
        groups = set(partition["group_hash"])
        if seen_groups & groups:
            raise ValueError(f"Normalized-text groups overlap in {name}.")
        seen_groups.update(groups)
    if sum(len(partition) for partition in result.values()) != len(records):
        raise ValueError("The generated partitions do not cover all training records.")
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_plots(frame: pd.DataFrame, categories: list[str], report_dir: Path) -> list[Path]:
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    counts = frame["category"].value_counts().reindex(categories, fill_value=0)
    fig, axis = plt.subplots(figsize=(12, max(4, len(categories) * 0.23)))
    axis.barh(categories, counts.to_numpy())
    axis.invert_yaxis()
    axis.set_xlabel("Official training records")
    axis.set_title("Banking77 training class counts (publisher label order)")
    fig.tight_layout()
    class_plot = report_dir / "training_class_counts.png"
    fig.savefig(class_plot, dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    lengths = (frame["text"].str.len(), _normalized_series(frame).str.split().str.len())
    for axis, values, unit in zip(axes, lengths, ("Characters", "Words"), strict=True):
        axis.hist(values, bins=40, edgecolor="white")
        axis.set_xlabel(unit)
        axis.set_ylabel("Official training records")
    fig.suptitle("Banking77 training text lengths")
    fig.tight_layout()
    length_plot = report_dir / "training_text_lengths.png"
    fig.savefig(length_plot, dpi=140)
    plt.close(fig)
    return [class_plot, length_plot]


def prepare_dataset(
    raw_dir: Path,
    output_dir: Path,
    report_dir: Path,
    *,
    seed: int = 42,
    expected_label_count: int | None = 77,
) -> dict[str, Any]:
    """Validate local raw files and write auditable data artifacts, without training."""
    raw_dir, output_dir, report_dir = (
        Path(raw_dir).resolve(),
        Path(output_dir).resolve(),
        Path(report_dir).resolve(),
    )
    if output_dir.is_relative_to(raw_dir) or report_dir.is_relative_to(raw_dir):
        raise ValueError("Output and report directories must be outside the raw directory.")
    categories = load_categories(raw_dir / "categories.json", expected_label_count)
    train = pd.read_csv(raw_dir / "train.csv", dtype=str, keep_default_na=False)
    test = pd.read_csv(raw_dir / "test.csv", dtype=str, keep_default_na=False)
    _validate_frame(train, categories, "official training")
    _validate_frame(test, categories, "official test integrity")
    partitions = split_training_frame(train, categories, seed=seed)
    official_test = _with_metadata(test, categories, "test")
    training_hashes = set(train["text"].map(_text_hash))
    test_hashes = set(official_test["group_hash"])
    cross_split_hashes = training_hashes & test_hashes
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    manifest_parts = []
    all_partitions = {**partitions, "official_test": official_test}
    for name, records in all_partitions.items():
        artifact = output_dir / f"{name}.csv"
        records.to_csv(artifact, index=False, lineterminator="\n")
        output_files.append(artifact)
        manifest_parts.append(
            records.drop(columns="text").assign(partition=name)
        )
    manifest_path = output_dir / "split_manifest.csv"
    pd.concat(manifest_parts, ignore_index=True).to_csv(
        manifest_path, index=False, lineterminator="\n"
    )
    output_files.append(manifest_path)
    mapping = [{"label_id": index, "category": label} for index, label in enumerate(categories)]
    mapping_path = output_dir / "label_mapping.json"
    _write_json(mapping_path, mapping)
    output_files.append(mapping_path)
    plot_files = _write_plots(train, categories, report_dir)
    source_checksums = {
        path.name: sha256_file(path)
        for path in sorted(raw_dir.iterdir())
        if path.is_file()
    }
    audit = {
        "schema_version": 1,
        "seed": seed,
        "split_algorithm": "StratifiedGroupKFold(n_splits=10, shuffle=True)",
        "fold_policy": {
            "development": [0],
            "calibration": [1],
            "policy": [2],
            "fit": list(range(3, 10)),
        },
        "normalization": "lowercase; collapse whitespace; strip surrounding whitespace",
        "label_mapping": mapping,
        "source_checksums_sha256": source_checksums,
        "package_versions": {
            name: version(name) for name in ("pandas", "numpy", "matplotlib", "scikit-learn")
        },
        "official_train": audit_frame(train, categories),
        "official_test_integrity": {
            "records": int(len(test)),
            "all_expected_labels_present": True,
            "invalid_records": 0,
        },
        "reference_counts": {
            "published_train": 10003,
            "published_test": 3080,
            "matches_published_counts": len(train) == 10003 and len(test) == 3080,
        },
        "partitions": {
            name: {
                "records": int(len(records)),
                "unique_groups": int(records["group_hash"].nunique()),
                "class_counts": {
                    label: int((records["category"] == label).sum()) for label in categories
                },
            }
            for name, records in partitions.items()
        },
        "generated_partition_group_overlap": 0,
        "official_train_test_hash_overlap": {
            "unique_groups": len(cross_split_hashes),
            "training_records": int(train["text"].map(_text_hash).isin(cross_split_hashes).sum()),
            "test_records": int(official_test["group_hash"].isin(cross_split_hashes).sum()),
            "group_hashes": sorted(cross_split_hashes),
            "handling": "Reported only; raw records and official test split remain unchanged.",
        },
        "notes": [
            "Fold sizes are approximate because duplicate groups remain indivisible.",
            "Conflicting-label duplicate groups are retained together and reported.",
            "Official test labels are inspected only for schema/mapping integrity.",
            "No model, calibration, threshold selection, or bank action is implemented here.",
        ],
    }
    audit_path = report_dir / "audit.json"
    _write_json(audit_path, audit)
    checksums = {
        "algorithm": "sha256",
        "seed": seed,
        "raw_files": source_checksums,
        "processed_files": {path.name: sha256_file(path) for path in output_files},
        "report_files": {path.name: sha256_file(path) for path in [*plot_files, audit_path]},
    }
    _write_json(output_dir / "checksums.json", checksums)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/banking77"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/banking77"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-label-count", type=int, default=77)
    args = parser.parse_args(argv)
    try:
        audit = prepare_dataset(
            args.raw_dir,
            args.output_dir,
            args.report_dir,
            seed=args.seed,
            expected_label_count=args.expected_label_count,
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.exit(1, f"Dataset preparation failed: {error}\n")
    counts = {name: result["records"] for name, result in audit["partitions"].items()}
    print(json.dumps({"seed": audit["seed"], "partition_counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
